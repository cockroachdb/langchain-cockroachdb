"""A batch that fails before its statements run leaves no phantom prepared statement.

psycopg prepares a statement the sixth time a connection runs it. Five is
psycopg's default prepare_threshold, and the default of every
from_conn_string here. In pipeline mode psycopg records the statement as
prepared as soon as it sends it, without waiting for the server's answer.
When an earlier statement in the same batch fails, the server skips the rest
of the batch, so the statement is never prepared, but psycopg used it by name
from then on. Each later run failed with "prepared statement ... does not
exist" (SQLSTATE 26000). A checkpoint write failed although its retry should
have saved it. A store read failed on that connection until the connection
closed, because no rollback cleared psycopg's cache. A demo found it when
CockroachDB's license limit failed BEGIN.

Each test runs an operation five times on one connection. On the sixth run a
retryable error fails the batch before the statement that run would prepare.
The retry and the run after it must succeed, and the connection must keep
the prepare_threshold it came with.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from psycopg import AsyncConnection, Connection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import (
    AsyncCockroachDBSaver,
    AsyncCockroachDBStore,
    CockroachDBSaver,
    CockroachDBStore,
    RetryPolicy,
)
from tests.integration.faults import PsycopgFaults

FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)
MODES = ("pool", "connection", "pipeline")
# psycopg's default prepare_threshold, and every from_conn_string's: a
# connection prepares a statement on its sixth run
THRESHOLD = 5
# A fault just before BEGIN fails the batch before its transaction opens
BEFORE_BEGIN = "begin"
# The store's get, a read that runs in a pipeline without BEGIN
STORE_GET = "select key, value, created_at, updated_at"
NAMESPACE = ("user", "a")


@pytest.fixture
def table_name(libpq_url: str) -> Iterator[str]:
    """A fresh store table name; its tables are dropped after the test."""
    name = f"store_{uuid.uuid4().hex[:10]}"
    yield name
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        for suffix in ("_vectors", "", "_migrations"):
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(name + suffix)))


def config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def checkpoint_with_blobs() -> tuple[Checkpoint, dict[str, str]]:
    """A checkpoint whose put sends BEGIN, a blob row, the checkpoint row and COMMIT."""
    checkpoint = empty_checkpoint()
    versions = {"messages": "1"}
    checkpoint["channel_values"] = {"messages": ["hello"]}
    checkpoint["channel_versions"] = dict(versions)
    return checkpoint, versions


def threshold_of(conn: Connection[Any] | ConnectionPool[Any]) -> int | None:
    """The prepare_threshold of a component's connection, the only one in its pool."""
    if isinstance(conn, ConnectionPool):
        with conn.connection() as pooled:
            return pooled.prepare_threshold
    return conn.prepare_threshold


async def athreshold_of(conn: AsyncConnection[Any] | AsyncConnectionPool[Any]) -> int | None:
    """The async version of threshold_of."""
    if isinstance(conn, AsyncConnectionPool):
        async with conn.connection() as pooled:
            return pooled.prepare_threshold
    return conn.prepare_threshold


def pool_kwargs(cursor_factory: Any) -> dict[str, Any]:
    """Connection options as an application passes them: psycopg's own prepare_threshold."""
    return {"autocommit": True, "row_factory": dict_row, "cursor_factory": cursor_factory}


@contextlib.contextmanager
def saver_for(mode: str, libpq_url: str, faults: PsycopgFaults) -> Iterator[CockroachDBSaver]:
    """A sync saver that runs every operation, and every retry, on one connection."""
    if mode == "pool":
        kwargs = pool_kwargs(faults.cursor_class())
        with ConnectionPool(libpq_url, min_size=1, max_size=1, kwargs=kwargs, open=True) as pool:
            yield CockroachDBSaver(pool, retry_policy=FAST)
        return
    with CockroachDBSaver.from_conn_string(
        libpq_url,
        pipeline=mode == "pipeline",
        retry_policy=FAST,
        connect_kwargs={"cursor_factory": faults.cursor_class()},
    ) as saver:
        yield saver


@contextlib.asynccontextmanager
async def asaver_for(
    mode: str, libpq_url: str, faults: PsycopgFaults
) -> AsyncIterator[AsyncCockroachDBSaver]:
    """The async version of saver_for."""
    if mode == "pool":
        kwargs = pool_kwargs(faults.async_cursor_class())
        async with AsyncConnectionPool(
            libpq_url, min_size=1, max_size=1, kwargs=kwargs, open=False
        ) as pool:
            yield AsyncCockroachDBSaver(pool, retry_policy=FAST)
        return
    async with AsyncCockroachDBSaver.from_conn_string(
        libpq_url,
        pipeline=mode == "pipeline",
        retry_policy=FAST,
        connect_kwargs={"cursor_factory": faults.async_cursor_class()},
    ) as saver:
        yield saver


@contextlib.contextmanager
def store_for(
    mode: str, libpq_url: str, table_name: str, faults: PsycopgFaults
) -> Iterator[CockroachDBStore]:
    """A sync store that runs every operation, and every retry, on one connection."""
    with CockroachDBStore.from_conn_string(
        libpq_url,
        pipeline=mode == "pipeline",
        pool_config={"min_size": 1, "max_size": 1} if mode == "pool" else None,
        connect_kwargs={"cursor_factory": faults.cursor_class()},
        retry_policy=FAST,
        table_name=table_name,
    ) as store:
        yield store


@contextlib.asynccontextmanager
async def astore_for(
    mode: str, libpq_url: str, table_name: str, faults: PsycopgFaults
) -> AsyncIterator[AsyncCockroachDBStore]:
    """The async version of store_for."""
    async with AsyncCockroachDBStore.from_conn_string(
        libpq_url,
        pipeline=mode == "pipeline",
        pool_config={"min_size": 1, "max_size": 1} if mode == "pool" else None,
        connect_kwargs={"cursor_factory": faults.async_cursor_class()},
        retry_policy=FAST,
        table_name=table_name,
    ) as store:
        yield store


@pytest.mark.parametrize("mode", MODES)
def test_sync_saver_put_survives_a_batch_that_fails_before_begin(
    libpq_url: str, unsafe_internals: bool, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    thread = str(uuid.uuid4())
    with saver_for(mode, libpq_url, faults) as saver:
        saver.setup()

        def put() -> None:
            checkpoint, versions = checkpoint_with_blobs()
            saver.put(config(thread), checkpoint, {}, versions)

        for _ in range(THRESHOLD):
            put()
        faults.fail_next("40001", match=BEFORE_BEGIN)
        put()
        put()
        assert faults.injected == ["40001"]
        assert len(list(saver.list(config(thread)))) == 7
        saver.delete_thread(thread)
        assert threshold_of(saver.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
async def test_async_saver_put_survives_a_batch_that_fails_before_begin(
    libpq_url: str, unsafe_internals: bool, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    thread = str(uuid.uuid4())
    async with asaver_for(mode, libpq_url, faults) as saver:
        await saver.setup()

        async def put() -> None:
            checkpoint, versions = checkpoint_with_blobs()
            await saver.aput(config(thread), checkpoint, {}, versions)

        for _ in range(THRESHOLD):
            await put()
        faults.fail_next("40001", match=BEFORE_BEGIN)
        await put()
        await put()
        assert faults.injected == ["40001"]
        assert len([c async for c in saver.alist(config(thread))]) == 7
        await saver.adelete_thread(thread)
        assert await athreshold_of(saver.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
def test_sync_saver_delete_thread_survives_a_batch_that_fails_before_begin(
    libpq_url: str, unsafe_internals: bool, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    thread = str(uuid.uuid4())
    with saver_for(mode, libpq_url, faults) as saver:
        saver.setup()
        for _ in range(THRESHOLD):
            saver.delete_thread(thread)
        checkpoint, versions = checkpoint_with_blobs()
        saver.put(config(thread), checkpoint, {}, versions)
        faults.fail_next("40001", match=BEFORE_BEGIN)
        saver.delete_thread(thread)
        assert saver.get_tuple(config(thread)) is None
        saver.delete_thread(thread)
        assert faults.injected == ["40001"]
        assert threshold_of(saver.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
async def test_async_saver_delete_thread_survives_a_batch_that_fails_before_begin(
    libpq_url: str, unsafe_internals: bool, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    thread = str(uuid.uuid4())
    async with asaver_for(mode, libpq_url, faults) as saver:
        await saver.setup()
        for _ in range(THRESHOLD):
            await saver.adelete_thread(thread)
        checkpoint, versions = checkpoint_with_blobs()
        await saver.aput(config(thread), checkpoint, {}, versions)
        faults.fail_next("40001", match=BEFORE_BEGIN)
        await saver.adelete_thread(thread)
        assert await saver.aget_tuple(config(thread)) is None
        await saver.adelete_thread(thread)
        assert faults.injected == ["40001"]
        assert await athreshold_of(saver.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
def test_sync_store_get_survives_a_batch_that_fails_before_its_read(
    libpq_url: str, unsafe_internals: bool, table_name: str, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    with store_for(mode, libpq_url, table_name, faults) as store:
        store.setup()
        store.put(NAMESPACE, "k", {"v": 1})
        for _ in range(THRESHOLD):
            assert store.get(NAMESPACE, "k").value == {"v": 1}
        faults.fail_next("40001", match=STORE_GET)
        assert store.get(NAMESPACE, "k").value == {"v": 1}
        assert store.get(NAMESPACE, "k").value == {"v": 1}
        assert faults.injected == ["40001"]
        assert threshold_of(store.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
async def test_async_store_get_survives_a_batch_that_fails_before_its_read(
    libpq_url: str, unsafe_internals: bool, table_name: str, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with astore_for(mode, libpq_url, table_name, faults) as store:
        await store.setup()
        await store.aput(NAMESPACE, "k", {"v": 1})
        for _ in range(THRESHOLD):
            assert (await store.aget(NAMESPACE, "k")).value == {"v": 1}
        faults.fail_next("40001", match=STORE_GET)
        assert (await store.aget(NAMESPACE, "k")).value == {"v": 1}
        assert (await store.aget(NAMESPACE, "k")).value == {"v": 1}
        assert faults.injected == ["40001"]
        assert await athreshold_of(store.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
def test_sync_store_put_survives_a_batch_that_fails_before_begin(
    libpq_url: str, unsafe_internals: bool, table_name: str, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    with store_for(mode, libpq_url, table_name, faults) as store:
        store.setup()
        for value in range(THRESHOLD):
            store.put(NAMESPACE, "k", {"v": value})
        faults.fail_next("40001", match=BEFORE_BEGIN)
        store.put(NAMESPACE, "k", {"v": 5})
        assert store.get(NAMESPACE, "k").value == {"v": 5}
        store.put(NAMESPACE, "k", {"v": 6})
        assert store.get(NAMESPACE, "k").value == {"v": 6}
        assert faults.injected == ["40001"]
        assert threshold_of(store.conn) == THRESHOLD


@pytest.mark.parametrize("mode", MODES)
async def test_async_store_put_survives_a_batch_that_fails_before_begin(
    libpq_url: str, unsafe_internals: bool, table_name: str, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with astore_for(mode, libpq_url, table_name, faults) as store:
        await store.setup()
        for value in range(THRESHOLD):
            await store.aput(NAMESPACE, "k", {"v": value})
        faults.fail_next("40001", match=BEFORE_BEGIN)
        await store.aput(NAMESPACE, "k", {"v": 5})
        assert (await store.aget(NAMESPACE, "k")).value == {"v": 5}
        await store.aput(NAMESPACE, "k", {"v": 6})
        assert (await store.aget(NAMESPACE, "k")).value == {"v": 6}
        assert faults.injected == ["40001"]
        assert await athreshold_of(store.conn) == THRESHOLD
