"""Writes on a component-wide pipeline finish on a slow link.

A saver or store given ``pipe=``, or made with ``from_conn_string(...,
pipeline=True)``, ends each write with psycopg's ``Pipeline.sync()``. That
sync flushes libpq's output buffer once and then waits for results. When the
socket cannot take the rest of the batch at that moment, the end of the batch
and its Sync stay in the buffer: the server waits for them, and the client
waits for the server. A TLS run against a CockroachDB Cloud cluster found it.

SlowProxy brings that link to a local cluster. It passes the client's bytes
on a few KB at a time, and the client's socket gets a small fixed send
buffer, so a write of about a megabyte cannot leave in one flush. Each write
must finish within DEADLINE seconds. One that does not is stopped by cutting
the proxy, and the test fails.

The components now send what libpq still holds as each operation starts and
before each sync. On a link that went dead that wait cannot end, and the
hang guard must cut it off like any other.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.store.base import PutOp
from psycopg import AsyncConnection, AsyncPipeline, Connection, Pipeline, sql

from langchain_cockroachdb import (
    AsyncCockroachDBSaver,
    AsyncCockroachDBStore,
    CockroachDBSaver,
    CockroachDBStore,
    RetryPolicy,
)
from langchain_cockroachdb._connection import connection_options
from tests.integration.faults import SlowProxy, limit_send_buffer
from tests.integration.store.conftest import drop_store_tables

T = TypeVar("T")

# How long a write may take. Through the proxy each one takes about a second
DEADLINE = 20.0
SEND_BUFFER = 32 * 1024
# 200 items with 256 dimensions: two vector statements of about 500 KB each
DIMS = 256
ITEMS = 200
NAMESPACE = ("user", "alice")
# Five checkpoint writes of 200 KB each
WRITES = 5
WRITE_BYTES = 200_000
# One attempt, so a failed write shows up as an error instead of a retry
ONE_ATTEMPT = RetryPolicy(max_attempts=1)
# The hang guard's attempt timeout: the write reaches its drain well within it
GUARD = 5.0
GUARDED = RetryPolicy(max_attempts=1, attempt_timeout=GUARD)
ABANDONED = "attempt timeout; abandoning its connection"


@pytest.fixture
def table_name(libpq_url: str) -> Iterator[str]:
    """A fresh items table name, whose tables are dropped after the test."""
    name = f"store_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        drop_store_tables(libpq_url, name)
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"store table cleanup failed: {error!r}", UserWarning, stacklevel=1)


def index_config() -> dict[str, Any]:
    return {"dims": DIMS, "embed": DeterministicFakeEmbedding(size=DIMS), "fields": ["text"]}


def put_ops() -> list[PutOp]:
    return [PutOp(NAMESPACE, f"k{i:04}", {"text": f"memory {i}"}) for i in range(ITEMS)]


def large_writes() -> list[tuple[str, str]]:
    """WRITES checkpoint writes of WRITE_BYTES characters each."""
    return [(f"channel{i}", os.urandom(WRITE_BYTES // 2).hex()) for i in range(WRITES)]


def config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def store_rows(libpq_url: str, table_name: str) -> tuple[int, int]:
    """Item rows and vector rows of the store, read without the proxy."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        items, vectors = (
            conn.execute(f"SELECT count(*) FROM {table_name}{suffix}").fetchone()
            for suffix in ("", "_vectors")
        )
    assert items is not None and vectors is not None
    return items[0], vectors[0]


def write_rows(libpq_url: str, thread: str) -> int:
    """Rows of ``thread`` in checkpoint_writes, read without the proxy."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s", (thread,)
        ).fetchone()
    assert row is not None
    return row[0]


def delete_thread(libpq_url: str, thread: str) -> None:
    """Delete the checkpoint rows of ``thread`` without the proxy; warn if that fails.

    The checkpoint tables are shared, so rows left behind pile up in a
    database that is not thrown away after the run.
    """
    try:
        with psycopg.connect(libpq_url, autocommit=True) as conn:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                conn.execute(
                    sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                    (thread,),
                )
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"checkpoint row cleanup failed: {error!r}", UserWarning, stacklevel=1)


@contextlib.contextmanager
def pipelined(libpq_url: str, proxy: SlowProxy) -> Iterator[tuple[Connection[Any], Pipeline]]:
    """A single connection through ``proxy`` with a small send buffer, in pipeline mode.

    When the block finished after its connection was cut, leaving the
    pipeline fails too. That error is dropped: the block expected the cut.
    """
    options = connection_options(libpq_url, proxy.connect_kwargs, 10.0, 5)
    with Connection.connect(libpq_url, **options) as conn:
        limit_send_buffer(conn, SEND_BUFFER)
        finished = False
        try:
            with conn.pipeline() as pipe:
                yield conn, pipe
                finished = True
        except psycopg.OperationalError:
            if not (finished and conn.closed):
                raise


@contextlib.asynccontextmanager
async def apipelined(
    libpq_url: str, proxy: SlowProxy
) -> AsyncIterator[tuple[AsyncConnection[Any], AsyncPipeline]]:
    """The async version of pipelined."""
    options = connection_options(libpq_url, proxy.connect_kwargs, 10.0, 5)
    async with await AsyncConnection.connect(libpq_url, **options) as conn:
        limit_send_buffer(conn, SEND_BUFFER)
        finished = False
        try:
            async with conn.pipeline() as pipe:
                yield conn, pipe
                finished = True
        except psycopg.OperationalError:
            if not (finished and conn.closed):
                raise


def finish_within(proxy: SlowProxy, call: Callable[[], T]) -> T:
    """Return ``call()``, or fail if it is still running after DEADLINE seconds.

    A call that hangs is stopped by cutting the proxy, which makes it fail,
    so its thread ends.
    """
    results: list[T] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(call())
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(DEADLINE)
    if worker.is_alive():
        proxy.close()
        worker.join(DEADLINE)
        pytest.fail(f"the write was still waiting for the server after {DEADLINE} s")
    if errors:
        raise errors[0]
    return results[0]


async def afinish_within(proxy: SlowProxy, call: Awaitable[T]) -> T:
    """The async version of finish_within."""
    task = asyncio.ensure_future(call)
    done, _ = await asyncio.wait({task}, timeout=DEADLINE)
    if not done:
        proxy.close()
        done, _ = await asyncio.wait({task}, timeout=DEADLINE)
        if done:
            # The error the cut caused, retrieved so asyncio does not log it
            task.exception()
        else:
            task.cancel()
        pytest.fail(f"the write was still waiting for the server after {DEADLINE} s")
    return task.result()


@pytest.mark.timeout(120)
def test_sync_store_put_finishes_on_a_slow_link(libpq_url: str, table_name: str) -> None:
    with SlowProxy.to(libpq_url) as proxy, pipelined(libpq_url, proxy) as (conn, pipe):
        store = CockroachDBStore(
            conn, pipe=pipe, index=index_config(), table_name=table_name, retry_policy=ONE_ATTEMPT
        )
        store.setup()
        finish_within(proxy, lambda: store.batch(put_ops()))
        # The pipeline is in order: the next call on it works
        assert store.get(NAMESPACE, "k0000") is not None
        assert proxy.connections == 1
    assert store_rows(libpq_url, table_name) == (ITEMS, ITEMS)


@pytest.mark.timeout(120)
async def test_async_store_put_finishes_on_a_slow_link(libpq_url: str, table_name: str) -> None:
    with SlowProxy.to(libpq_url) as proxy:
        async with apipelined(libpq_url, proxy) as (conn, pipe):
            store = AsyncCockroachDBStore(
                conn,
                pipe=pipe,
                index=index_config(),
                table_name=table_name,
                retry_policy=ONE_ATTEMPT,
            )
            await store.setup()
            await afinish_within(proxy, store.abatch(put_ops()))
            assert await store.aget(NAMESPACE, "k0000") is not None
            assert proxy.connections == 1
    assert store_rows(libpq_url, table_name) == (ITEMS, ITEMS)


@pytest.mark.timeout(120)
def test_sync_saver_put_writes_finishes_on_a_slow_link(libpq_url: str) -> None:
    thread = str(uuid.uuid4())
    try:
        with SlowProxy.to(libpq_url) as proxy, pipelined(libpq_url, proxy) as (conn, pipe):
            saver = CockroachDBSaver(conn, pipe, retry_policy=ONE_ATTEMPT)
            saver.setup()
            saved = saver.put(config(thread), empty_checkpoint(), {}, {})
            writes = large_writes()
            finish_within(proxy, lambda: saver.put_writes(saved, writes, task_id="t1"))
            found = saver.get_tuple(saved)
            assert found is not None
            pending = {channel: value for _, channel, value in found.pending_writes}
            assert pending == dict(writes)
            assert proxy.connections == 1
        assert write_rows(libpq_url, thread) == WRITES
    finally:
        delete_thread(libpq_url, thread)


@pytest.mark.timeout(120)
async def test_async_saver_put_writes_finishes_on_a_slow_link(libpq_url: str) -> None:
    thread = str(uuid.uuid4())
    try:
        with SlowProxy.to(libpq_url) as proxy:
            async with apipelined(libpq_url, proxy) as (conn, pipe):
                saver = AsyncCockroachDBSaver(conn, pipe, retry_policy=ONE_ATTEMPT)
                await saver.setup()
                saved = await saver.aput(config(thread), empty_checkpoint(), {}, {})
                writes = large_writes()
                await afinish_within(proxy, saver.aput_writes(saved, writes, task_id="t1"))
                found = await saver.aget_tuple(saved)
                assert found is not None
                pending = {channel: value for _, channel, value in found.pending_writes}
                assert pending == dict(writes)
                assert proxy.connections == 1
        assert write_rows(libpq_url, thread) == WRITES
    finally:
        delete_thread(libpq_url, thread)


class Stall:
    """Stops the proxy when a drain finds bytes to send, and records how each drain ended.

    The drain that starts an operation finds nothing to send, so it passes.
    The drain at the end of the write finds the rest of the batch.
    """

    def __init__(self, proxy: SlowProxy) -> None:
        self.proxy = proxy
        self.ended: list[str] = []
        # Where in ``ended`` the drain that stalled the proxy is
        self.stalled: int | None = None

    def _start(self, conn: Any) -> None:
        if self.stalled is None and conn.pgconn.flush():
            self.stalled = len(self.ended)
            self.proxy.stall()

    def _end(self, error: BaseException | None) -> None:
        self.ended.append("sent" if error is None else type(error).__name__)

    def around(self, drain: Callable[[Any], None]) -> Callable[[Any], None]:
        def stall_then_drain(conn: Any) -> None:
            try:
                self._start(conn)
                drain(conn)
            except BaseException as error:
                self._end(error)
                raise
            self._end(None)

        return stall_then_drain

    def around_async(
        self, drain: Callable[[Any], Awaitable[None]]
    ) -> Callable[[Any], Awaitable[None]]:
        async def stall_then_drain(conn: Any) -> None:
            try:
                self._start(conn)
                await drain(conn)
            except BaseException as error:
                self._end(error)
                raise
            self._end(None)

        return stall_then_drain

    def assert_cut_off_by_the_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        assert self.stalled is not None, "no drain found bytes left to send"
        # That drain never finished: the guard shut its socket after GUARD
        # seconds, long before the test would have cut the proxy
        assert self.ended[self.stalled] == "OperationalError"
        assert ABANDONED in caplog.text


@pytest.mark.timeout(120)
def test_sync_drain_on_a_dead_link_is_cut_off_by_the_hang_guard(
    libpq_url: str, table_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    with SlowProxy.to(libpq_url) as proxy, pipelined(libpq_url, proxy) as (conn, pipe):
        store = CockroachDBStore(
            conn, pipe=pipe, index=index_config(), table_name=table_name, retry_policy=GUARDED
        )
        store.setup()
        stall = Stall(proxy)
        store._drain_output = stall.around(store._drain_output)  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING), pytest.raises(psycopg.OperationalError):
            finish_within(proxy, lambda: store.batch(put_ops()))
        stall.assert_cut_off_by_the_guard(caplog)


@pytest.mark.timeout(120)
async def test_async_drain_on_a_dead_link_is_cut_off_by_the_hang_guard(
    libpq_url: str, table_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    with SlowProxy.to(libpq_url) as proxy:
        async with apipelined(libpq_url, proxy) as (conn, pipe):
            store = AsyncCockroachDBStore(
                conn, pipe=pipe, index=index_config(), table_name=table_name, retry_policy=GUARDED
            )
            await store.setup()
            stall = Stall(proxy)
            store._drain_output = stall.around_async(store._drain_output)  # type: ignore[method-assign]
            with caplog.at_level(logging.WARNING), pytest.raises(psycopg.OperationalError):
                await afinish_within(proxy, store.abatch(put_ops()))
            stall.assert_cut_off_by_the_guard(caplog)
