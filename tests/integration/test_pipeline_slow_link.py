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
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.store.base import PutOp
from psycopg import AsyncConnection, AsyncPipeline, Connection, Pipeline

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

# How long a write may take. Through the proxy each one takes a second or two
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


@pytest.fixture
def table_name(libpq_url: str) -> Iterator[str]:
    """A fresh items table name, whose tables are dropped after the test."""
    name = f"store_{uuid.uuid4().hex[:10]}"
    yield name
    drop_store_tables(libpq_url, name)


def index_config() -> dict[str, Any]:
    return {"dims": DIMS, "embed": DeterministicFakeEmbedding(size=DIMS), "fields": ["text"]}


def put_ops() -> list[PutOp]:
    return [PutOp(NAMESPACE, f"k{i:04}", {"text": f"memory {i}"}) for i in range(ITEMS)]


def large_writes() -> list[tuple[str, str]]:
    """Checkpoint writes whose values do not compress."""
    return [(f"channel{i}", os.urandom(WRITE_BYTES // 2).hex()) for i in range(WRITES)]


def config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def store_rows(libpq_url: str, table_name: str) -> tuple[int, int]:
    """Item rows and vector rows of the store, read without the proxy."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        counts = [
            conn.execute(f"SELECT count(*) FROM {table_name}{suffix}").fetchone()
            for suffix in ("", "_vectors")
        ]
    return counts[0][0], counts[1][0]  # type: ignore[index]


def write_rows(libpq_url: str, thread: str) -> int:
    """Rows of ``thread`` in checkpoint_writes, read without the proxy."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s", (thread,)
        ).fetchone()
    assert row is not None
    return row[0]


@contextlib.contextmanager
def pipelined(libpq_url: str, proxy: SlowProxy) -> Iterator[tuple[Connection[Any], Pipeline]]:
    """A single connection through ``proxy`` with a small send buffer, in pipeline mode."""
    options = connection_options(libpq_url, proxy.connect_kwargs, 10.0, 5)
    with Connection.connect(libpq_url, **options) as conn:
        limit_send_buffer(conn, SEND_BUFFER)
        with conn.pipeline() as pipe:
            yield conn, pipe


@contextlib.asynccontextmanager
async def apipelined(
    libpq_url: str, proxy: SlowProxy
) -> AsyncIterator[tuple[AsyncConnection[Any], AsyncPipeline]]:
    """The async version of pipelined."""
    options = connection_options(libpq_url, proxy.connect_kwargs, 10.0, 5)
    async with await AsyncConnection.connect(libpq_url, **options) as conn:
        limit_send_buffer(conn, SEND_BUFFER)
        async with conn.pipeline() as pipe:
            yield conn, pipe


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
    with SlowProxy.to(libpq_url) as proxy, pipelined(libpq_url, proxy) as (conn, pipe):
        saver = CockroachDBSaver(conn, pipe, retry_policy=ONE_ATTEMPT)
        saver.setup()
        saved = saver.put(config(thread), empty_checkpoint(), {}, {})
        writes = large_writes()
        finish_within(proxy, lambda: saver.put_writes(saved, writes, task_id="t1"))
        found = saver.get_tuple(saved)
        assert found is not None
        assert {channel: value for _, channel, value in found.pending_writes} == dict(writes)
        assert proxy.connections == 1
    assert write_rows(libpq_url, thread) == WRITES


@pytest.mark.timeout(120)
async def test_async_saver_put_writes_finishes_on_a_slow_link(libpq_url: str) -> None:
    thread = str(uuid.uuid4())
    with SlowProxy.to(libpq_url) as proxy:
        async with apipelined(libpq_url, proxy) as (conn, pipe):
            saver = AsyncCockroachDBSaver(conn, pipe, retry_policy=ONE_ATTEMPT)
            await saver.setup()
            saved = await saver.aput(config(thread), empty_checkpoint(), {}, {})
            writes = large_writes()
            await afinish_within(proxy, saver.aput_writes(saved, writes, task_id="t1"))
            found = await saver.aget_tuple(saved)
            assert found is not None
            assert {channel: value for _, channel, value in found.pending_writes} == dict(writes)
            assert proxy.connections == 1
    assert write_rows(libpq_url, thread) == WRITES
