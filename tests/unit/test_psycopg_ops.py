"""Tests for the psycopg machinery that the checkpointers and the store share."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.waiting import Wait

from tests.unit.checkpointer_fakes import (
    FakeAsyncPipe,
    FakePGconn,
    FakePipe,
    FakeSingleAsyncConnection,
    FakeSingleConnection,
    close_sockets,
)

CHECKPOINTER_CLOSED = (
    "Not retrying: the checkpointer's only connection is closed and cannot be "
    "reopened. Pass a psycopg_pool.{pool} so broken connections are replaced."
)
CHECKPOINTER_STUCK = (
    "a write batch of the checkpointer that stopped partway may still be open "
    "on its connection and could not be rolled back. This happens when the "
    "saver's single connection is shared and a transaction the caller opened "
    "is open on it (psycopg forbids rollback() inside conn.transaction()). "
    "End that transaction, or give the saver a pool."
)
STORE_STUCK = (
    "a write batch of the store that stopped partway may still be open "
    "on its connection and could not be rolled back. This happens when the "
    "store's single connection is shared and a transaction the caller opened "
    "is open on it (psycopg forbids rollback() inside conn.transaction()). "
    "End that transaction, or give the store a pool."
)


def _host(kind: str, component: str, conn: Any) -> Any:
    from langchain_cockroachdb._psycopg_ops import AsyncPsycopgOps, PsycopgOps
    from langchain_cockroachdb.retry import RetryPolicy

    base: type = PsycopgOps if kind == "sync" else AsyncPsycopgOps
    host = type("Host", (base,), {"component": component})()
    host.conn = conn
    host.pipe = None
    host.lock = threading.Lock() if kind == "sync" else asyncio.Lock()
    host.supports_pipeline = True
    host.retry_policy = RetryPolicy()
    host._batch_open = False
    return host


def _closed(kind: str) -> Any:
    cls = FakeSingleConnection if kind == "sync" else FakeSingleAsyncConnection
    conn = cls(None)
    conn._broken = True
    return conn


def _stuck(kind: str) -> Any:
    # A batch of ours is open and its rollback fails, so it stays open
    cls = FakeSingleConnection if kind == "sync" else FakeSingleAsyncConnection
    conn = cls(None, rollback_error=psycopg.ProgrammingError("rollback refused"))
    conn.info.transaction_status = TransactionStatus.INTRANS
    return conn


async def _prepare(host: Any, conn: Any) -> Any:
    result = host._prepare_batch(conn, pipeline=True, transaction=True)
    return await result if asyncio.iscoroutine(result) else result


async def test_component_names_appear_in_messages(caplog: pytest.LogCaptureFixture) -> None:
    for kind in ("sync", "async"):
        pool = "ConnectionPool" if kind == "sync" else "AsyncConnectionPool"
        for component in ("checkpointer", "store"):
            caplog.clear()
            host = _host(kind, component, _closed(kind))
            with caplog.at_level(logging.WARNING):
                assert host._can_retry(psycopg.OperationalError("lost")) is False
            logged = [record.getMessage() for record in caplog.records]
            expected = CHECKPOINTER_CLOSED.format(pool=pool)
            if component == "store":
                expected = expected.replace("the checkpointer's", "the store's")
            assert logged == [expected]

            conn = _stuck(kind)
            host = _host(kind, component, conn)
            host._batch_open = True
            with pytest.raises(psycopg.errors.ActiveSqlTransaction) as info:
                await _prepare(host, conn)
            stuck = CHECKPOINTER_STUCK if component == "checkpointer" else STORE_STUCK
            assert str(info.value) == stuck


async def test_savers_use_the_shared_mixins(caplog: pytest.LogCaptureFixture) -> None:
    from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver
    from langchain_cockroachdb._psycopg_ops import AsyncPsycopgOps, PsycopgOps

    assert issubclass(CockroachDBSaver, PsycopgOps)
    assert issubclass(AsyncCockroachDBSaver, AsyncPsycopgOps)

    # The "Not retrying" warning still comes from each saver's own module logger
    for saver, module in (
        (CockroachDBSaver(_closed("sync")), "langchain_cockroachdb.checkpointer.saver"),
        (
            AsyncCockroachDBSaver(_closed("async")),
            "langchain_cockroachdb.checkpointer.async_saver",
        ),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert saver._can_retry(psycopg.OperationalError("lost")) is False
        assert [record.name for record in caplog.records] == [module]


async def test_cursor_default_label_follows_component(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_cockroachdb import _psycopg_ops

    labels: list[str] = []

    @contextlib.contextmanager
    def record(fileno: Any, timeout: Any, *, label: str) -> Iterator[None]:
        labels.append(label)
        yield

    @contextlib.asynccontextmanager
    async def arecord(fileno: Any, timeout: Any, *, label: str) -> AsyncIterator[None]:
        labels.append(label)
        yield

    monkeypatch.setattr(_psycopg_ops, "abandon_after", record)
    monkeypatch.setattr(_psycopg_ops, "aabandon_after", arecord)
    for component in ("checkpointer", "store"):
        sync_host = _host("sync", component, FakeSingleConnection(None))
        with sync_host._cursor():
            pass
        async_host = _host("async", component, FakeSingleAsyncConnection(None))
        async with async_host._cursor():
            pass
    # The checkpointer keeps its 0.4.1 label, "checkpoint operation"
    assert labels == ["checkpoint operation"] * 2 + ["store operation"] * 2


def test_hosts_must_name_their_component() -> None:
    from langchain_cockroachdb._psycopg_ops import AsyncPsycopgOps, PsycopgOps

    for base in (PsycopgOps, AsyncPsycopgOps):
        with pytest.raises(TypeError, match="component"):
            type("Nameless", (base,), {})
        with pytest.raises(TypeError, match="component"):
            type("Blank", (base,), {"component": ""})


@pytest.mark.parametrize(
    "module",
    [
        "langchain_cockroachdb._psycopg_ops",
        "langchain_cockroachdb.store",
        "langchain_cockroachdb.checkpointer.saver",
        "langchain_cockroachdb.indexes",
    ],
)
def test_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    # The shared machinery used to import the checkpointer package, which
    # imports the savers, which import the machinery. Nothing may depend on
    # which of these a program imports first.
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_old_internal_paths_reexport_the_shared_helpers() -> None:
    from langchain_cockroachdb import _psycopg_conn
    from langchain_cockroachdb.checkpointer import _ainternal, _internal

    for name in (
        "OPEN_TRANSACTION",
        "Conn",
        "check_committed",
        "flag_broken_connection",
        "get_connection",
        "in_caller_transaction",
        "mark_in_caller_transaction",
        "stuck_batch_error",
    ):
        assert getattr(_internal, name) is getattr(_psycopg_conn, name), name
    assert _ainternal.Conn is _psycopg_conn.AsyncConn
    assert _ainternal.get_connection is _psycopg_conn.aget_connection
    assert _ainternal.flag_broken_connection is _psycopg_conn.flag_broken_connection


# The wait for the socket between two flushes, and the flushes around it. RW:
# send() takes a readable socket too, and reads what the server sent meanwhile
LEFT_OVER = [("flush", 1), ("wait", Wait.RW)]
EMPTY = ("flush", 0)


@pytest.fixture
def _sockets() -> Iterator[None]:
    yield
    close_sockets()


def _single(kind: str, error: BaseException | None = None, **kwargs: Any) -> Any:
    cls = FakeSingleConnection if kind == "sync" else FakeSingleAsyncConnection
    return cls(error, **kwargs)


async def _maybe_await(result: Any) -> Any:
    return await result if asyncio.iscoroutine(result) else result


@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_drain_flushes_until_libpq_has_sent_everything(kind: str) -> None:
    conn = _single(kind)
    conn.pgconn = FakePGconn([1, 1, 0])
    host = _host(kind, "store", conn)
    await _maybe_await(host._drain_output(conn))
    # Three flushes, and a wait for the socket to take more before each retry
    assert conn.pgconn.io == [*LEFT_OVER, *LEFT_OVER, EMPTY]
    assert all(state & Wait.W for event, state in conn.pgconn.io if event == "wait")


def _run_sync_path(host: Any, where: str) -> None:
    if where == "statement":
        with host._cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS t (id INT PRIMARY KEY)")
            host._sync_pipeline()
    elif where == "stopped batch":
        with host._cursor() as cur:
            cur.execute("SELECT 1")
    else:
        with host._cursor(pipeline=True, transaction=True) as cur:
            cur.execute("INSERT INTO t VALUES (1)")


async def _run_async_path(host: Any, where: str) -> None:
    if where == "statement":
        async with host._cursor() as cur:
            await cur.execute("CREATE TABLE IF NOT EXISTS t (id INT PRIMARY KEY)")
            await host._sync_pipeline()
    elif where == "stopped batch":
        async with host._cursor() as cur:
            await cur.execute("SELECT 1")
    else:
        async with host._cursor(pipeline=True, transaction=True) as cur:
            await cur.execute("INSERT INTO t VALUES (1)")


@pytest.mark.usefixtures("_sockets")
@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("where", ["write", "statement", "error", "stopped batch"])
async def test_every_component_pipeline_sync_comes_after_a_drain(kind: str, where: str) -> None:
    # The places that sync a component-wide pipeline: the end of a write, a
    # sync after a single statement, the resync after an error, and the sync
    # before a stopped batch of ours is rolled back
    error = psycopg.errors.UndefinedTable("relation does not exist")
    conn = _single(kind, error if where == "error" else None, fail_at=2)
    conn.in_pipeline_mode = True
    # The first drain finds bytes that the last flush left behind
    conn.pgconn = FakePGconn([1])
    host = _host(kind, "store", conn)
    host.pipe = FakePipe(conn.pgconn.io) if kind == "sync" else FakeAsyncPipe(conn.pgconn.io)
    if where == "stopped batch":
        host._batch_open = True
        conn.info.transaction_status = TransactionStatus.INTRANS
    run = _run_sync_path if kind == "sync" else _run_async_path
    with pytest.raises(type(error)) if where == "error" else contextlib.nullcontext():
        await _maybe_await(run(host, where))
    assert host.pipe.syncs == 1
    assert conn.pgconn.io == [*LEFT_OVER, EMPTY, ("sync",)]


@pytest.mark.usefixtures("_sockets")
@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_a_write_in_its_own_pipeline_block_is_not_drained(kind: str) -> None:
    # Without a component-wide pipeline a write gets its own pipeline block,
    # and psycopg flushes everything as the block ends
    conn = _single(kind)
    host = _host(kind, "store", conn)
    await _maybe_await((_run_sync_path if kind == "sync" else _run_async_path)(host, "write"))
    assert ("pipeline", "open") in conn.events
    assert conn.pgconn.io == []
