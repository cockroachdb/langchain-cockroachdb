"""Tests for the psycopg machinery that the checkpointers and the store share."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from tests.unit.checkpointer_fakes import FakeSingleAsyncConnection, FakeSingleConnection

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
