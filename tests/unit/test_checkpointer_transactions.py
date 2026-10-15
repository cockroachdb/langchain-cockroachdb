"""How the checkpointers group a write's statements, on fake psycopg connections.

A write with several statements (``put`` with channel blobs, ``put_writes``
with several writes, ``delete_thread``) runs them in one explicit
transaction. An error, a cancellation or a client-side failure partway
through then leaves nothing behind, and a retry runs the whole transaction
again. A write with one statement needs no explicit transaction.

The fakes record when pipeline and transaction blocks open and close. A
transaction that closes with an error is one psycopg rolls back; the
integration tests check that against CockroachDB.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import errors as pg_errors

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver, RetryPolicy
from tests.unit.checkpointer_fakes import (
    FakeAsyncConnection,
    FakeAsyncPipe,
    FakeAsyncPool,
    FakeConnection,
    FakePipe,
    FakePool,
    FakeSingleAsyncConnection,
    FakeSingleConnection,
    close_sockets,
)

CONFIG: Any = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
SAVED: Any = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "1"}}
POLICY = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)

BLOBS = CockroachDBSaver.UPSERT_CHECKPOINT_BLOBS_SQL
CHECKPOINT = CockroachDBSaver.UPSERT_CHECKPOINTS_SQL
WRITES = CockroachDBSaver.INSERT_CHECKPOINT_WRITES_SQL
DELETES = [
    "DELETE FROM checkpoints WHERE thread_id = %s",
    "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
    "DELETE FROM checkpoint_writes WHERE thread_id = %s",
]
# What each write sends. The fakes record one statement per executemany call.
SEVERAL = {"put": [BLOBS, CHECKPOINT], "put_writes": [WRITES], "delete_thread": DELETES}
ONE = {"put": [CHECKPOINT], "put_writes": [WRITES]}

KINDS = pytest.mark.parametrize("kind", ["sync", "async"])
# A pool, a single connection, a single connection with a saver-wide pipeline,
# and a pool when libpq has no pipeline mode
MODES = ["pool", "connection", "pipeline", "no_pipeline_support"]


@pytest.fixture(autouse=True)
def _close_sockets() -> Iterator[None]:
    yield
    close_sockets()


def fake(kind: str, mode: str, error: BaseException | None = None, **kwargs: Any) -> Any:
    """A fake connection of the right type for a saver in ``mode``."""
    pooled = mode in ("pool", "no_pipeline_support")
    if kind == "sync":
        cls: type[FakeConnection] = FakeConnection if pooled else FakeSingleConnection
    else:
        cls = FakeAsyncConnection if pooled else FakeSingleAsyncConnection
    return cls(error, **kwargs)


def saver_for(kind: str, mode: str, *scripted: Any) -> tuple[Any, list[Any]]:
    """A saver in ``mode``, and the connections it has used so far."""
    saver_class: Any = CockroachDBSaver if kind == "sync" else AsyncCockroachDBSaver
    if mode in ("pool", "no_pipeline_support"):
        pool: Any = FakePool(*scripted) if kind == "sync" else FakeAsyncPool(*scripted)
        saver = saver_class(pool, retry_policy=POLICY)
        saver.supports_pipeline = mode == "pool"
        return saver, pool.handed_out
    conn = scripted[0] if scripted else fake(kind, mode)
    pipe: Any = None
    if mode == "pipeline":
        pipe = FakePipe(conn.events) if kind == "sync" else FakeAsyncPipe(conn.events)
    return saver_class(conn, pipe, retry_policy=POLICY), [conn]


def checkpoint_with_blobs() -> tuple[Any, dict[str, str]]:
    """A checkpoint with one channel stored as a blob and one stored inline."""
    checkpoint = empty_checkpoint()
    versions = {"messages": "1", "count": "1"}
    checkpoint["channel_values"] = {"messages": ["hello"], "count": 1}
    checkpoint["channel_versions"] = dict(versions)
    return checkpoint, versions


async def write(kind: str, saver: Any, operation: str, *, several: bool = True) -> None:
    """Run ``operation`` with several statements, or with one."""
    if operation == "put":
        checkpoint, versions = checkpoint_with_blobs() if several else (empty_checkpoint(), {})
        args: tuple[Any, ...] = (CONFIG, checkpoint, {}, versions)
    elif operation == "put_writes":
        args = (SAVED, [("a", 1), ("b", 2)] if several else [("a", 1)], "task-1")
    else:
        args = ("thread-1",)
    if kind == "sync":
        getattr(saver, operation)(*args)
    else:
        await getattr(saver, f"a{operation}")(*args)


def transaction(mode: str, statements: list[str], error: str | None = None) -> list[tuple]:
    """The events of one write that runs ``statements`` in an explicit transaction.

    ``error`` names the error the transaction closed with, if any.
    """
    body: list[tuple] = [("statement", statement) for statement in statements]
    if mode in ("pool", "connection"):
        body = [("pipeline", "open"), *body, ("pipeline", "close", error)]
    events = [("transaction", "open"), *body, ("transaction", "close", error)]
    if mode == "pipeline" and error is not None:
        # The resync after an error, while the saver still holds its lock
        events.append(("sync",))
    return events


def server_error(code: str, message: str) -> psycopg.Error:
    return pg_errors.lookup(code)(message)


@KINDS
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation", ["put", "put_writes", "delete_thread"])
async def test_a_write_with_several_statements_runs_them_in_one_transaction(
    kind: str, mode: str, operation: str
) -> None:
    saver, used = saver_for(kind, mode)
    await write(kind, saver, operation)
    assert len(used) == 1
    # In pipeline mode the transaction's exit already synced, so no extra sync follows
    assert used[0].events == transaction(mode, SEVERAL[operation])


@KINDS
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation", ["put", "put_writes"])
async def test_a_write_with_one_statement_needs_no_transaction(
    kind: str, mode: str, operation: str
) -> None:
    saver, used = saver_for(kind, mode)
    await write(kind, saver, operation, several=False)
    statement = [("statement", ONE[operation][0])]
    expected = {
        "pool": [("pipeline", "open"), *statement, ("pipeline", "close", None)],
        "connection": [("pipeline", "open"), *statement, ("pipeline", "close", None)],
        "pipeline": [*statement, ("sync",)],
        "no_pipeline_support": statement,
    }
    assert used[0].events == expected[mode]


@KINDS
@pytest.mark.parametrize("mode", ["pool", "connection", "pipeline"])
@pytest.mark.parametrize("operation", ["put", "put_writes", "delete_thread"])
async def test_a_retried_write_runs_its_whole_transaction_again(
    kind: str, mode: str, operation: str
) -> None:
    # The last statement fails with 40001. The transaction closes with the
    # error, so psycopg rolls it back, and the retry sends every statement again.
    statements = SEVERAL[operation]
    first = fake(kind, mode, server_error("40001", "restart transaction"), fail_at=len(statements))
    saver, used = saver_for(kind, mode, first)
    await write(kind, saver, operation)
    failed = transaction(mode, statements, error="SerializationFailure")
    committed = transaction(mode, statements)
    if mode == "pool":
        # The retry runs on another pooled connection
        assert [conn.events for conn in used] == [failed, committed]
    else:
        assert used[0].events == failed + committed


@KINDS
@pytest.mark.parametrize("mode", ["pool", "connection", "pipeline"])
async def test_a_client_side_failure_partway_through_closes_the_transaction_with_it(
    kind: str, mode: str
) -> None:
    # Anything that stops the client between statements, here an error raised
    # as the second statement runs, leaves the transaction to psycopg's rollback
    def fail_on_the_checkpoint(query: str) -> None:
        if query == CHECKPOINT:
            raise RuntimeError("client-side failure")

    async def afail_on_the_checkpoint(query: str) -> None:
        fail_on_the_checkpoint(query)

    hook = fail_on_the_checkpoint if kind == "sync" else afail_on_the_checkpoint
    saver, used = saver_for(kind, mode, fake(kind, mode, on_statement=hook))
    with pytest.raises(RuntimeError, match="client-side failure"):
        await write(kind, saver, "put")
    assert used[0].events == transaction(mode, [BLOBS, CHECKPOINT], error="RuntimeError")


@KINDS
@pytest.mark.parametrize("operation", ["put", "get_tuple"])
async def test_a_saver_wide_pipeline_resyncs_under_the_lock_after_an_error(
    kind: str, operation: str
) -> None:
    conn = fake(kind, "pipeline", server_error("42P01", "relation does not exist"))
    held: list[bool] = []
    saver: Any = None

    def note_the_lock() -> None:
        held.append(saver.lock.locked())

    pipe: Any = FakePipe if kind == "sync" else FakeAsyncPipe
    saver_class: Any = CockroachDBSaver if kind == "sync" else AsyncCockroachDBSaver
    saver = saver_class(conn, pipe(conn.events, note_the_lock), retry_policy=POLICY)
    with pytest.raises(pg_errors.UndefinedTable):
        if operation == "put":
            await write(kind, saver, "put")
        elif kind == "sync":
            saver.get_tuple(CONFIG)
        else:
            await saver.aget_tuple(CONFIG)
    assert held == [True]
    assert conn.events[-1] == ("sync",)
