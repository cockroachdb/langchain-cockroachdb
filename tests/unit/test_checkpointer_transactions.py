"""How the checkpointers group a write's statements, on fake psycopg connections.

A write with several statements (``put`` with channel blobs, ``put_writes``
with several writes, ``delete_thread``) commits all of them or none. On an
autocommit connection in pipeline mode, BEGIN, the statements and COMMIT go
out as one batch: one round trip, which CockroachDB can retry by itself. If
the batch stops between BEGIN and COMMIT, an error, a client-side failure and
a cancellation included, the saver rolls the transaction back before the
connection is used again. Connections not in autocommit mode, and libpq
without pipeline mode, use psycopg's ``conn.transaction()``. A write with one
statement needs no explicit transaction.

The fakes record statements, rollbacks and when pipeline and transaction
blocks open and close; the integration tests check the same against
CockroachDB.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import errors as pg_errors
from psycopg.pq import TransactionStatus

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
OPERATIONS = pytest.mark.parametrize("operation", list(SEVERAL))

KINDS = pytest.mark.parametrize("kind", ["sync", "async"])
# Autocommit connections in pipeline mode: a pool, a single connection, and a
# single connection with a saver-wide pipeline
BATCHED = ["pool", "connection", "pipeline"]
# Where psycopg's transaction runs a write instead: libpq without pipeline
# mode, and connections not in autocommit mode
PSYCOPG = ["no_pipeline_support", "not_autocommit", "pipeline_not_autocommit"]


@pytest.fixture(autouse=True)
def _close_sockets() -> Iterator[None]:
    yield
    close_sockets()


def fake(kind: str, mode: str, error: BaseException | None = None, **kwargs: Any) -> Any:
    """A fake connection of the right type and autocommit setting for ``mode``."""
    pooled = mode in ("pool", "no_pipeline_support", "not_autocommit")
    if kind == "sync":
        cls: type[FakeConnection] = FakeConnection if pooled else FakeSingleConnection
    else:
        cls = FakeAsyncConnection if pooled else FakeSingleAsyncConnection
    conn = cls(error, **kwargs)
    conn.autocommit = mode not in ("not_autocommit", "pipeline_not_autocommit")
    return conn


def saver_for(kind: str, mode: str, *scripted: Any) -> tuple[Any, list[Any]]:
    """A saver in ``mode``, and the connections it has used so far."""
    saver_class: Any = CockroachDBSaver if kind == "sync" else AsyncCockroachDBSaver
    if mode in ("pool", "no_pipeline_support", "not_autocommit"):
        pool_class: Any = FakePool if kind == "sync" else FakeAsyncPool
        pool = pool_class(*scripted, factory=lambda: fake(kind, mode))
        saver = saver_class(pool, retry_policy=POLICY)
        saver.supports_pipeline = mode != "no_pipeline_support"
        return saver, pool.handed_out
    conn = scripted[0] if scripted else fake(kind, mode)
    pipe: Any = None
    if mode in ("pipeline", "pipeline_not_autocommit"):
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


async def read(kind: str, saver: Any) -> None:
    if kind == "sync":
        saver.get_tuple(CONFIG)
    else:
        await saver.aget_tuple(CONFIG)


def batch(mode: str, statements: list[str], error: str | None = None) -> list[tuple]:
    """The events of a write that sends BEGIN, ``statements`` and COMMIT as one batch.

    With ``error``, the last statement stopped the batch with that error: no
    COMMIT is sent, and the saver rolls the transaction back.
    """
    body: list[tuple] = [("statement", statement) for statement in ["BEGIN", *statements]]
    if error is None:
        body.append(("statement", "COMMIT"))
    if mode == "pipeline":
        # One sync sends the batch; after an error it is the resync instead
        events = [*body, ("sync",)]
        if error is not None:
            # In pipeline mode the ROLLBACK waits in the queue until a sync
            events += [("rollback",), ("sync",)]
        return events
    events = [("pipeline", "open"), *body, ("pipeline", "close", error)]
    if error is not None:
        events.append(("rollback",))
    return events


def psycopg_transaction(mode: str, statements: list[str], error: str | None = None) -> list[tuple]:
    """The events of a write that runs ``statements`` in psycopg's conn.transaction().

    psycopg rolls back a transaction that closes with an error.
    """
    body: list[tuple] = [("statement", statement) for statement in statements]
    if mode == "not_autocommit":
        body = [("pipeline", "open"), *body, ("pipeline", "close", error)]
    events = [("transaction", "open"), *body, ("transaction", "close", error)]
    if mode == "pipeline_not_autocommit" and error is not None:
        events.append(("sync",))
    return events


def server_error(code: str, message: str) -> psycopg.Error:
    return pg_errors.lookup(code)(message)


@KINDS
@pytest.mark.parametrize("mode", BATCHED)
@OPERATIONS
async def test_a_write_with_several_statements_sends_begin_and_commit_in_its_batch(
    kind: str, mode: str, operation: str
) -> None:
    saver, used = saver_for(kind, mode)
    await write(kind, saver, operation)
    assert len(used) == 1
    assert used[0].events == batch(mode, SEVERAL[operation])
    assert used[0].info.transaction_status == TransactionStatus.IDLE


@KINDS
@pytest.mark.parametrize("mode", PSYCOPG)
@OPERATIONS
async def test_a_write_falls_back_to_psycopgs_transaction(
    kind: str, mode: str, operation: str
) -> None:
    # Without autocommit psycopg opens a transaction itself, so a BEGIN of ours
    # would fail; without pipeline mode the batch cannot go out at once
    saver, used = saver_for(kind, mode)
    await write(kind, saver, operation)
    # In pipeline mode psycopg's transaction syncs the pipeline itself, so no sync follows
    assert used[0].events == psycopg_transaction(mode, SEVERAL[operation])


@KINDS
@pytest.mark.parametrize("mode", BATCHED + PSYCOPG)
@pytest.mark.parametrize("operation", list(ONE))
async def test_a_write_with_one_statement_needs_no_transaction(
    kind: str, mode: str, operation: str
) -> None:
    saver, used = saver_for(kind, mode)
    await write(kind, saver, operation, several=False)
    statement = [("statement", ONE[operation][0])]
    in_a_pipeline = [("pipeline", "open"), *statement, ("pipeline", "close", None)]
    expected = {
        "pool": in_a_pipeline,
        "connection": in_a_pipeline,
        "not_autocommit": in_a_pipeline,
        "pipeline": [*statement, ("sync",)],
        "pipeline_not_autocommit": [*statement, ("sync",)],
        "no_pipeline_support": statement,
    }
    assert used[0].events == expected[mode]


@KINDS
@pytest.mark.parametrize("mode", [*BATCHED, "not_autocommit"])
@OPERATIONS
async def test_a_retried_write_runs_its_whole_transaction_again(
    kind: str, mode: str, operation: str
) -> None:
    # The last statement fails with 40001. Nothing of the attempt is committed,
    # and the retry sends every statement again.
    statements = SEVERAL[operation]
    last = len(statements) + (mode in BATCHED)  # BEGIN comes first in a batch
    error = server_error("40001", "restart transaction")
    saver, used = saver_for(kind, mode, fake(kind, mode, error, fail_at=last))
    await write(kind, saver, operation)
    events = batch if mode in BATCHED else psycopg_transaction
    failed = events(mode, statements, error="SerializationFailure")
    committed = events(mode, statements)
    if mode in ("pool", "not_autocommit"):
        # The retry runs on another pooled connection
        assert [conn.events for conn in used] == [failed, committed]
    else:
        assert used[0].events == failed + committed


@KINDS
@pytest.mark.parametrize("mode", BATCHED)
async def test_a_write_stopped_by_a_client_side_failure_is_rolled_back(
    kind: str, mode: str
) -> None:
    # Anything that stops the caller between BEGIN and COMMIT, here an error
    # raised as the checkpoint row is sent
    def fail_on_the_checkpoint(query: str) -> None:
        if query == CHECKPOINT:
            raise RuntimeError("client-side failure")

    async def afail_on_the_checkpoint(query: str) -> None:
        fail_on_the_checkpoint(query)

    hook = fail_on_the_checkpoint if kind == "sync" else afail_on_the_checkpoint
    saver, used = saver_for(kind, mode, fake(kind, mode, on_statement=hook))
    with pytest.raises(RuntimeError, match="client-side failure"):
        await write(kind, saver, "put")
    assert used[0].events == batch(mode, [BLOBS, CHECKPOINT], error="RuntimeError")
    assert used[0].info.transaction_status == TransactionStatus.IDLE


@pytest.mark.timeout(10)
@pytest.mark.parametrize("mode", BATCHED)
async def test_a_write_cancelled_between_begin_and_commit_is_rolled_back(mode: str) -> None:
    reached = asyncio.Event()

    async def wait_on_the_checkpoint(query: str) -> None:
        if query == CHECKPOINT:
            reached.set()
            await asyncio.Event().wait()

    saver, used = saver_for("async", mode, fake("async", mode, on_statement=wait_on_the_checkpoint))
    task = asyncio.create_task(write("async", saver, "put"))
    await asyncio.wait_for(reached.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert used[0].events == batch(mode, [BLOBS, CHECKPOINT], error="CancelledError")
    assert used[0].info.transaction_status == TransactionStatus.IDLE


@KINDS
@pytest.mark.parametrize("mode", BATCHED)
async def test_a_commit_answered_with_rollback_raises(kind: str, mode: str) -> None:
    # CockroachDB answers COMMIT in a failed transaction with ROLLBACK and no
    # error. The batch never sends COMMIT after a failure, and the saver
    # refuses to call such an answer a success.
    saver, used = saver_for(kind, mode, fake(kind, mode, commit_answer="ROLLBACK"))
    with pytest.raises(pg_errors.InFailedSqlTransaction, match="answered COMMIT with 'ROLLBACK'"):
        await write(kind, saver, "put")
    expected = batch(mode, [BLOBS, CHECKPOINT])
    if mode == "pipeline":
        expected.append(("sync",))  # the resync after the error
    # One attempt: the write is not retried
    assert [conn.events for conn in used] == [expected]


@KINDS
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
@pytest.mark.parametrize("autocommit", [True, False])
async def test_a_transaction_left_open_is_rolled_back_before_the_next_operation(
    kind: str, mode: str, autocommit: bool
) -> None:
    # A rollback that was itself interrupted, for example by a second
    # cancellation, leaves the single connection inside the transaction
    conn = fake(kind, mode)
    conn.autocommit = autocommit
    conn.info.transaction_status = TransactionStatus.INTRANS
    saver, _ = saver_for(kind, mode, conn)
    await read(kind, saver)
    rollback = [("rollback",), ("sync",)] if mode == "pipeline" else [("rollback",)]
    first_statement = 2 if mode == "pipeline" else 1
    if autocommit:
        assert conn.events[: len(rollback)] == rollback
    else:
        # Without autocommit an open transaction is psycopg's own, so it stays
        assert conn.events[0][0] == "statement"
        first_statement = 0
    assert conn.events[first_statement][1].strip().startswith("select")


@KINDS
@pytest.mark.parametrize("operation", ["put", "get_tuple"])
async def test_a_saver_wide_pipeline_resyncs_and_rolls_back_under_the_lock(
    kind: str, operation: str
) -> None:
    # The write fails on its first statement after BEGIN, the read on its only one
    error = server_error("42P01", "relation does not exist")
    conn = fake(kind, "pipeline", error, fail_at=2 if operation == "put" else 1)
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
        else:
            await read(kind, saver)
    # The resync, and in a write the sync that sends the ROLLBACK, both under the lock
    assert held and all(held)
    if operation == "put":
        assert conn.events == batch("pipeline", [BLOBS], error="UndefinedTable")
    else:
        assert conn.events[-1] == ("sync",)
        assert ("rollback",) not in conn.events
