"""Connection and transaction helpers for the components that use psycopg directly.

The checkpointers and the store share these, through the machinery in
``_psycopg_ops``: checking out a connection from a pool or using the single
one, recognizing a transaction the caller opened, and checking that a
batch's COMMIT committed. The sync names have async twins: ``AsyncConn``
and ``aget_connection``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeAlias

import psycopg
from psycopg import AsyncConnection, AsyncCursor, Connection, Cursor
from psycopg.pq import TransactionStatus
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool, ConnectionPool

Conn: TypeAlias = Connection[DictRow] | ConnectionPool[Connection[DictRow]]
AsyncConn: TypeAlias = AsyncConnection[DictRow] | AsyncConnectionPool[AsyncConnection[DictRow]]

# What libpq reports while a transaction is open, or failed and not yet rolled back
OPEN_TRANSACTION = frozenset({TransactionStatus.INTRANS, TransactionStatus.INERROR})


def stuck_batch_error(component: str = "checkpointer") -> psycopg.Error:
    """The error raised when a stopped write batch of ours could not be rolled back.

    Running the operation anyway would nest it in that transaction, which
    never commits. ``component`` names the owner of the batch. The
    checkpointer's message calls its owner the saver, as it always has.
    """
    owner = "saver" if component == "checkpointer" else component
    return psycopg.errors.ActiveSqlTransaction(
        f"a write batch of the {component} that stopped partway may still be open "
        "on its connection and could not be rolled back. This happens when the "
        f"{owner}'s single connection is shared and a transaction the caller opened "
        "is open on it (psycopg forbids rollback() inside conn.transaction()). "
        f"End that transaction, or give the {owner} a pool."
    )


def mark_in_caller_transaction(error: BaseException) -> None:
    """Mark ``error`` as raised inside a transaction the caller opened.

    The component that runs the operation raises such an error at once. The
    transaction is the caller's: a retry inside it would keep it open through
    the backoff, hide the error from the caller's own retry logic, and cannot
    help after an error that aborted the transaction. The caller can retry the whole transaction.
    """
    error.in_caller_transaction = True  # type: ignore[attr-defined]


def in_caller_transaction(error: BaseException) -> bool:
    """True if ``error`` was raised inside a transaction the caller opened."""
    return bool(getattr(error, "in_caller_transaction", False))


def check_committed(cursor: Cursor[Any] | AsyncCursor[Any]) -> None:
    """Raise unless the server answered the COMMIT that ``cursor`` sent with COMMIT.

    CockroachDB answers COMMIT in a failed transaction with ROLLBACK and no
    error. A write sends COMMIT in the same pipeline batch as its statements,
    so the server skips it after any failure and the failure itself is
    raised. This makes sure a COMMIT never reports success for a rollback.
    """
    if cursor.statusmessage != "COMMIT":
        raise psycopg.errors.InFailedSqlTransaction(
            f"the server answered COMMIT with {cursor.statusmessage!r}, so the write was not saved"
        )


def flag_broken_connection(
    error: psycopg.Error, conn: Connection[Any] | AsyncConnection[Any]
) -> None:
    """Mark ``error`` as a lost connection if psycopg reports ``conn`` broken.

    RetryPolicy then retries the error even if its message is not in
    TRANSIENT_MESSAGES, as it does when SQLAlchemy invalidates a connection.
    A usable SQLSTATE and the non-transient phrases still decide first. Call
    this only for a pooled connection: the pool replaces a broken connection,
    so the next attempt runs on another one.
    """
    if conn.broken:
        error.connection_invalidated = True  # type: ignore[attr-defined]


@contextmanager
def get_connection(conn: Conn) -> Iterator[Connection[DictRow]]:
    """Use the single connection, or check one out of the pool.

    A driver error on a pooled connection that broke is marked as a lost
    connection, which the retry policy retries on a new connection.
    """
    if isinstance(conn, Connection):
        yield conn
    elif isinstance(conn, ConnectionPool):
        with conn.connection() as pool_conn:
            try:
                yield pool_conn
            except psycopg.Error as error:
                flag_broken_connection(error, pool_conn)
                raise
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")


@asynccontextmanager
async def aget_connection(conn: AsyncConn) -> AsyncIterator[AsyncConnection[DictRow]]:
    """Use the single connection, or check one out of the pool.

    A driver error on a pooled connection that broke is marked as a lost
    connection, which the retry policy retries on a new connection.
    """
    if isinstance(conn, AsyncConnection):
        yield conn
    elif isinstance(conn, AsyncConnectionPool):
        async with conn.connection() as pool_conn:
            try:
                yield pool_conn
            except psycopg.Error as error:
                flag_broken_connection(error, pool_conn)
                raise
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
