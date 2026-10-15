"""Shared sync utility functions for the CockroachDB checkpoint classes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypeAlias

import psycopg
from psycopg import AsyncConnection, AsyncCursor, Connection, Cursor
from psycopg.pq import TransactionStatus
from psycopg.rows import DictRow
from psycopg_pool import ConnectionPool

Conn: TypeAlias = Connection[DictRow] | ConnectionPool[Connection[DictRow]]

# What libpq reports while a transaction is open, or failed and not yet rolled back
OPEN_TRANSACTION = frozenset({TransactionStatus.INTRANS, TransactionStatus.INERROR})


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
