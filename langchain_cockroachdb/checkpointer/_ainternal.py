"""Shared async utility functions for the CockroachDB checkpoint classes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeAlias

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb.checkpointer._internal import flag_broken_connection

Conn: TypeAlias = AsyncConnection[DictRow] | AsyncConnectionPool[AsyncConnection[DictRow]]


@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[AsyncConnection[DictRow]]:
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
