"""Shared async utility functions for the CockroachDB checkpoint classes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeAlias

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

Conn: TypeAlias = AsyncConnection[DictRow] | AsyncConnectionPool[AsyncConnection[DictRow]]


@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[AsyncConnection[DictRow]]:
    if isinstance(conn, AsyncConnection):
        yield conn
    elif isinstance(conn, AsyncConnectionPool):
        async with conn.connection() as pool_conn:
            yield pool_conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
