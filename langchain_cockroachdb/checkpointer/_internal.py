"""Shared sync utility functions for the CockroachDB checkpoint classes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

from psycopg import Connection
from psycopg.rows import DictRow
from psycopg_pool import ConnectionPool

Conn: TypeAlias = Connection[DictRow] | ConnectionPool[Connection[DictRow]]


@contextmanager
def get_connection(conn: Conn) -> Iterator[Connection[DictRow]]:
    if isinstance(conn, Connection):
        yield conn
    elif isinstance(conn, ConnectionPool):
        with conn.connection() as pool_conn:
            yield pool_conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
