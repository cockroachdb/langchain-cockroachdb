"""Fixtures for the store integration tests."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import psycopg
import pytest
from langgraph.store.memory import InMemoryStore
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import CockroachDBStore, RetryPolicy
from langchain_cockroachdb._connection import connection_options

MODES = ("pool", "connection", "pipeline")
FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)


def drop_store_tables(libpq_url: str, table_name: str, schema: str = "public") -> None:
    """Drop the items, vectors and migrations tables of a store."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        for suffix in ("", "_vectors", "_migrations"):
            conn.execute(f"DROP TABLE IF EXISTS {schema}.{table_name}{suffix} CASCADE")


@pytest.fixture
def table_name(libpq_url: str) -> Iterator[str]:
    """A fresh items table name, whose tables are dropped after the test."""
    name = f"store_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        drop_store_tables(libpq_url, name)
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"store table cleanup failed: {error!r}", UserWarning, stacklevel=1)


@pytest.fixture
def store_factory(libpq_url: str, table_name: str) -> Iterator[Callable[..., CockroachDBStore]]:
    """Make stores on a pool, a connection or a pipelined connection.

    Every store of a test shares the test's table name unless ``table_name``
    is passed. ``setup()`` runs unless ``setup=False``. Connections close
    after the test.
    """
    resources: list[Any] = []

    def make(mode: str, *, setup: bool = True, **kwargs: Any) -> CockroachDBStore:
        kwargs.setdefault("table_name", table_name)
        kwargs.setdefault("retry_policy", FAST)
        options = connection_options(libpq_url, None, 10.0, 5, component="store")
        if mode == "pool":
            pool: ConnectionPool[Connection[Any]] = ConnectionPool(
                libpq_url, min_size=1, max_size=4, kwargs=options, open=True
            )
            resources.append(pool)
            store = CockroachDBStore(pool, **kwargs)
        elif mode in ("connection", "pipeline"):
            conn = Connection.connect(libpq_url, **options)
            resources.append(conn)
            pipe = None
            if mode == "pipeline":
                pipe_cm = conn.pipeline()
                pipe = pipe_cm.__enter__()
                resources.append(pipe_cm)
            store = CockroachDBStore(conn, pipe=pipe, **kwargs)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        if setup:
            store.setup()
        return store

    yield make
    for resource in reversed(resources):
        try:
            if isinstance(resource, (ConnectionPool, Connection)):
                resource.close()
            else:
                resource.__exit__(None, None, None)
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"store cleanup failed: {error!r}", UserWarning, stacklevel=1)


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    """Each connection mode in turn."""
    return request.param


@pytest.fixture
def reference() -> Callable[[], InMemoryStore]:
    """LangGraph's InMemoryStore, the behavior the store must match."""
    return InMemoryStore


@pytest.fixture
def admin(libpq_url: str) -> Iterator[Connection[Any]]:
    """A plain autocommit connection for checking rows directly."""
    with Connection.connect(libpq_url, autocommit=True, row_factory=dict_row) as conn:
        yield conn
