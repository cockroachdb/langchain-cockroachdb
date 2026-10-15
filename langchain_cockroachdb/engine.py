"""CockroachDB async engine management with transaction retry support.

Async methods run on the caller's event loop. Sync methods run their database
work on a shared background event loop, so they are safe to call from any
number of threads and from inside a running event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import weakref
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from langchain_cockroachdb._connection import to_sqlalchemy_url, with_libpq_defaults
from langchain_cockroachdb._guard import aabandon_after
from langchain_cockroachdb._loop import (
    arun_on_external_loop,
    get_background_loop,
    is_background_loop,
    run_coroutine_sync,
    run_on_external_loop,
)
from langchain_cockroachdb._sql import validate_identifier
from langchain_cockroachdb.hybrid_search_config import validate_fts_language
from langchain_cockroachdb.retry import RetryPolicy, resolve_retry_policy

logger = logging.getLogger(__name__)

T = TypeVar("T")

# id column types such as UUID, TEXT, STRING, INT8 or VARCHAR(64)
_ID_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\(\d+\))?$")


async def _driver_fileno(conn: AsyncConnection) -> int | None:
    """File descriptor of the driver connection's socket, if the driver exposes one."""
    try:
        raw = await conn.get_raw_connection()
        fileno = getattr(raw.driver_connection, "fileno", None)
        return int(fileno()) if callable(fileno) else None
    except Exception:
        return None


class CockroachDBEngine:
    """Manages an async SQLAlchemy engine for CockroachDB with retry support.

    Example:
        ```python
        engine = CockroachDBEngine.from_connection_string(
            "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
            retry_policy=RetryPolicy(max_attempts=8),
        )
        ```
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        retry_policy: RetryPolicy | None = None,
        retry_max_attempts: int | None = None,
        retry_initial_backoff: float | None = None,
        retry_max_backoff: float | None = None,
        retry_backoff_multiplier: float | None = None,
        retry_jitter: bool | None = None,
    ):
        """Initialize with an existing async engine.

        Args:
            engine: SQLAlchemy AsyncEngine connected to CockroachDB
            loop: Event loop that owns ``engine`` when it runs in another
                thread. Sync calls, and async calls from other loops, are sent
                there. Keep it running while the engine is in use: a call
                raises RuntimeError instead of waiting forever if the loop is
                closed or stops, or in a forked child process. Default: async
                calls run on the caller's loop and sync calls on a shared
                background loop.
            retry_policy: How operations retry (default: RetryPolicy())
            retry_max_attempts: Shorthand for RetryPolicy.max_attempts
            retry_initial_backoff: Shorthand for RetryPolicy.initial_backoff
            retry_max_backoff: Shorthand for RetryPolicy.max_backoff
            retry_backoff_multiplier: Shorthand for RetryPolicy.backoff_multiplier
            retry_jitter: Shorthand for RetryPolicy.jitter (True means 0.5)
        """
        self._engine = engine
        self._loop = loop
        # A loop passed as loop= runs only in this process, so a fork child must not use it
        self._pid = os.getpid()
        self.retry_policy = resolve_retry_policy(
            retry_policy,
            base=RetryPolicy(),
            max_attempts=retry_max_attempts,
            initial_backoff=retry_initial_backoff,
            max_backoff=retry_max_backoff,
            backoff_multiplier=retry_backoff_multiplier,
            jitter=retry_jitter,
        )
        self._usage_lock = threading.Lock()
        self._active = {"sync": 0, "async": 0}
        self._warned_mixed_use = False
        _ENGINES.add(self)

    @property
    def retry_max_attempts(self) -> int:
        """Shorthand for ``retry_policy.max_attempts``."""
        return self.retry_policy.max_attempts

    @property
    def retry_initial_backoff(self) -> float:
        """Shorthand for ``retry_policy.initial_backoff``."""
        return self.retry_policy.initial_backoff

    @property
    def retry_max_backoff(self) -> float:
        """Shorthand for ``retry_policy.max_backoff``."""
        return self.retry_policy.max_backoff

    @property
    def retry_backoff_multiplier(self) -> float:
        """Shorthand for ``retry_policy.backoff_multiplier``."""
        return self.retry_policy.backoff_multiplier

    @property
    def retry_jitter(self) -> bool:
        """True when ``retry_policy.jitter`` is above zero."""
        return self.retry_policy.jitter > 0

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        *,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = True,
        pool_recycle: int = 3600,
        pool_timeout: float = 30.0,
        connect_timeout: float | None = 10.0,
        retry_policy: RetryPolicy | None = None,
        retry_max_attempts: int | None = None,
        retry_initial_backoff: float | None = None,
        retry_max_backoff: float | None = None,
        retry_backoff_multiplier: float | None = None,
        retry_jitter: bool | None = None,
        **kwargs: Any,
    ) -> CockroachDBEngine:
        """Create an engine from a connection string.

        Any common spelling works (cockroachdb://, postgresql://, postgres://,
        with or without +psycopg or +psycopg2); all of them use psycopg 3.

        Args:
            connection_string: CockroachDB connection URL
            pool_size: Connection pool size (default: 10)
            max_overflow: Connections allowed beyond pool_size (default: 20)
            pool_pre_ping: Check connections before use (default: True)
            pool_recycle: Recycle connections after N seconds (default: 3600)
            pool_timeout: Seconds to wait for a pooled connection (default: 30.0)
            connect_timeout: Seconds to wait when opening a connection, or None.
                Ignored when the URL or connect_args already set one (default: 10.0)
            retry_policy: How operations retry (default: RetryPolicy())
            retry_max_attempts: Shorthand for RetryPolicy.max_attempts
            retry_initial_backoff: Shorthand for RetryPolicy.initial_backoff
            retry_max_backoff: Shorthand for RetryPolicy.max_backoff
            retry_backoff_multiplier: Shorthand for RetryPolicy.backoff_multiplier
            retry_jitter: Shorthand for RetryPolicy.jitter (True means 0.5)
            **kwargs: Passed to create_async_engine

        Returns:
            CockroachDBEngine instance
        """
        url = to_sqlalchemy_url(connection_string)
        # Only add libpq defaults for psycopg driver; asyncpg has different parameters
        if make_url(url).get_driver_name() == "psycopg":
            kwargs["connect_args"] = with_libpq_defaults(
                url, kwargs.get("connect_args"), connect_timeout
            )
        engine = create_async_engine(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            pool_timeout=pool_timeout,
            **kwargs,
        )
        return cls(
            engine,
            retry_policy=retry_policy,
            retry_max_attempts=retry_max_attempts,
            retry_initial_backoff=retry_initial_backoff,
            retry_max_backoff=retry_max_backoff,
            retry_backoff_multiplier=retry_backoff_multiplier,
            retry_jitter=retry_jitter,
        )

    @classmethod
    def from_engine(
        cls,
        engine: AsyncEngine,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        retry_policy: RetryPolicy | None = None,
        retry_max_attempts: int | None = None,
        retry_initial_backoff: float | None = None,
        retry_max_backoff: float | None = None,
        retry_backoff_multiplier: float | None = None,
        retry_jitter: bool | None = None,
    ) -> CockroachDBEngine:
        """Wrap an existing AsyncEngine.

        If you also use ``engine`` directly from your own event loop while
        calling sync methods here, pass that loop as ``loop`` so all work stays
        on it. Keep that loop running in its thread while the engine is in use.
        A call raises RuntimeError instead of waiting forever if the loop is
        closed or stops, or in a forked child process.

        Args:
            engine: SQLAlchemy AsyncEngine
            loop: Event loop that owns ``engine`` (default: none)
            retry_policy: How operations retry (default: RetryPolicy())
            retry_max_attempts: Shorthand for RetryPolicy.max_attempts
            retry_initial_backoff: Shorthand for RetryPolicy.initial_backoff
            retry_max_backoff: Shorthand for RetryPolicy.max_backoff
            retry_backoff_multiplier: Shorthand for RetryPolicy.backoff_multiplier
            retry_jitter: Shorthand for RetryPolicy.jitter (True means 0.5)

        Returns:
            CockroachDBEngine instance
        """
        return cls(
            engine,
            loop=loop,
            retry_policy=retry_policy,
            retry_max_attempts=retry_max_attempts,
            retry_initial_backoff=retry_initial_backoff,
            retry_max_backoff=retry_max_backoff,
            retry_backoff_multiplier=retry_backoff_multiplier,
            retry_jitter=retry_jitter,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Get the underlying AsyncEngine."""
        return self._engine

    @contextmanager
    def _usage(self, mode: str) -> Iterator[None]:
        """Track sync and async use so overlapping use can be reported once."""
        other = "async" if mode == "sync" else "sync"
        with self._usage_lock:
            self._active[mode] += 1
            report = self._active[other] > 0 and not self._warned_mixed_use
            if report:
                self._warned_mixed_use = True
        if report:
            logger.warning(
                "This CockroachDBEngine is being used from sync methods and from an event loop "
                "at the same time. Its connection pool belongs to one event loop at a time, so "
                "use one style per engine, or create a second engine for the other style."
            )
        try:
            yield
        finally:
            with self._usage_lock:
                self._active[mode] -= 1

    async def _run_as_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run database work for the async API."""
        if self._loop is not None:
            if asyncio.get_running_loop() is self._loop:
                return await coro
            return await arun_on_external_loop(coro, self._loop, owner_pid=self._pid)
        current_loop = asyncio.get_running_loop()
        if is_background_loop(current_loop):
            return await coro
        with self._usage("async"):
            return await coro

    def _run_as_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run database work for the sync API and wait for the result."""
        if self._loop is not None:
            return run_on_external_loop(coro, self._loop, owner_pid=self._pid)
        with self._usage("sync"):
            return run_coroutine_sync(coro, get_background_loop())

    @asynccontextmanager
    async def _connect(
        self, *, begin: bool, timeout: float | None, label: str
    ) -> AsyncIterator[AsyncConnection]:
        """Check out a connection, optionally inside a transaction, with the hang guard.

        The guard also covers COMMIT, which is where CockroachDB can hang while
        a range has lost quorum.
        """
        async with self._engine.connect() as conn:
            fileno = await _driver_fileno(conn) if timeout is not None else None
            async with aabandon_after(fileno, timeout, label=label):
                if begin:
                    async with conn.begin():
                        yield conn
                else:
                    yield conn

    async def _run_ddl(
        self, statements: list[str], *, label: str, policy: RetryPolicy | None = None
    ) -> None:
        """Run schema statements, each in its own implicit transaction.

        Each statement is retried on its own. Under the default autocommit_before_ddl
        setting, CockroachDB commits each schema change immediately, so an explicit
        transaction never made setup atomic. With autocommit_before_ddl off, an explicit
        transaction on a schema_locked table raises 57000.
        """
        retry = policy or self.retry_policy

        for stmt in statements:

            async def attempt(statement: str = stmt) -> None:
                async with self._connect(
                    begin=False, timeout=retry.ddl_attempt_timeout, label=label
                ) as conn:
                    await conn.execution_options(isolation_level="AUTOCOMMIT")
                    await conn.execute(text(statement))

            await retry.arun(attempt, label=label)

    async def ainit_vectorstore_table(
        self,
        table_name: str,
        vector_dimension: int,
        *,
        schema: str = "public",
        id_type: str = "UUID",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
        namespace_column: str | None = None,
        create_tsvector: bool = False,
        fts_language: str = "english",
        drop_if_exists: bool = False,
    ) -> None:
        """Create vector store table with optional full-text search and namespace.

        Retries transient errors under the engine's retry policy.

        Args:
            table_name: Name of the table to create
            vector_dimension: Dimension of vector embeddings
            schema: Database schema (default: public)
            id_type: Type for ID column (default: UUID)
            content_column: Name of content column
            embedding_column: Name of embedding column
            metadata_column: Name of metadata column
            namespace_column: Name of namespace column for multi-tenancy (default: None, disabled)
            create_tsvector: Create TSVECTOR column for FTS
            fts_language: Text search config for the generated tsvector column
                (default: english). Must match HybridSearchConfig.fts_query_language
                when hybrid search is used.
            drop_if_exists: Drop table if it exists
        """
        await self._run_as_async(
            self._init_vectorstore_table(
                table_name,
                vector_dimension,
                schema=schema,
                id_type=id_type,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_column=metadata_column,
                namespace_column=namespace_column,
                create_tsvector=create_tsvector,
                fts_language=fts_language,
                drop_if_exists=drop_if_exists,
            )
        )

    def init_vectorstore_table(self, table_name: str, vector_dimension: int, **kwargs: Any) -> None:
        """Sync version of ainit_vectorstore_table."""
        self._run_as_sync(self._init_vectorstore_table(table_name, vector_dimension, **kwargs))

    async def _init_vectorstore_table(
        self,
        table_name: str,
        vector_dimension: int,
        *,
        schema: str = "public",
        id_type: str = "UUID",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
        namespace_column: str | None = None,
        create_tsvector: bool = False,
        fts_language: str = "english",
        drop_if_exists: bool = False,
    ) -> None:
        validate_identifier(table_name, what="table name")
        validate_identifier(schema, what="schema")
        for column in (content_column, embedding_column, metadata_column):
            validate_identifier(column, what="column name")
        if namespace_column is not None:
            validate_identifier(namespace_column, what="column name")
        if not _ID_TYPE.fullmatch(id_type):
            raise ValueError(f"Invalid id_type {id_type!r}")
        if (
            isinstance(vector_dimension, bool)
            or not isinstance(vector_dimension, int)
            or vector_dimension < 1
        ):
            raise ValueError("vector_dimension must be a positive integer")
        validate_fts_language(fts_language)

        fqn = f"{schema}.{table_name}"
        statements = []
        if drop_if_exists:
            statements.append(f"DROP TABLE IF EXISTS {fqn}")
        ns_col = f"{namespace_column} TEXT NOT NULL DEFAULT ''," if namespace_column else ""
        statements.append(
            f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                id {id_type} PRIMARY KEY DEFAULT gen_random_uuid(),
                {ns_col}
                {content_column} TEXT,
                {embedding_column} VECTOR({vector_dimension}),
                {metadata_column} JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        if namespace_column:
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {table_name}_{namespace_column}_idx "
                f"ON {fqn} ({namespace_column})"
            )
        if create_tsvector:
            tsvector_col = f"{content_column}_tsvector"
            statements.append(
                f"ALTER TABLE {fqn} ADD COLUMN IF NOT EXISTS {tsvector_col} TSVECTOR "
                f"GENERATED ALWAYS AS (to_tsvector('{fts_language}', {content_column})) STORED"
            )
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {table_name}_{tsvector_col}_idx "
                f"ON {fqn} USING GIN ({tsvector_col})"
            )
        await self._run_ddl(statements, label=f"create table {fqn}")

    async def adrop_table(self, table_name: str, *, schema: str = "public") -> None:
        """Drop a table.

        Args:
            table_name: Name of the table to drop
            schema: Database schema (default: public)
        """
        await self._run_as_async(self._drop_table(table_name, schema=schema))

    def drop_table(self, table_name: str, **kwargs: Any) -> None:
        """Sync version of adrop_table."""
        self._run_as_sync(self._drop_table(table_name, **kwargs))

    async def _drop_table(self, table_name: str, *, schema: str = "public") -> None:
        validate_identifier(table_name, what="table name")
        validate_identifier(schema, what="schema")
        fqn = f"{schema}.{table_name}"
        await self._run_ddl([f"DROP TABLE IF EXISTS {fqn}"], label=f"drop table {fqn}")

    async def aclose(self) -> None:
        """Dispose the connection pool."""
        await self._run_as_async(self._engine.dispose())

    def close(self) -> None:
        """Sync version of aclose."""
        self._run_as_sync(self._engine.dispose())

    async def __aenter__(self) -> CockroachDBEngine:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()


_ENGINES: weakref.WeakSet[CockroachDBEngine] = weakref.WeakSet()


def _after_fork_in_child() -> None:
    """Give every engine a fresh pool in a forked child.

    ``dispose(close=False)`` drops the inherited connections without closing
    them, and psycopg only finishes a connection in the process that opened
    it, so the parent's sessions keep working.
    """
    for engine in list(_ENGINES):
        engine._usage_lock = threading.Lock()
        engine._active = {"sync": 0, "async": 0}
        engine.engine.sync_engine.dispose(close=False)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
