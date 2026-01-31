"""CockroachDB async engine management with transaction retry support."""

import asyncio
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from langchain_cockroachdb.retry import async_retry_with_backoff


class CockroachDBEngine:
    """Manages async SQLAlchemy engine for CockroachDB with retry support."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        retry_max_attempts: int = 5,
        retry_initial_backoff: float = 0.1,
        retry_max_backoff: float = 10.0,
        retry_backoff_multiplier: float = 2.0,
        retry_jitter: bool = True,
    ):
        """Initialize with existing async engine.

        Args:
            engine: SQLAlchemy AsyncEngine connected to CockroachDB
            retry_max_attempts: Maximum retry attempts (default: 5)
            retry_initial_backoff: Initial backoff delay in seconds (default: 0.1)
            retry_max_backoff: Maximum backoff delay in seconds (default: 10.0)
            retry_backoff_multiplier: Backoff multiplier (default: 2.0)
            retry_jitter: Add randomization to backoff (default: True)
        """
        self._engine = engine
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_backoff = retry_initial_backoff
        self.retry_max_backoff = retry_max_backoff
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.retry_jitter = retry_jitter

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
        retry_max_attempts: int = 5,
        retry_initial_backoff: float = 0.1,
        retry_max_backoff: float = 10.0,
        retry_backoff_multiplier: float = 2.0,
        retry_jitter: bool = True,
        **kwargs: Any,
    ) -> "CockroachDBEngine":
        """Create engine from connection string.

        Args:
            connection_string: CockroachDB connection URL
            pool_size: Connection pool size (default: 10)
            max_overflow: Max connections beyond pool_size (default: 20)
            pool_pre_ping: Enable connection health checks (default: True)
            pool_recycle: Recycle connections after N seconds (default: 3600)
            pool_timeout: Connection timeout in seconds (default: 30.0)
            retry_max_attempts: Maximum retry attempts (default: 5)
            retry_initial_backoff: Initial backoff delay in seconds (default: 0.1)
            retry_max_backoff: Maximum backoff delay in seconds (default: 10.0)
            retry_backoff_multiplier: Backoff multiplier (default: 2.0)
            retry_jitter: Add randomization to backoff (default: True)
            **kwargs: Additional arguments for create_async_engine

        Returns:
            CockroachDBEngine instance
        """
        # Ensure we use the async driver (psycopg, not psycopg2)
        if connection_string.startswith("cockroachdb://"):
            connection_string = connection_string.replace(
                "cockroachdb://", "cockroachdb+psycopg://", 1
            )

        engine = create_async_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            pool_timeout=pool_timeout,
            **kwargs,
        )
        return cls(
            engine,
            retry_max_attempts=retry_max_attempts,
            retry_initial_backoff=retry_initial_backoff,
            retry_max_backoff=retry_max_backoff,
            retry_backoff_multiplier=retry_backoff_multiplier,
            retry_jitter=retry_jitter,
        )

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> "CockroachDBEngine":
        """Create from existing AsyncEngine.

        Args:
            engine: SQLAlchemy AsyncEngine

        Returns:
            CockroachDBEngine instance
        """
        return cls(engine)

    @property
    def engine(self) -> AsyncEngine:
        """Get underlying AsyncEngine."""
        return self._engine

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for sync operations."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine from sync context."""
        loop = self._get_loop()
        try:
            loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

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
        create_tsvector: bool = False,
        drop_if_exists: bool = False,
    ) -> None:
        """Create vector store table with optional full-text search.

        Uses retry logic configured on engine instance.

        Args:
            table_name: Name of the table to create
            vector_dimension: Dimension of vector embeddings
            schema: Database schema (default: public)
            id_type: Type for ID column (default: UUID)
            content_column: Name of content column
            embedding_column: Name of embedding column
            metadata_column: Name of metadata column
            create_tsvector: Create TSVECTOR column for FTS
            drop_if_exists: Drop table if it exists
        """

        # Apply retry with instance configuration
        @async_retry_with_backoff(
            max_retries=self.retry_max_attempts,
            initial_backoff=self.retry_initial_backoff,
            max_backoff=self.retry_max_backoff,
            backoff_multiplier=self.retry_backoff_multiplier,
            jitter=self.retry_jitter,
        )
        async def _create_table() -> None:
            fqn = f"{schema}.{table_name}"

            async with self._engine.begin() as conn:
                if drop_if_exists:
                    await conn.execute(text(f"DROP TABLE IF EXISTS {fqn}"))

                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {fqn} (
                        id {id_type} PRIMARY KEY DEFAULT gen_random_uuid(),
                        {content_column} TEXT,
                        {embedding_column} VECTOR({vector_dimension}),
                        {metadata_column} JSONB DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT now()
                    )
                """
                await conn.execute(text(create_sql))

                if create_tsvector:
                    tsvector_col = f"{content_column}_tsvector"
                    alter_sql = f"""
                        ALTER TABLE {fqn} 
                        ADD COLUMN IF NOT EXISTS {tsvector_col} TSVECTOR 
                        GENERATED ALWAYS AS (to_tsvector('english', {content_column})) STORED
                    """
                    await conn.execute(text(alter_sql))

                    index_sql = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_{tsvector_col}_idx 
                        ON {fqn} USING GIN ({tsvector_col})
                    """
                    await conn.execute(text(index_sql))

        await _create_table()

    def init_vectorstore_table(
        self,
        table_name: str,
        vector_dimension: int,
        **kwargs: Any,
    ) -> None:
        """Sync wrapper for ainit_vectorstore_table."""
        self._run_async(self.ainit_vectorstore_table(table_name, vector_dimension, **kwargs))

    async def aclose(self) -> None:
        """Close async engine."""
        await self._engine.dispose()

    def close(self) -> None:
        """Sync wrapper for aclose."""
        self._run_async(self.aclose())

    async def __aenter__(self) -> "CockroachDBEngine":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()
