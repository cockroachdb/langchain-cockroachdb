"""Integration tests for CockroachDBEngine."""

import pytest
from sqlalchemy import text

from langchain_cockroachdb.engine import CockroachDBEngine


@pytest.mark.asyncio
class TestCockroachDBEngine:
    """Test CockroachDBEngine with real database."""

    async def test_from_connection_string(self, connection_string: str) -> None:
        """Test engine creation from connection string."""
        engine = CockroachDBEngine.from_connection_string(connection_string)

        assert engine is not None
        assert engine.engine is not None

        await engine.aclose()

    async def test_from_engine(self, async_engine) -> None:
        """Test engine creation from existing engine."""
        engine = CockroachDBEngine.from_engine(async_engine)

        assert engine is not None
        assert engine.engine == async_engine

    async def test_ainit_vectorstore_table(self, cockroachdb_engine: CockroachDBEngine) -> None:
        """Test vector store table creation."""
        table_name = "test_vectors"
        vector_dim = 3

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=vector_dim,
            drop_if_exists=True,
        )

        async with cockroachdb_engine.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """),
                {"table_name": table_name},
            )
            columns = {row[0]: row[1] for row in result.fetchall()}

        assert "id" in columns
        assert "content" in columns
        assert "embedding" in columns
        assert "metadata" in columns
        assert columns["metadata"] == "jsonb"

    async def test_ainit_vectorstore_table_with_tsvector(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """Test table creation with full-text search support."""
        table_name = "test_fts_vectors"
        vector_dim = 3

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=vector_dim,
            create_tsvector=True,
            drop_if_exists=True,
        )

        async with cockroachdb_engine.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    AND column_name LIKE '%tsvector%'
                """),
                {"table_name": table_name},
            )
            columns = result.fetchall()

        assert len(columns) > 0
        assert columns[0][1] == "tsvector"

    async def test_context_manager(self, connection_string: str) -> None:
        """Test async context manager."""
        async with CockroachDBEngine.from_connection_string(connection_string) as engine:
            assert engine is not None

            async with engine.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

    async def test_connection_pooling(self, connection_string: str) -> None:
        """Test connection pool configuration."""
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            pool_size=5,
            max_overflow=10,
        )

        assert engine.engine.pool.size() == 5

        await engine.aclose()
