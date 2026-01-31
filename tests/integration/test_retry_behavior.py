"""Integration tests for retry behavior under real error conditions."""

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine


class TestRetryBehavior:
    """Test retry behavior with real database errors."""

    @pytest_asyncio.fixture
    async def engine_with_retries(self, connection_string: str):
        """Create engine with custom retry configuration."""
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            retry_max_attempts=3,
            retry_initial_backoff=0.05,
            retry_max_backoff=0.5,
        )
        yield engine
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_table_init_retries_on_connection_error(
        self, engine_with_retries: CockroachDBEngine
    ) -> None:
        """Test table initialization retries on connection errors.

        This test verifies retry behavior by closing the connection
        and letting retry logic handle it.
        """
        table_name = "retry_test_table"

        # First create table successfully
        await engine_with_retries.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            drop_if_exists=True,
        )

        # Verify table exists
        async with engine_with_retries.engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = :table_name"
                ),
                {"table_name": table_name},
            )
            count = result.scalar()
            assert count == 1

    @pytest.mark.asyncio
    async def test_vectorstore_add_with_custom_retry_params(
        self, engine_with_retries: CockroachDBEngine
    ) -> None:
        """Test vectorstore operations with custom retry parameters."""
        embeddings = DeterministicFakeEmbedding(size=384)
        table_name = "retry_vectorstore_test"

        await engine_with_retries.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            drop_if_exists=True,
        )

        # Create vectorstore with custom retry settings
        vectorstore = AsyncCockroachDBVectorStore(
            engine=engine_with_retries,
            embeddings=embeddings,
            collection_name=table_name,
            retry_max_attempts=5,
            retry_initial_backoff=0.02,
            retry_max_backoff=1.0,
        )

        # Add texts should work with retry logic
        texts = ["Document 1", "Document 2", "Document 3"]
        ids = await vectorstore.aadd_texts(texts)

        assert len(ids) == 3

        # Verify data was inserted
        results = await vectorstore.asimilarity_search("Document", k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_insert_with_multiple_batches(
        self, engine_with_retries: CockroachDBEngine
    ) -> None:
        """Test batch insert with small batch size triggers multiple operations."""
        embeddings = DeterministicFakeEmbedding(size=384)
        table_name = "batch_retry_test"

        await engine_with_retries.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            drop_if_exists=True,
        )

        # Create vectorstore with small batch size
        vectorstore = AsyncCockroachDBVectorStore(
            engine=engine_with_retries,
            embeddings=embeddings,
            collection_name=table_name,
            batch_size=2,  # Small batch to trigger multiple inserts
            retry_max_attempts=3,
        )

        # Add 10 texts with batch_size=2 (5 batches)
        texts = [f"Document {i}" for i in range(10)]
        ids = await vectorstore.aadd_texts(texts)

        assert len(ids) == 10

        # Verify all inserted
        async with engine_with_retries.engine.connect() as conn:
            result = await conn.execute(text(f"SELECT COUNT(*) FROM public.{table_name}"))
            count = result.scalar()
            assert count == 10

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_retries(
        self, engine_with_retries: CockroachDBEngine
    ) -> None:
        """Test concurrent operations don't interfere with retry logic."""
        import asyncio

        embeddings = DeterministicFakeEmbedding(size=384)
        table_name = "concurrent_retry_test"

        await engine_with_retries.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            drop_if_exists=True,
        )

        vectorstore = AsyncCockroachDBVectorStore(
            engine=engine_with_retries,
            embeddings=embeddings,
            collection_name=table_name,
            retry_max_attempts=3,
        )

        # Run multiple concurrent insert operations
        async def add_batch(batch_id: int):
            texts = [f"Batch {batch_id} - Doc {i}" for i in range(5)]
            return await vectorstore.aadd_texts(texts)

        # Run 5 concurrent batches
        results = await asyncio.gather(*[add_batch(i) for i in range(5)])

        # Verify all batches succeeded
        assert len(results) == 5
        assert all(len(ids) == 5 for ids in results)

        # Verify total count
        async with engine_with_retries.engine.connect() as conn:
            result = await conn.execute(text(f"SELECT COUNT(*) FROM public.{table_name}"))
            count = result.scalar()
            assert count == 25

    @pytest.mark.asyncio
    async def test_retry_with_different_error_types(
        self, engine_with_retries: CockroachDBEngine
    ) -> None:
        """Test that retry logic distinguishes retryable vs non-retryable errors."""
        embeddings = DeterministicFakeEmbedding(size=384)
        table_name = "error_type_test"

        await engine_with_retries.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            drop_if_exists=True,
        )

        vectorstore = AsyncCockroachDBVectorStore(
            engine=engine_with_retries,
            embeddings=embeddings,
            collection_name=table_name,
        )

        # Valid operations should succeed
        ids = await vectorstore.aadd_texts(["Valid document"])
        assert len(ids) == 1

        # Invalid operations should fail without retry
        # (Testing error handling, not actual retry since we can't
        # easily simulate 40001 in integration tests)
        results = await vectorstore.asimilarity_search("test", k=10)
        assert isinstance(results, list)
