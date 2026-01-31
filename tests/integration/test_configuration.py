"""Integration tests for configuration parameters."""

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine


class TestEngineConfiguration:
    """Test engine configuration parameters."""

    @pytest.mark.asyncio
    async def test_custom_pool_size(self, connection_string: str) -> None:
        """Test custom connection pool configuration."""
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

        # Verify engine works with custom pool settings
        await engine.ainit_vectorstore_table(
            table_name="pool_test",
            vector_dimension=384,
            drop_if_exists=True,
        )

        await engine.aclose()

    @pytest.mark.asyncio
    async def test_custom_pool_timeout(self, connection_string: str) -> None:
        """Test custom pool timeout configuration."""
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            pool_size=3,
            pool_timeout=5.0,
            pool_recycle=1800,
        )

        await engine.ainit_vectorstore_table(
            table_name="timeout_test",
            vector_dimension=384,
            drop_if_exists=True,
        )

        await engine.aclose()

    @pytest.mark.asyncio
    async def test_custom_retry_configuration(self, connection_string: str) -> None:
        """Test custom retry parameters."""
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            retry_max_attempts=10,
            retry_initial_backoff=0.05,
            retry_max_backoff=30.0,
            retry_backoff_multiplier=3.0,
            retry_jitter=False,
        )

        # Verify custom retry params are stored
        assert engine.retry_max_attempts == 10
        assert engine.retry_initial_backoff == 0.05
        assert engine.retry_max_backoff == 30.0
        assert engine.retry_backoff_multiplier == 3.0
        assert engine.retry_jitter is False

        await engine.aclose()

    @pytest.mark.asyncio
    async def test_minimal_configuration(self, connection_string: str) -> None:
        """Test engine with minimal (default) configuration."""
        engine = CockroachDBEngine.from_connection_string(connection_string)

        # Verify defaults
        assert engine.retry_max_attempts == 5
        assert engine.retry_initial_backoff == 0.1
        assert engine.retry_max_backoff == 10.0
        assert engine.retry_backoff_multiplier == 2.0
        assert engine.retry_jitter is True

        await engine.aclose()


class TestVectorStoreConfiguration:
    """Test vectorstore configuration parameters."""

    @pytest_asyncio.fixture
    async def configured_engine(self, connection_string: str):
        """Create engine for vectorstore tests."""
        engine = CockroachDBEngine.from_connection_string(connection_string)
        await engine.ainit_vectorstore_table(
            table_name="config_test_vectors",
            vector_dimension=384,
            drop_if_exists=True,
        )
        yield engine
        await engine.aclose()

    @pytest.mark.asyncio
    async def test_custom_batch_size(self, configured_engine: CockroachDBEngine) -> None:
        """Test custom batch size configuration."""
        embeddings = DeterministicFakeEmbedding(size=384)

        vectorstore = AsyncCockroachDBVectorStore(
            engine=configured_engine,
            embeddings=embeddings,
            collection_name="config_test_vectors",
            batch_size=10,
        )

        assert vectorstore.batch_size == 10

        # Add texts with custom batch size
        texts = [f"Doc {i}" for i in range(25)]
        ids = await vectorstore.aadd_texts(texts)
        assert len(ids) == 25

    @pytest.mark.asyncio
    async def test_custom_retry_params_vectorstore(
        self, configured_engine: CockroachDBEngine
    ) -> None:
        """Test custom retry parameters for vectorstore operations."""
        embeddings = DeterministicFakeEmbedding(size=384)

        vectorstore = AsyncCockroachDBVectorStore(
            engine=configured_engine,
            embeddings=embeddings,
            collection_name="config_test_vectors",
            retry_max_attempts=7,
            retry_initial_backoff=0.02,
            retry_max_backoff=20.0,
            retry_backoff_multiplier=2.5,
            retry_jitter=False,
        )

        # Verify params stored
        assert vectorstore.retry_max_attempts == 7
        assert vectorstore.retry_initial_backoff == 0.02
        assert vectorstore.retry_max_backoff == 20.0
        assert vectorstore.retry_backoff_multiplier == 2.5
        assert vectorstore.retry_jitter is False

        # Operations should work with custom retry config
        texts = ["Test doc 1", "Test doc 2"]
        ids = await vectorstore.aadd_texts(texts)
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_custom_column_names(self, configured_engine: CockroachDBEngine) -> None:
        """Test custom column name configuration."""
        embeddings = DeterministicFakeEmbedding(size=384)

        # Create table with custom columns
        table_name = "custom_columns_test"
        await configured_engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            content_column="text_content",
            embedding_column="vector_data",
            metadata_column="meta",
            drop_if_exists=True,
        )

        vectorstore = AsyncCockroachDBVectorStore(
            engine=configured_engine,
            embeddings=embeddings,
            collection_name=table_name,
            content_column="text_content",
            embedding_column="vector_data",
            metadata_column="meta",
        )

        # Operations should work with custom columns
        texts = ["Custom column test"]
        ids = await vectorstore.aadd_texts(texts)
        assert len(ids) == 1

        results = await vectorstore.asimilarity_search("test", k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_different_batch_sizes(self, configured_engine: CockroachDBEngine) -> None:
        """Test operations with different batch sizes."""
        embeddings = DeterministicFakeEmbedding(size=384)

        # Small batch size
        vs_small = AsyncCockroachDBVectorStore(
            engine=configured_engine,
            embeddings=embeddings,
            collection_name="config_test_vectors",
            batch_size=2,
        )

        texts = [f"Small batch {i}" for i in range(10)]
        ids_small = await vs_small.aadd_texts(texts)
        assert len(ids_small) == 10

        # Large batch size
        vs_large = AsyncCockroachDBVectorStore(
            engine=configured_engine,
            embeddings=embeddings,
            collection_name="config_test_vectors",
            batch_size=100,
        )

        texts_large = [f"Large batch {i}" for i in range(10)]
        ids_large = await vs_large.aadd_texts(texts_large)
        assert len(ids_large) == 10

    @pytest.mark.asyncio
    async def test_override_batch_size_at_runtime(
        self, configured_engine: CockroachDBEngine
    ) -> None:
        """Test overriding batch size at runtime."""
        embeddings = DeterministicFakeEmbedding(size=384)

        vectorstore = AsyncCockroachDBVectorStore(
            engine=configured_engine,
            embeddings=embeddings,
            collection_name="config_test_vectors",
            batch_size=10,  # Default
        )

        # Override batch size in add_texts call
        texts = [f"Runtime batch {i}" for i in range(20)]
        ids = await vectorstore.aadd_texts(texts, batch_size=5)
        assert len(ids) == 20

    @pytest.mark.asyncio
    async def test_production_configuration(self, connection_string: str) -> None:
        """Test production-ready configuration."""
        # Production-style configuration
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_timeout=30.0,
            retry_max_attempts=10,
            retry_initial_backoff=0.1,
            retry_max_backoff=30.0,
            retry_backoff_multiplier=2.0,
            retry_jitter=True,
        )

        await engine.ainit_vectorstore_table(
            table_name="production_test",
            vector_dimension=384,
            drop_if_exists=True,
        )

        embeddings = DeterministicFakeEmbedding(size=384)

        vectorstore = AsyncCockroachDBVectorStore(
            engine=engine,
            embeddings=embeddings,
            collection_name="production_test",
            batch_size=100,
            retry_max_attempts=5,
            retry_initial_backoff=0.05,
        )

        # Should work with production config
        texts = [f"Production doc {i}" for i in range(50)]
        ids = await vectorstore.aadd_texts(texts)
        assert len(ids) == 50

        await engine.aclose()
