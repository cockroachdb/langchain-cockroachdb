"""Integration tests for vector store."""

import pytest
from langchain_core.embeddings import Embeddings

from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions, DistanceStrategy


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        return [[float(i), float(i + 1), float(i + 2)] for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed query."""
        return [1.0, 2.0, 3.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed query."""
        return self.embed_query(text)


@pytest.mark.asyncio
class TestAsyncCockroachDBVectorStore:
    """Test async vector store with real database."""

    @pytest.fixture
    async def vectorstore(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> AsyncCockroachDBVectorStore:
        """Create vector store for testing."""
        embeddings = FakeEmbeddings()
        collection_name = "test_collection"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_dimension=3,
            drop_if_exists=True,
        )

        return AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=collection_name,
        )

    async def test_aadd_texts(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test adding texts."""
        ids = await vectorstore.aadd_texts(sample_texts)

        assert len(ids) == len(sample_texts)
        assert all(isinstance(id, str) for id in ids)

    async def test_aadd_texts_with_metadata(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test adding texts with metadata."""
        ids = await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)

        assert len(ids) == len(sample_texts)

    async def test_aadd_texts_with_ids(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test adding texts with custom IDs."""
        import uuid

        custom_ids = [str(uuid.uuid4()) for _ in range(len(sample_texts))]
        ids = await vectorstore.aadd_texts(sample_texts, ids=custom_ids)

        assert ids == custom_ids

    async def test_asimilarity_search(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test similarity search."""
        await vectorstore.aadd_texts(sample_texts)

        results = await vectorstore.asimilarity_search("database", k=3)

        assert len(results) <= 3
        assert all(hasattr(doc, "page_content") for doc in results)

    async def test_asimilarity_search_with_score(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test similarity search with scores."""
        await vectorstore.aadd_texts(sample_texts)

        results = await vectorstore.asimilarity_search_with_score("database", k=3)

        assert len(results) <= 3
        assert all(isinstance(score, float) for _, score in results)

    async def test_asimilarity_search_with_filter(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test similarity search with metadata filter."""
        await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)

        filter_dict = {"category": {"$eq": "database"}}
        results = await vectorstore.asimilarity_search("query", k=5, filter=filter_dict)

        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("category") == "database"

    async def test_asimilarity_search_with_and_filter(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test similarity search with AND filter."""
        await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)

        filter_dict = {
            "$and": [
                {"category": {"$eq": "database"}},
                {"page": {"$gt": 2}},
            ]
        }
        results = await vectorstore.asimilarity_search("query", k=5, filter=filter_dict)

        for doc in results:
            assert doc.metadata.get("category") == "database"
            assert doc.metadata.get("page", 0) > 2

    async def test_adelete(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test deleting documents."""
        ids = await vectorstore.aadd_texts(sample_texts)

        delete_ids = ids[:2]
        result = await vectorstore.adelete(delete_ids)

        assert result is True

        remaining = await vectorstore.asimilarity_search("", k=10)
        assert len(remaining) == len(sample_texts) - len(delete_ids)

    async def test_aapply_vector_index(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test creating vector index."""
        await vectorstore.aadd_texts(sample_texts)

        index = CSPANNIndex(
            distance_strategy=DistanceStrategy.COSINE,
            min_partition_size=10,
        )

        await vectorstore.aapply_vector_index(index)

    async def test_query_with_beam_size(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test query with beam size option."""
        await vectorstore.aadd_texts(sample_texts)

        index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
        await vectorstore.aapply_vector_index(index)

        query_options = CSPANNQueryOptions(beam_size=100)
        results = await vectorstore.asimilarity_search_with_score(
            "database",
            k=3,
            query_options=query_options,
        )

        assert len(results) <= 3

    async def test_amax_marginal_relevance_search(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test MMR search."""
        await vectorstore.aadd_texts(sample_texts)

        results = await vectorstore.amax_marginal_relevance_search(
            "database",
            k=3,
            fetch_k=5,
            lambda_mult=0.5,
        )

        assert len(results) <= 3

    async def test_batch_insert_with_custom_batch_size(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
    ) -> None:
        """Test batch insert with custom batch size."""
        texts = [f"text_{i}" for i in range(250)]

        ids = await vectorstore.aadd_texts(texts, batch_size=50)

        assert len(ids) == 250

    async def test_afrom_texts(
        self,
        cockroachdb_engine: CockroachDBEngine,
        sample_texts: list[str],
    ) -> None:
        """Test creating vectorstore from texts."""
        embeddings = FakeEmbeddings()

        vectorstore = await AsyncCockroachDBVectorStore.afrom_texts(
            texts=sample_texts,
            embedding=embeddings,
            engine=cockroachdb_engine,
            collection_name="from_texts_test",
        )

        assert vectorstore is not None

        results = await vectorstore.asimilarity_search("database", k=3)
        assert len(results) <= 3

    async def test_multiple_distance_strategies(
        self,
        cockroachdb_engine: CockroachDBEngine,
        sample_texts: list[str],
    ) -> None:
        """Test different distance strategies."""
        embeddings = FakeEmbeddings()

        for strategy in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN,
            DistanceStrategy.INNER_PRODUCT,
        ]:
            collection_name = f"test_{strategy.value}"

            await cockroachdb_engine.ainit_vectorstore_table(
                table_name=collection_name,
                vector_dimension=3,
                drop_if_exists=True,
            )

            vectorstore = AsyncCockroachDBVectorStore(
                engine=cockroachdb_engine,
                embeddings=embeddings,
                collection_name=collection_name,
                distance_strategy=strategy,
            )

            await vectorstore.aadd_texts(sample_texts)

            results = await vectorstore.asimilarity_search("database", k=3)
            assert len(results) <= 3
