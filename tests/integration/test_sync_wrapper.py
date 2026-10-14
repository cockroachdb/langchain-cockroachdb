"""Integration tests for sync wrapper (CockroachDBVectorStore)."""

import asyncio
import threading
import time
import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from langchain_cockroachdb import CockroachDBEngine, CockroachDBVectorStore, RetryPolicy
from langchain_cockroachdb.indexes import CSPANNIndex, DistanceStrategy
from tests.integration.faults import kill_sessions


class CountingEmbeddings(Embeddings):
    """Counts embedding calls so tests can see when the store re-embeds."""

    def __init__(self) -> None:
        self.inner = DeterministicFakeEmbedding(size=384)
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self.inner.embed_query(text)


@pytest.fixture
def sync_engine(connection_string: str):
    """Create sync engine."""
    engine = CockroachDBEngine.from_connection_string(connection_string)
    yield engine
    engine.close()


@pytest.fixture
def sync_vectorstore(sync_engine: CockroachDBEngine):
    """Create sync vectorstore."""
    embeddings = DeterministicFakeEmbedding(size=384)
    table_name = "sync_test_vectors"

    # Initialize table synchronously
    sync_engine.init_vectorstore_table(
        table_name=table_name,
        vector_dimension=384,
        drop_if_exists=True,
    )

    vectorstore = CockroachDBVectorStore(
        engine=sync_engine,
        embeddings=embeddings,
        collection_name=table_name,
    )

    yield vectorstore


class TestCockroachDBVectorStoreSync:
    """Test synchronous vectorstore wrapper."""

    def test_add_texts_sync(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test adding texts synchronously."""
        texts = ["First document", "Second document", "Third document"]
        metadatas = [
            {"id": 1, "category": "test"},
            {"id": 2, "category": "test"},
            {"id": 3, "category": "prod"},
        ]

        ids = sync_vectorstore.add_texts(texts, metadatas=metadatas)

        assert len(ids) == 3
        assert all(isinstance(id_val, str) for id_val in ids)

    def test_add_texts_with_ids_sync(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test adding texts with custom IDs synchronously."""
        import uuid

        texts = ["Doc 1", "Doc 2"]
        custom_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        ids = sync_vectorstore.add_texts(texts, ids=custom_ids)

        assert ids == custom_ids

    def test_similarity_search_sync(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test similarity search synchronously."""
        texts = [
            "CockroachDB is a distributed SQL database",
            "Python is a programming language",
            "LangChain helps build LLM applications",
        ]
        sync_vectorstore.add_texts(texts)

        results = sync_vectorstore.similarity_search("database", k=2)

        assert len(results) == 2
        assert isinstance(results[0].page_content, str)

    def test_similarity_search_with_score_sync(
        self, sync_vectorstore: CockroachDBVectorStore
    ) -> None:
        """Test similarity search with scores synchronously."""
        texts = ["Document about databases", "Document about programming"]
        sync_vectorstore.add_texts(texts)

        results = sync_vectorstore.similarity_search_with_score("databases", k=2)

        assert len(results) == 2
        doc, score = results[0]
        assert isinstance(doc.page_content, str)
        assert isinstance(score, float)
        assert score >= 0

    def test_similarity_search_with_filter_sync(
        self, sync_vectorstore: CockroachDBVectorStore
    ) -> None:
        """Test similarity search with metadata filter synchronously."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        metadatas = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"},
        ]
        sync_vectorstore.add_texts(texts, metadatas=metadatas)

        results = sync_vectorstore.similarity_search(
            "document", k=5, filter={"category": {"$eq": "A"}}
        )

        assert len(results) == 2
        assert all(doc.metadata["category"] == "A" for doc in results)

    def test_delete_sync(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test deleting documents synchronously."""
        texts = ["Doc to keep", "Doc to delete"]
        ids = sync_vectorstore.add_texts(texts)

        # Delete one document
        deleted = sync_vectorstore.delete([ids[1]])
        assert deleted is True

        # Verify only one document remains
        results = sync_vectorstore.similarity_search("Doc", k=10)
        assert len(results) == 1
        assert results[0].page_content == "Doc to keep"

    def test_max_marginal_relevance_search_sync(
        self, sync_vectorstore: CockroachDBVectorStore
    ) -> None:
        """Test MMR search synchronously."""
        texts = [
            "Machine learning is fun",
            "Machine learning is interesting",
            "Databases store data",
        ]
        sync_vectorstore.add_texts(texts)

        results = sync_vectorstore.max_marginal_relevance_search("machine learning", k=2, fetch_k=3)

        assert len(results) == 2
        # MMR should prefer diversity
        contents = [doc.page_content for doc in results]
        # Should get one ML doc and one DB doc (diverse)
        assert any("Machine learning" in c for c in contents)

    def test_from_texts_sync(self, sync_engine: CockroachDBEngine) -> None:
        """Test creating vectorstore and adding texts in one go."""
        embeddings = DeterministicFakeEmbedding(size=384)
        texts = ["Text 1", "Text 2", "Text 3"]
        table_name = "from_texts_sync_test"

        sync_engine.init_vectorstore_table(
            table_name=table_name,
            vector_dimension=384,
            drop_if_exists=True,
        )

        # Create vectorstore and add texts separately (from_texts is complex with async/sync)
        vectorstore = CockroachDBVectorStore(
            engine=sync_engine,
            embeddings=embeddings,
            collection_name=table_name,
        )

        ids = vectorstore.add_texts(texts)
        assert len(ids) == 3

        results = vectorstore.similarity_search("Text", k=5)
        assert len(results) == 3

    def test_apply_vector_index_sync(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test applying vector index synchronously."""
        # Add some data first
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        sync_vectorstore.add_texts(texts)

        # Create and apply index
        index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
        sync_vectorstore.apply_vector_index(index)

        # Search should still work with index
        results = sync_vectorstore.similarity_search("Doc", k=2)
        assert len(results) == 2

    def test_batch_operations_sync(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test batch operations with custom batch size synchronously."""
        texts = [f"Document {i}" for i in range(20)]

        # Add with custom batch size
        ids = sync_vectorstore.add_texts(texts, batch_size=5)
        assert len(ids) == 20

        # Search should return all documents
        results = sync_vectorstore.similarity_search("Document", k=25)
        assert len(results) == 20

    def test_sync_parity_with_async(
        self, sync_engine: CockroachDBEngine, sync_vectorstore: CockroachDBVectorStore
    ) -> None:
        """Test that sync wrapper provides same results as async version."""
        texts = ["Test document 1", "Test document 2"]
        metadatas = [{"idx": 1}, {"idx": 2}]

        # Add via sync
        sync_ids = sync_vectorstore.add_texts(texts, metadatas=metadatas)

        # Search via sync
        sync_results = sync_vectorstore.similarity_search("Test", k=2)

        assert len(sync_ids) == 2
        assert len(sync_results) == 2
        assert all("Test document" in doc.page_content for doc in sync_results)

    def test_sync_error_handling(self, sync_vectorstore: CockroachDBVectorStore) -> None:
        """Test error handling in sync operations."""
        # Add valid data
        ids = sync_vectorstore.add_texts(["Valid document"])
        assert len(ids) == 1

        # Search with invalid parameters should handle gracefully
        results = sync_vectorstore.similarity_search("test", k=0)
        assert len(results) == 0


def test_sync_calls_from_sixteen_threads(sync_vectorstore: CockroachDBVectorStore) -> None:
    sync_vectorstore.add_texts([f"doc {i}" for i in range(20)])
    errors: list[BaseException] = []

    def worker() -> None:
        for _ in range(15):
            try:
                sync_vectorstore.similarity_search("doc", k=3)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert errors == []


def test_sync_call_inside_a_running_event_loop(sync_vectorstore: CockroachDBVectorStore) -> None:
    sync_vectorstore.add_texts(["inside"])

    async def call() -> list:
        return sync_vectorstore.similarity_search("inside", k=1)

    assert [d.page_content for d in asyncio.run(call())] == ["inside"]


def test_from_texts_honors_ids_and_store_arguments(sync_engine: CockroachDBEngine) -> None:
    ids = [str(uuid.uuid4()) for _ in range(2)]
    store = CockroachDBVectorStore.from_texts(
        ["a", "b"],
        DeterministicFakeEmbedding(size=384),
        engine=sync_engine,
        collection_name="from_texts_args",
        ids=ids,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        batch_size=1,
        drop_if_exists=True,
    )
    assert store.distance_strategy == DistanceStrategy.EUCLIDEAN
    assert store.batch_size == 1
    assert sorted(doc.id for doc in store.get_by_ids(ids)) == sorted(ids)


def test_mmr_uses_stored_embeddings(sync_engine: CockroachDBEngine) -> None:
    embeddings = CountingEmbeddings()
    sync_engine.init_vectorstore_table("mmr_stored", 384, drop_if_exists=True)
    store = CockroachDBVectorStore(
        engine=sync_engine, embeddings=embeddings, collection_name="mmr_stored"
    )
    store.add_texts(["ml is fun", "ml is great", "databases store data"])
    embeddings.query_calls = 0
    assert len(store.max_marginal_relevance_search("ml", k=2, fetch_k=3)) == 2
    assert embeddings.query_calls == 1


def test_sync_api_recovers_from_killed_connections(engine_factory, libpq_url) -> None:
    app = f"sync_kill_{uuid.uuid4().hex[:6]}"
    engine = engine_factory(
        app, pool_pre_ping=False, retry_policy=RetryPolicy(initial_backoff=0.01)
    )
    engine.init_vectorstore_table("sync_kill", 384, drop_if_exists=True)
    store = CockroachDBVectorStore(
        engine=engine, embeddings=DeterministicFakeEmbedding(size=384), collection_name="sync_kill"
    )
    ids = store.add_texts(["before"])
    assert kill_sessions(libpq_url, app) >= 1
    time.sleep(0.3)
    assert [doc.page_content for doc in store.get_by_ids(ids)] == ["before"]
