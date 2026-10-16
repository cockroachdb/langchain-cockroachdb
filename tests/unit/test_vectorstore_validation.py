"""Unit tests for input validation in vector store operations."""

import math

import pytest

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CockroachDBVectorStore,
)

UNREACHABLE = "cockroachdb://root@localhost:1/none"


class CountingEmbeddings:
    """Embedding model that counts how many times it's called."""

    def __init__(self, size: int = 8):
        self.size = size
        self.embed_documents_calls = 0
        self.embed_query_calls = 0
        self.sync_embed_documents_calls = 0
        self.sync_embed_query_calls = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Count calls and return fake embeddings."""
        self.embed_documents_calls += 1
        return [[0.0] * self.size for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Count calls and return fake embedding."""
        self.embed_query_calls += 1
        return [0.0] * self.size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Count sync calls and return fake embeddings."""
        self.sync_embed_documents_calls += 1
        return [[0.0] * self.size for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Count sync calls and return fake embedding."""
        self.sync_embed_query_calls += 1
        return [0.0] * self.size


class TestValidationBeforeEmbedding:
    """Test that aadd_texts validates input before calling the embedding model."""

    @pytest.mark.asyncio
    async def test_duplicate_ids_validated_before_embedding(self) -> None:
        """Duplicate IDs should be rejected before calling aembed_documents."""
        embeddings = CountingEmbeddings()
        engine = CockroachDBEngine.from_connection_string("cockroachdb://root@localhost:1/none")
        try:
            store = AsyncCockroachDBVectorStore(engine, embeddings, "test_table")

            with pytest.raises(ValueError, match="Duplicate id"):
                await store.aadd_texts(["text1", "text2"], ids=["same-id", "same-id"])

            assert embeddings.embed_documents_calls == 0
        finally:
            engine.close()

    @pytest.mark.asyncio
    async def test_mismatched_lengths_validated_before_embedding(self) -> None:
        """Mismatched lengths should be rejected before calling aembed_documents."""
        embeddings = CountingEmbeddings()
        engine = CockroachDBEngine.from_connection_string("cockroachdb://root@localhost:1/none")
        try:
            store = AsyncCockroachDBVectorStore(engine, embeddings, "test_table")

            with pytest.raises(ValueError, match="Got 1 metadatas for 2 texts"):
                await store.aadd_texts(["text1", "text2"], metadatas=[{"key": "value"}])

            assert embeddings.embed_documents_calls == 0

            with pytest.raises(ValueError, match="Got 1 ids for 2 texts"):
                await store.aadd_texts(["text1", "text2"], ids=["id1"])

            assert embeddings.embed_documents_calls == 0
        finally:
            engine.close()


# Each case is (texts, keyword arguments, expected error, message pattern)
INVALID_ADDS = [
    (["a", "b"], {"ids": ["same", "same"]}, ValueError, "Duplicate id"),
    (["a", "b"], {"metadatas": [{}]}, ValueError, "Got 1 metadatas for 2 texts"),
    (["a", "b"], {"ids": ["x"]}, ValueError, "Got 1 ids for 2 texts"),
    (["a"], {"metadatas": [{"when": object()}]}, TypeError, "not JSON serializable"),
    (["a"], {"metadatas": [{"score": math.nan}]}, TypeError, "not JSON serializable"),
    (["a"], {"batch_size": 0}, ValueError, "batch_size must be a positive integer"),
    (["a"], {"batch_size": 20_000}, ValueError, "exceeds the maximum"),
]


@pytest.mark.parametrize(("texts", "kwargs", "error", "pattern"), INVALID_ADDS)
def test_sync_add_texts_validates_before_embedding(texts, kwargs, error, pattern) -> None:
    embeddings = CountingEmbeddings()
    engine = CockroachDBEngine.from_connection_string(UNREACHABLE)
    try:
        store = CockroachDBVectorStore(engine, embeddings, "test_table")
        with pytest.raises(error, match=pattern):
            store.add_texts(texts, **kwargs)
        assert embeddings.sync_embed_documents_calls == 0
    finally:
        engine.close()


@pytest.mark.parametrize(("texts", "kwargs", "error", "pattern"), INVALID_ADDS)
async def test_async_add_texts_validates_before_embedding(texts, kwargs, error, pattern) -> None:
    embeddings = CountingEmbeddings()
    engine = CockroachDBEngine.from_connection_string(UNREACHABLE)
    try:
        store = AsyncCockroachDBVectorStore(engine, embeddings, "test_table")
        with pytest.raises(error, match=pattern):
            await store.aadd_texts(texts, **kwargs)
        assert embeddings.embed_documents_calls == 0
    finally:
        engine.close()


def test_sync_from_texts_validates_before_embedding() -> None:
    embeddings = CountingEmbeddings()
    engine = CockroachDBEngine.from_connection_string(UNREACHABLE)
    try:
        with pytest.raises(ValueError, match="Duplicate id"):
            CockroachDBVectorStore.from_texts(
                ["a", "b"], embeddings, ids=["same", "same"], engine=engine
            )
        with pytest.raises(TypeError, match="not JSON serializable"):
            CockroachDBVectorStore.from_texts(
                ["a"], embeddings, metadatas=[{"when": object()}], engine=engine
            )
        assert embeddings.sync_embed_documents_calls == 0
        assert embeddings.sync_embed_query_calls == 0
    finally:
        engine.close()


async def test_async_from_texts_validates_before_embedding() -> None:
    embeddings = CountingEmbeddings()
    engine = CockroachDBEngine.from_connection_string(UNREACHABLE)
    try:
        with pytest.raises(ValueError, match="Duplicate id"):
            await AsyncCockroachDBVectorStore.afrom_texts(
                ["a", "b"], embeddings, ids=["same", "same"], engine=engine
            )
        with pytest.raises(TypeError, match="not JSON serializable"):
            await AsyncCockroachDBVectorStore.afrom_texts(
                ["a"], embeddings, metadatas=[{"when": object()}], engine=engine
            )
        assert embeddings.embed_documents_calls == 0
        assert embeddings.embed_query_calls == 0
    finally:
        engine.close()


def test_store_retry_jitter_names_the_argument() -> None:
    engine = CockroachDBEngine.from_connection_string(UNREACHABLE)
    try:
        with pytest.raises(TypeError, match=r"^retry_jitter must be"):
            AsyncCockroachDBVectorStore(engine, CountingEmbeddings(), "t", retry_jitter=0.3)
        store = AsyncCockroachDBVectorStore(engine, CountingEmbeddings(), "t", retry_jitter=0)
        assert store.retry_policy.jitter == 0.0
    finally:
        engine.close()
