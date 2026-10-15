"""Unit tests for input validation in vector store operations."""

import pytest

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine


class CountingEmbeddings:
    """Embedding model that counts how many times it's called."""

    def __init__(self, size: int = 8):
        self.size = size
        self.embed_documents_calls = 0
        self.embed_query_calls = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Count calls and return fake embeddings."""
        self.embed_documents_calls += 1
        return [[0.0] * self.size for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Count calls and return fake embedding."""
        self.embed_query_calls += 1
        return [0.0] * self.size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sync version for compatibility."""
        return [[0.0] * self.size for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Sync version for compatibility."""
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
