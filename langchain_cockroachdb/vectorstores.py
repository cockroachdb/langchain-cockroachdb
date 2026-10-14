"""Vector store with both sync and async methods."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_cockroachdb.async_vectorstore import (
    AsyncCockroachDBVectorStore,
    split_from_texts_args,
)
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions


class CockroachDBVectorStore(AsyncCockroachDBVectorStore):
    """Vector store with sync and async methods.

    Sync methods compute embeddings on the calling thread and run their
    database work on the engine's background event loop, so they are safe to
    call from any number of threads and from inside a running event loop.
    """

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to vector store (sync).

        Args:
            texts: Texts to add
            metadatas: Optional metadata
            ids: Optional IDs
            **kwargs: Additional arguments (batch_size override supported)

        Returns:
            List of IDs
        """
        texts_list = list(texts)
        if not texts_list:
            return []
        embeddings = self._embeddings.embed_documents(texts_list)
        return self.add_embeddings(texts_list, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts with precomputed embeddings (sync)."""
        rows = self._prepare_rows(list(texts), embeddings, metadatas, ids, kwargs)
        return self.engine._run_as_sync(self._add_embeddings(rows))

    def similarity_search(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        """Search for similar documents (sync)."""
        return [
            doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        query_options: CSPANNQueryOptions | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search with scores (sync); fused scores when hybrid search is configured."""
        embedding = self._embeddings.embed_query(query)
        if self.hybrid_search_config is not None:
            return self.engine._run_as_sync(
                self._hybrid_search(
                    query, embedding, k, filter, query_options, kwargs.get("fetch_k")
                )
            )
        return self.similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, query_options=query_options
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
        query_options: CSPANNQueryOptions | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search by embedding with distances (sync)."""
        matches = self.engine._run_as_sync(
            self._search_by_vector(embedding, k, filter, query_options)
        )
        return [(match.document, match.distance) for match in matches]

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to embedding vector (sync)."""
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding, k=k, filter=filter, **kwargs
            )
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """MMR search (sync), using the candidates' stored embeddings."""
        embedding = self._embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """MMR search from a query embedding (sync)."""
        matches = self.engine._run_as_sync(
            self._search_by_vector(
                embedding, fetch_k, filter, kwargs.get("query_options"), with_embeddings=True
            )
        )
        return self._select_mmr(embedding, matches, k, lambda_mult)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents (sync)."""
        if not ids:
            return True
        self.engine._run_as_sync(self._delete(ids))
        return True

    def apply_vector_index(
        self, index: CSPANNIndex, prefix_columns: list[str] | None = None
    ) -> None:
        """Apply a vector index (sync); warns if an old index has the wrong operator class."""
        self.engine._run_as_sync(self._apply_vector_index(index, prefix_columns))

    def drop_vector_index(self, index: CSPANNIndex) -> None:
        """Drop a vector index (sync)."""
        self.engine._run_as_sync(self._drop_vector_index(index))

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        engine: CockroachDBEngine | None = None,
        connection_string: str | None = None,
        collection_name: str = "langchain_vectors",
        **kwargs: Any,
    ) -> CockroachDBVectorStore:
        """Create the table if needed, then add texts (sync).

        Args:
            texts: Texts to add
            embedding: Embeddings model
            metadatas: Optional metadata
            ids: Optional IDs
            engine: CockroachDBEngine instance
            connection_string: Connection string (if engine not provided)
            collection_name: Table name
            **kwargs: Table arguments for init_vectorstore_table and store arguments

        Returns:
            CockroachDBVectorStore instance
        """
        resolved = cls._engine_for(engine, connection_string)
        texts = list(texts)
        vectors = embedding.embed_documents(texts) if texts else []
        dimension = len(vectors[0]) if vectors else len(embedding.embed_query("dimension probe"))
        table_args, store_args = split_from_texts_args(kwargs)
        resolved.init_vectorstore_table(collection_name, dimension, **table_args)
        store = cls(
            engine=resolved, embeddings=embedding, collection_name=collection_name, **store_args
        )
        if texts:
            store.add_embeddings(texts, vectors, metadatas=metadatas, ids=ids)
        return store
