"""Vector store implementations with sync/async support."""

import asyncio
from collections.abc import Iterable
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.indexes import CSPANNQueryOptions


class CockroachDBVectorStore(AsyncCockroachDBVectorStore):
    """Sync wrapper for AsyncCockroachDBVectorStore using background event loop."""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to vector store (sync).

        Args:
            texts: Texts to add
            metadatas: Optional metadata
            ids: Optional IDs
            **kwargs: Additional arguments

        Returns:
            List of IDs
        """
        return asyncio.run(self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs))

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents (sync).

        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        return asyncio.run(self.asimilarity_search(query, k=k, filter=filter, **kwargs))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        query_options: Optional[CSPANNQueryOptions] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search with scores (sync).

        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
            query_options: Query options
            **kwargs: Additional arguments

        Returns:
            List of (document, score) tuples
        """
        return asyncio.run(
            self.asimilarity_search_with_score(
                query, k=k, filter=filter, query_options=query_options, **kwargs
            )
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """MMR search (sync).

        Args:
            query: Query text
            k: Number of results
            fetch_k: Candidate pool size
            lambda_mult: Diversity parameter
            filter: Metadata filter
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        return asyncio.run(
            self.amax_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
            )
        )

    def delete(
        self,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents (sync).

        Args:
            ids: Document IDs
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        return asyncio.run(self.adelete(ids=ids, **kwargs))

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        engine: Optional[CockroachDBEngine] = None,
        connection_string: Optional[str] = None,
        collection_name: str = "langchain_vectors",
        **kwargs: Any,
    ) -> "CockroachDBVectorStore":
        """Create from texts (sync).

        Args:
            texts: Texts to add
            embedding: Embeddings model
            metadatas: Optional metadata
            engine: CockroachDBEngine instance
            connection_string: Connection string
            collection_name: Table name
            **kwargs: Additional arguments

        Returns:
            CockroachDBVectorStore instance
        """
        return asyncio.run(
            AsyncCockroachDBVectorStore.afrom_texts(
                texts,
                embedding,
                metadatas=metadatas,
                engine=engine,
                connection_string=connection_string,
                collection_name=collection_name,
                **kwargs,
            )
        )
