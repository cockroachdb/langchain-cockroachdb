"""Async vector store implementation for CockroachDB with transaction retry support."""

import uuid
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import text

from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.hybrid_search_config import FusionType, HybridSearchConfig
from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions, DistanceStrategy


class AsyncCockroachDBVectorStore(VectorStore):
    """Async vector store using CockroachDB native VECTOR type and C-SPANN indexes."""

    def __init__(
        self,
        engine: CockroachDBEngine,
        embeddings: Embeddings,
        collection_name: str,
        *,
        schema: str = "public",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
        id_column: str = "id",
        namespace_column: str = "namespace",
        namespace: str | None = None,
        hybrid_search_config: HybridSearchConfig | None = None,
        batch_size: int = 100,
        retry_max_attempts: int = 3,
        retry_initial_backoff: float = 0.1,
        retry_max_backoff: float = 5.0,
        retry_backoff_multiplier: float = 2.0,
        retry_jitter: bool = True,
    ):
        """Initialize async vector store.

        Args:
            engine: CockroachDBEngine instance
            embeddings: Embeddings model
            collection_name: Table name for this collection
            schema: Database schema (default: public)
            distance_strategy: Distance metric for similarity (default: COSINE)
            content_column: Name of content column (default: content)
            embedding_column: Name of embedding column (default: embedding)
            metadata_column: Name of metadata column (default: metadata)
            id_column: Name of ID column (default: id)
            namespace_column: Name of namespace column (default: namespace)
            namespace: Namespace for multi-tenancy isolation. When set, all operations
                are scoped to this namespace. Requires the table to have a namespace
                column (created with namespace_column param in ainit_vectorstore_table).
                Default: None (no namespace filtering, backward compatible).
            hybrid_search_config: Optional hybrid search configuration
            batch_size: Batch size for inserts - CockroachDB works best with smaller batches (default: 100)
            retry_max_attempts: Maximum retry attempts for operations (default: 3)
            retry_initial_backoff: Initial backoff delay in seconds (default: 0.1)
            retry_max_backoff: Maximum backoff delay in seconds (default: 5.0)
            retry_backoff_multiplier: Backoff multiplier (default: 2.0)
            retry_jitter: Add randomization to backoff (default: True)
        """
        self.engine = engine
        self._embeddings = embeddings
        self.collection_name = collection_name
        self.schema = schema
        self.distance_strategy = distance_strategy
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_column = metadata_column
        self.id_column = id_column
        self.namespace_column = namespace_column
        self.namespace = namespace
        self.hybrid_search_config = hybrid_search_config
        self.batch_size = batch_size
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_backoff = retry_initial_backoff
        self.retry_max_backoff = retry_max_backoff
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.retry_jitter = retry_jitter
        self._fqn = f"{schema}.{collection_name}"

    @property
    def embeddings(self) -> Embeddings:
        """Get embeddings model."""
        return self._embeddings

    def _namespace_filter(self) -> str:
        """Return a SQL WHERE fragment for namespace filtering, or empty string."""
        if self.namespace is not None:
            escaped = self.namespace.replace("'", "''")
            return f"{self.namespace_column} = '{escaped}'"
        return ""

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to vector store with automatic retry on failures.

        Args:
            texts: Texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for texts
            **kwargs: Additional arguments (batch_size override supported)

        Returns:
            List of IDs for added texts
        """

        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings = await self._embeddings.aembed_documents(texts_list)

        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]

        batch_size = kwargs.get("batch_size", self.batch_size)

        for i in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            await self._insert_batch(batch_texts, batch_embeddings, batch_metadatas, batch_ids)

        return ids

    async def _insert_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Insert a batch of vectors."""
        import json

        has_ns = self.namespace is not None
        ns_col = f", {self.namespace_column}" if has_ns else ""
        ns_escaped = self.namespace.replace("'", "''") if has_ns and self.namespace else ""

        values = []
        for content, embedding, metadata, doc_id in zip(
            texts, embeddings, metadatas, ids, strict=True
        ):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            escaped_content = content.replace("'", "''")
            metadata_json = json.dumps(metadata).replace("'", "''")
            ns_val = f", '{ns_escaped}'" if has_ns else ""
            values.append(
                f"('{doc_id}', '{escaped_content}', '{embedding_str}', "
                f"'{metadata_json}'::jsonb{ns_val})"
            )

        ns_update = (
            f", {self.namespace_column} = EXCLUDED.{self.namespace_column}" if has_ns else ""
        )
        sql = f"""
            INSERT INTO {self._fqn} ({self.id_column}, {self.content_column}, {self.embedding_column}, {self.metadata_column}{ns_col})
            VALUES {",".join(values)}
            ON CONFLICT ({self.id_column}) DO UPDATE SET
                {self.content_column} = EXCLUDED.{self.content_column},
                {self.embedding_column} = EXCLUDED.{self.embedding_column},
                {self.metadata_column} = EXCLUDED.{self.metadata_column}{ns_update}
        """

        async with self.engine.engine.begin() as conn:
            await conn.execute(text(sql))

    async def _fts_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Perform Full-Text Search and return (id, score) tuples."""
        lang = self.hybrid_search_config.fts_query_language if self.hybrid_search_config else "english"
        escaped_query = query.replace("'", "''")
        tsvector_col = f"{self.content_column}_tsvector"
        
        conditions = []
        ns_filter = self._namespace_filter()
        if ns_filter:
            conditions.append(ns_filter)
        if filter:
            conditions.append(self._build_filter_clause(filter))
            
        conditions.append(f"{tsvector_col} @@ websearch_to_tsquery('{lang}', '{escaped_query}')")
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SELECT {self.id_column}, ts_rank({tsvector_col}, websearch_to_tsquery('{lang}', '{escaped_query}')) AS fts_score
            FROM {self._fqn}
            {where_clause}
            ORDER BY fts_score DESC
            LIMIT {k}
        """
        
        async with self.engine.engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchall()
            
        return [(str(row[0]), float(row[1])) for row in rows]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        query_options: CSPANNQueryOptions | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
            query_options: C-SPANN query options
            **kwargs: Additional arguments

        Returns:
            List of (document, score) tuples
        """
        if self.hybrid_search_config:
            # Hybrid search execution
            query_embedding = await self._embeddings.aembed_query(query)
            fetch_k = kwargs.get("fetch_k", max(k * 4, 60))
            
            import asyncio
            fts_task = asyncio.create_task(self._fts_search(query, k=fetch_k, filter=filter))
            vec_task = asyncio.create_task(
                self.asimilarity_search_with_score_by_vector(
                    query_embedding, k=fetch_k, filter=filter, query_options=query_options, **kwargs
                )
            )
            fts_scores, v_results = await asyncio.gather(fts_task, vec_task)
            
            v_scores = []
            doc_map = {}
            for doc, dist in v_results:
                doc_map[doc.id] = doc
                if self.hybrid_search_config.fusion_type == FusionType.WEIGHTED_SUM:
                    v_scores.append((doc.id, 1.0 - dist))
                else:
                    v_scores.append((doc.id, dist))
                    
            fused_scores = self.hybrid_search_config.fuse_scores(fts_scores, v_scores)
            
            top_ids = [did for did, _ in fused_scores[:k]]
            missing_ids = [did for did in top_ids if did not in doc_map]
            
            if missing_ids:
                missing_docs = await self.aget_by_ids(missing_ids)
                for doc in missing_docs:
                    doc_map[doc.id] = doc
            
            final_results = []
            for did, final_score in fused_scores[:k]:
                if did in doc_map:
                    final_results.append((doc_map[did], final_score))
            return final_results
            
        query_embedding = await self._embeddings.aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            query_embedding, k=k, filter=filter, query_options=query_options, **kwargs
        )

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
        query_options: CSPANNQueryOptions | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents by embedding vector.

        Args:
            embedding: Query embedding vector
            k: Number of results
            filter: Metadata filter
            query_options: C-SPANN query options
            **kwargs: Additional arguments

        Returns:
            List of (document, score) tuples
        """
        operator = self.distance_strategy.get_operator()
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        conditions = []
        ns_filter = self._namespace_filter()
        if ns_filter:
            conditions.append(ns_filter)
        if filter:
            conditions.append(self._build_filter_clause(filter))
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
            SELECT {self.id_column}, {self.content_column}, {self.metadata_column},
                   {self.embedding_column} {operator} '{embedding_str}' AS distance
            FROM {self._fqn}
            {where_clause}
            ORDER BY {self.embedding_column} {operator} '{embedding_str}'
            LIMIT {k}
        """

        async with self.engine.engine.connect() as conn:
            if query_options:
                for setting, value in query_options.get_session_settings().items():
                    await conn.execute(text(f"SET {setting} = {value}"))

            result = await conn.execute(text(sql))
            rows = result.fetchall()

        documents = []
        for row in rows:
            doc = Document(
                id=str(row[0]),
                page_content=row[1],
                metadata=row[2] or {},
            )
            documents.append((doc, float(row[3])))

        return documents

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        results = await self.asimilarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of results to return.
            filter: Metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of documents most similar to the query vector.
        """
        results = await self.asimilarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in results]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Max marginal relevance search.

        Args:
            query: Query text
            k: Number of results
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Metadata filter
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        query_embedding = await self._embeddings.aembed_query(query)
        candidates = await self.asimilarity_search_with_score_by_vector(
            query_embedding, k=fetch_k, filter=filter, **kwargs
        )

        if not candidates:
            return []

        query_vec = np.array(query_embedding)
        candidate_vecs = []
        candidate_docs = []

        for doc, _score in candidates:
            doc_embedding = await self._embeddings.aembed_query(doc.page_content)
            candidate_vecs.append(doc_embedding)
            candidate_docs.append(doc)

        selected_indices = [0]
        selected = [candidate_vecs[0]]

        while len(selected_indices) < min(k, len(candidates)):
            best_score = -float("inf")
            best_idx = -1

            for i, vec in enumerate(candidate_vecs):
                if i in selected_indices:
                    continue

                relevance = np.dot(query_vec, vec)
                max_similarity = max(np.dot(vec, s) for s in selected)
                score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            selected.append(candidate_vecs[best_idx])

        return [candidate_docs[i] for i in selected_indices]

    async def adelete(
        self,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """Delete documents by IDs.

        Args:
            ids: Document IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        if not ids:
            return True

        ids_str = ",".join(f"'{id}'" for id in ids)
        ns_filter = self._namespace_filter()
        ns_clause = f" AND {ns_filter}" if ns_filter else ""
        sql = f"DELETE FROM {self._fqn} WHERE {self.id_column} IN ({ids_str}){ns_clause}"

        async with self.engine.engine.begin() as conn:
            await conn.execute(text(sql))

        return True

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of Document objects found.
        """
        if not ids:
            return []

        ids_str = ",".join(f"'{id}'" for id in ids)
        ns_filter = self._namespace_filter()
        ns_clause = f" AND {ns_filter}" if ns_filter else ""
        sql = f"""
            SELECT {self.id_column}, {self.content_column}, {self.metadata_column}
            FROM {self._fqn}
            WHERE {self.id_column} IN ({ids_str}){ns_clause}
        """

        async with self.engine.engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchall()

        return [
            Document(
                id=str(row[0]),
                page_content=row[1],
                metadata=row[2] or {},
            )
            for row in rows
        ]

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs (sync).

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of Document objects found.
        """
        import asyncio

        return asyncio.run(self.aget_by_ids(ids))

    async def aapply_vector_index(
        self,
        index: CSPANNIndex,
        prefix_columns: list[str] | None = None,
    ) -> None:
        """Apply C-SPANN vector index.

        Args:
            index: Index configuration
            prefix_columns: Optional prefix columns for multi-tenant indexes
        """
        sql = index.get_create_index_sql(
            self.collection_name,
            self.embedding_column,
            schema=self.schema,
            prefix_columns=prefix_columns,
        )

        async with self.engine.engine.begin() as conn:
            await conn.execute(text(sql))

    async def adrop_vector_index(self, index: CSPANNIndex) -> None:
        """Drop vector index.

        Args:
            index: Index configuration
        """
        sql = index.get_drop_index_sql(
            self.collection_name,
            self.embedding_column,
            schema=self.schema,
        )

        async with self.engine.engine.begin() as conn:
            await conn.execute(text(sql))

    def _build_filter_clause(self, filter: dict) -> str:
        """Build WHERE clause from filter dictionary.

        Supports operators: $and, $or, $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
        """
        import json

        if not filter:
            return ""

        if "$and" in filter:
            clauses = [self._build_filter_clause(f) for f in filter["$and"]]
            return "(" + " AND ".join(clauses) + ")"

        if "$or" in filter:
            clauses = [self._build_filter_clause(f) for f in filter["$or"]]
            return "(" + " OR ".join(clauses) + ")"

        clauses = []
        for key, value in filter.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    clause = self._build_operator_clause(key, op, op_value)
                    clauses.append(clause)
            elif isinstance(value, str):
                json_val = json.dumps(value)
                clauses.append(f"{self.metadata_column}->'{key}' = '{json_val}'::jsonb")
            else:
                clauses.append(f"{self.metadata_column}->'{key}' = '{value}'")

        return " AND ".join(clauses) if clauses else "true"

    def _build_operator_clause(self, key: str, operator: str, value: Any) -> str:
        """Build operator clause."""
        import json

        col = f"{self.metadata_column}->'{key}'"

        if operator == "$eq":
            if isinstance(value, str):
                json_val = json.dumps(value)
                return f"{col} = '{json_val}'::jsonb"
            return f"{col} = '{value}'"
        elif operator == "$ne":
            if isinstance(value, str):
                json_val = json.dumps(value)
                return f"{col} != '{json_val}'::jsonb"
            return f"{col} != '{value}'"
        elif operator == "$gt":
            return f"({col})::numeric > {value}"
        elif operator == "$gte":
            return f"({col})::numeric >= {value}"
        elif operator == "$lt":
            return f"({col})::numeric < {value}"
        elif operator == "$lte":
            return f"({col})::numeric <= {value}"
        elif operator == "$in":
            values = ",".join(
                f"'{json.dumps(v)}'::jsonb" if isinstance(v, str) else f"'{v}'" for v in value
            )
            return f"{col} IN ({values})"
        elif operator == "$nin":
            values = ",".join(
                f"'{json.dumps(v)}'::jsonb" if isinstance(v, str) else f"'{v}'" for v in value
            )
            return f"{col} NOT IN ({values})"
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        engine: CockroachDBEngine | None = None,
        connection_string: str | None = None,
        collection_name: str = "langchain_vectors",
        **kwargs: Any,
    ) -> "AsyncCockroachDBVectorStore":
        """Create vector store from texts.

        Args:
            texts: Texts to add
            embedding: Embeddings model
            metadatas: Optional metadata
            engine: CockroachDBEngine instance
            connection_string: Connection string (if engine not provided)
            collection_name: Table name
            **kwargs: Additional arguments

        Returns:
            AsyncCockroachDBVectorStore instance
        """
        if engine is None:
            if connection_string is None:
                raise ValueError("Either engine or connection_string must be provided")
            engine = CockroachDBEngine.from_connection_string(connection_string)

        sample_embedding = await embedding.aembed_query("sample")
        vector_dim = len(sample_embedding)

        await engine.ainit_vectorstore_table(collection_name, vector_dim, **kwargs)

        store = cls(
            engine=engine,
            embeddings=embedding,
            collection_name=collection_name,
            **kwargs,
        )

        await store.aadd_texts(texts, metadatas=metadatas)

        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Sync wrapper - not implemented for async-only store."""
        raise NotImplementedError("Use aadd_texts for async vector store")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Sync wrapper - not implemented for async-only store."""
        raise NotImplementedError("Use asimilarity_search for async vector store")

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> "AsyncCockroachDBVectorStore":
        """Sync wrapper - not implemented for async-only store."""
        raise NotImplementedError("Use afrom_texts for async vector store")
