"""Async vector store implementation for CockroachDB with transaction retry support."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from sqlalchemy import bindparam, text
from sqlalchemy.sql.elements import TextClause

from langchain_cockroachdb._filters import FilterBuilder
from langchain_cockroachdb._sql import parse_vector, validate_identifier, vector_literal
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.hybrid_search_config import (
    FusionType,
    HybridSearchConfig,
    min_max_normalize,
)
from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions, DistanceStrategy
from langchain_cockroachdb.retry import RetryPolicy, resolve_retry_policy

logger = logging.getLogger(__name__)

# Arguments of ainit_vectorstore_table. The schema and column ones also
# configure the store, so from_texts passes those to both.
TABLE_ARGS = frozenset(
    {
        "schema",
        "id_type",
        "content_column",
        "embedding_column",
        "metadata_column",
        "namespace_column",
        "create_tsvector",
        "fts_language",
        "drop_if_exists",
    }
)
SHARED_ARGS = frozenset(
    {"schema", "content_column", "embedding_column", "metadata_column", "namespace_column"}
)


def split_from_texts_args(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split from_texts keyword arguments into table arguments and store arguments."""
    table_args = {k: v for k, v in kwargs.items() if k in TABLE_ARGS}
    store_args = {k: v for k, v in kwargs.items() if k not in TABLE_ARGS or k in SHARED_ARGS}
    return table_args, store_args


@dataclass
class _Match:
    """One search hit before it becomes a public result."""

    document: Document
    distance: float
    embedding: list[float] | None = None


@dataclass
class _Rows:
    """Validated input for an insert, one entry per document."""

    texts: list[str]
    embeddings: list[list[float]]
    metadata_json: list[str]
    ids: list[str]
    batch_size: int


def _check_batch_size(batch_size: Any) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return batch_size


class AsyncCockroachDBVectorStore(VectorStore):
    """Async vector store using CockroachDB native VECTOR type and C-SPANN indexes.

    Every database call retries transient CockroachDB errors under
    ``retry_policy``, which defaults to the engine's policy. Every write is
    an idempotent UPSERT or DELETE, so retrying after an ambiguous result is
    safe.
    """

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
        retry_policy: RetryPolicy | None = None,
        retry_max_attempts: int | None = None,
        retry_initial_backoff: float | None = None,
        retry_max_backoff: float | None = None,
        retry_backoff_multiplier: float | None = None,
        retry_jitter: bool | None = None,
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
            batch_size: Rows per insert statement; CockroachDB works best with
                smaller VECTOR batches (default: 100)
            retry_policy: How operations retry (default: the engine's policy)
            retry_max_attempts: Override for retry_policy.max_attempts
            retry_initial_backoff: Override for retry_policy.initial_backoff
            retry_max_backoff: Override for retry_policy.max_backoff
            retry_backoff_multiplier: Override for retry_policy.backoff_multiplier
            retry_jitter: Override for retry_policy.jitter (True means 0.5)
        """
        validate_identifier(collection_name, what="collection name")
        validate_identifier(schema, what="schema")
        for column in (
            content_column,
            embedding_column,
            metadata_column,
            id_column,
            namespace_column,
        ):
            validate_identifier(column, what="column name")
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
        self.batch_size = _check_batch_size(batch_size)
        self.retry_policy = resolve_retry_policy(
            retry_policy,
            base=engine.retry_policy,
            max_attempts=retry_max_attempts,
            initial_backoff=retry_initial_backoff,
            max_backoff=retry_max_backoff,
            backoff_multiplier=retry_backoff_multiplier,
            jitter=retry_jitter,
        )
        self._fqn = f"{schema}.{collection_name}"

    @property
    def embeddings(self) -> Embeddings:
        """Get embeddings model."""
        return self._embeddings

    @property
    def retry_max_attempts(self) -> int:
        """Shorthand for ``retry_policy.max_attempts``."""
        return self.retry_policy.max_attempts

    @property
    def retry_initial_backoff(self) -> float:
        """Shorthand for ``retry_policy.initial_backoff``."""
        return self.retry_policy.initial_backoff

    @property
    def retry_max_backoff(self) -> float:
        """Shorthand for ``retry_policy.max_backoff``."""
        return self.retry_policy.max_backoff

    @property
    def retry_backoff_multiplier(self) -> float:
        """Shorthand for ``retry_policy.backoff_multiplier``."""
        return self.retry_policy.backoff_multiplier

    @property
    def retry_jitter(self) -> bool:
        """True when ``retry_policy.jitter`` is above zero."""
        return self.retry_policy.jitter > 0

    # Database work. These coroutines run on whichever loop the engine picks:
    # the caller's loop for async methods, the background loop for sync ones.

    async def _execute(
        self,
        statement: TextClause,
        params: dict[str, Any] | None = None,
        *,
        label: str,
        begin: bool = False,
        ddl: bool = False,
        settings: dict[str, int] | None = None,
        fetch: bool = False,
    ) -> list[Any]:
        """Run one statement under the retry policy and the hang guard."""
        timeout = (
            self.retry_policy.ddl_attempt_timeout if ddl else self.retry_policy.attempt_timeout
        )

        async def attempt() -> list[Any]:
            async with self.engine._connect(
                begin=begin or bool(settings), timeout=timeout, label=label
            ) as conn:
                # DDL runs in AUTOCOMMIT mode (implicit transaction) so it commits immediately
                if ddl:
                    await conn.execution_options(isolation_level="AUTOCOMMIT")
                # SET LOCAL ends with the transaction, so nothing leaks into the pool
                for name, value in (settings or {}).items():
                    validate_identifier(name, what="session setting")
                    await conn.execute(text(f"SET LOCAL {name} = {int(value)}"))
                result = await conn.execute(statement, params or {})
                return list(result.fetchall()) if fetch else []

        return await self.retry_policy.arun(attempt, label=label)

    def _where(self, builder: FilterBuilder, filter: dict | None, *conditions: str) -> str:
        clauses = list(conditions)
        if self.namespace is not None:
            clauses.append(f"{self.namespace_column} = {builder.bind(self.namespace)}")
        if filter:
            clauses.append(builder.build(filter))
        return "WHERE " + " AND ".join(clauses) if clauses else ""

    def _prepare_rows(
        self,
        texts: list[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: list[dict] | None,
        ids: Sequence[str | None] | None,
        kwargs: dict[str, Any],
    ) -> _Rows:
        """Validate and serialize everything up front, so bad input writes nothing."""
        count = len(texts)
        if len(embeddings) != count:
            raise ValueError(f"Got {len(embeddings)} embeddings for {count} texts")
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != count:
            raise ValueError(f"Got {len(metadatas)} metadatas for {count} texts")
        if ids is None:
            resolved_ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != count:
            raise ValueError(f"Got {len(ids)} ids for {count} texts")
        else:
            resolved_ids = [str(i) if i is not None else str(uuid.uuid4()) for i in ids]
        seen = set()
        for doc_id in resolved_ids:
            if doc_id in seen:
                raise ValueError(f"Duplicate id in one add call: {doc_id}")
            seen.add(doc_id)
        metadata_json = []
        for position, metadata in enumerate(metadatas):
            try:
                metadata_json.append(
                    json.dumps(metadata if metadata is not None else {}, allow_nan=False)
                )
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Metadata for text {position} is not JSON serializable: {error}"
                ) from error
        batch_size = _check_batch_size(kwargs.get("batch_size", self.batch_size))
        return _Rows(
            texts,
            [list(map(float, e)) for e in embeddings],
            metadata_json,
            resolved_ids,
            batch_size,
        )

    def _normalize_id(self, id_str: str) -> str:
        """Normalize id to canonical form for comparison (lowercase UUID if parseable)."""
        try:
            return str(uuid.UUID(id_str)).lower()
        except (ValueError, AttributeError):
            return id_str

    async def _add_embeddings(self, rows: _Rows) -> list[str]:
        if self.namespace is not None and rows.ids:
            for start in range(0, len(rows.ids), rows.batch_size):
                end = start + rows.batch_size
                batch_ids = rows.ids[start:end]
                statement = text(
                    f"SELECT {self.id_column} FROM {self._fqn} "
                    f"WHERE {self.id_column} IN :ids AND {self.namespace_column} <> :namespace"
                ).bindparams(bindparam("ids", expanding=True))
                conflicting = await self._execute(
                    statement,
                    {"ids": [str(i) for i in batch_ids], "namespace": self.namespace},
                    label=f"check namespace conflicts in {self._fqn}",
                    fetch=True,
                )
                if conflicting:
                    conflicting_ids = [str(row[0]) for row in conflicting]
                    sample = conflicting_ids[:5]
                    suffix = "..." if len(conflicting_ids) > 5 else ""
                    raise ValueError(
                        f"{len(conflicting_ids)} ids belong to another namespace in {self._fqn}. "
                        f"Ids must be unique across namespaces: {', '.join(sample)}{suffix}"
                    )

        for start in range(0, len(rows.texts), rows.batch_size):
            end = start + rows.batch_size
            await self._insert_batch(
                rows.texts[start:end],
                rows.embeddings[start:end],
                rows.metadata_json[start:end],
                rows.ids[start:end],
            )
        return rows.ids

    async def _insert_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata_json: list[str],
        ids: list[str],
    ) -> None:
        """Upsert one batch; each batch is its own retried transaction."""
        columns = [self.id_column, self.content_column, self.embedding_column, self.metadata_column]
        if self.namespace is not None:
            columns.append(self.namespace_column)
        params: dict[str, Any] = {}
        rows = []
        for n, (doc_id, content, embedding, metadata) in enumerate(
            zip(ids, texts, embeddings, metadata_json, strict=True)
        ):
            params.update(
                {
                    f"id{n}": doc_id,
                    f"content{n}": content,
                    f"embedding{n}": vector_literal(embedding),
                    f"metadata{n}": metadata,
                }
            )
            values = [
                f":id{n}",
                f":content{n}",
                f"CAST(:embedding{n} AS VECTOR)",
                f"CAST(:metadata{n} AS JSONB)",
            ]
            if self.namespace is not None:
                values.append(":namespace")
            rows.append("(" + ", ".join(values) + ")")
        if self.namespace is not None:
            params["namespace"] = self.namespace
        updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in columns[1:])

        if self.namespace is not None:
            sql = (
                f"INSERT INTO {self._fqn} AS existing ({', '.join(columns)}) VALUES {', '.join(rows)} "
                f"ON CONFLICT ({self.id_column}) DO UPDATE SET {updates} "
                f"WHERE existing.{self.namespace_column} = EXCLUDED.{self.namespace_column} "
                f"RETURNING {self.id_column}"
            )
            result = await self._execute(
                text(sql), params, label=f"insert into {self._fqn}", begin=True, fetch=True
            )
            returned_ids = {self._normalize_id(str(row[0])) for row in result}
            expected_ids = {self._normalize_id(str(id_str)) for id_str in ids}
            if returned_ids != expected_ids:
                skipped = expected_ids - returned_ids
                skipped_list = sorted(skipped)[:5]
                suffix = "..." if len(skipped) > 5 else ""
                raise ValueError(
                    f"{len(skipped)} ids belong to another namespace: {', '.join(skipped_list)}{suffix}"
                )
        else:
            sql = (
                f"INSERT INTO {self._fqn} ({', '.join(columns)}) VALUES {', '.join(rows)} "
                f"ON CONFLICT ({self.id_column}) DO UPDATE SET {updates}"
            )
            await self._execute(text(sql), params, label=f"insert into {self._fqn}", begin=True)

    async def _search_by_vector(
        self,
        embedding: Sequence[float],
        k: int,
        filter: dict | None,
        query_options: CSPANNQueryOptions | None,
        *,
        with_embeddings: bool = False,
    ) -> list[_Match]:
        if int(k) <= 0:
            return []
        builder = FilterBuilder(self.metadata_column)
        where = self._where(builder, filter)
        distance = f"{self.embedding_column} {self.distance_strategy.get_operator()} CAST(:query_vector AS VECTOR)"
        extra = f", {self.embedding_column}" if with_embeddings else ""
        sql = (
            f"SELECT {self.id_column}, {self.content_column}, {self.metadata_column}, "
            f"{distance} AS distance{extra} FROM {self._fqn} {where} "
            f"ORDER BY {distance} LIMIT {int(k)}"
        )
        params = dict(builder.params, query_vector=vector_literal(embedding))
        settings = query_options.get_session_settings() if query_options else None
        rows = await self._execute(
            text(sql), params, label=f"search {self._fqn}", settings=settings or None, fetch=True
        )
        return [
            _Match(
                Document(id=str(row[0]), page_content=row[1], metadata=row[2] or {}),
                float(row[3]),
                parse_vector(row[4]) if with_embeddings else None,
            )
            for row in rows
        ]

    async def _fts_search(self, query: str, k: int, filter: dict | None) -> list[tuple[str, float]]:
        """Full-text search returning (id, ts_rank score), best first.

        Uses plainto_tsquery because CockroachDB has no websearch_to_tsquery.
        """
        config = self.hybrid_search_config
        language = config.fts_query_language if config else "english"
        normalization = config.fts_rank_normalization if config else 0
        tsvector = f"{self.content_column}_tsvector"
        tsquery = f"plainto_tsquery('{language}', :fts_query)"
        builder = FilterBuilder(self.metadata_column)
        where = self._where(builder, filter, f"{tsvector} @@ {tsquery}")
        sql = (
            f"SELECT {self.id_column}, ts_rank({tsvector}, {tsquery}, {int(normalization)}) AS fts_score "
            f"FROM {self._fqn} {where} ORDER BY fts_score DESC LIMIT {int(k)}"
        )
        rows = await self._execute(
            text(sql),
            dict(builder.params, fts_query=query),
            label=f"full-text search {self._fqn}",
            fetch=True,
        )
        return [(str(row[0]), float(row[1])) for row in rows]

    async def _hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        k: int,
        filter: dict | None,
        query_options: CSPANNQueryOptions | None,
        fetch_k: int | None,
    ) -> list[tuple[Document, float]]:
        """Run the vector and full-text legs together and fuse their rankings."""
        config = self.hybrid_search_config
        assert config is not None
        pool = fetch_k if fetch_k is not None else max(k * 4, 20)
        matches, fts_scores = await asyncio.gather(
            self._search_by_vector(embedding, pool, filter, query_options),
            self._fts_search(query, pool, filter),
        )
        documents = {str(m.document.id): m.document for m in matches}
        # Distances sort ascending for every strategy, so negating them gives a
        # "higher is better" score without caring which operator ran
        vector_scores = [(str(m.document.id), -m.distance) for m in matches]
        if config.fusion_type == FusionType.WEIGHTED_SUM:
            vector_scores = min_max_normalize(vector_scores)
            fts_scores = min_max_normalize(fts_scores)
        fused = config.fuse_scores(fts_scores, vector_scores)[:k]
        missing = [doc_id for doc_id, _ in fused if doc_id not in documents]
        if missing:
            for doc in await self._get_by_ids(missing):
                documents[str(doc.id)] = doc
        return [(documents[doc_id], score) for doc_id, score in fused if doc_id in documents]

    async def _get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        if not ids:
            return []
        builder = FilterBuilder(self.metadata_column)
        where = self._where(builder, None, f"{self.id_column} IN :ids")
        statement = text(
            f"SELECT {self.id_column}, {self.content_column}, {self.metadata_column} FROM {self._fqn} {where}"
        ).bindparams(bindparam("ids", expanding=True))
        rows = await self._execute(
            statement,
            dict(builder.params, ids=[str(i) for i in ids]),
            label=f"get from {self._fqn}",
            fetch=True,
        )
        return [
            Document(id=str(row[0]), page_content=row[1], metadata=row[2] or {}) for row in rows
        ]

    async def _delete(self, ids: Sequence[str]) -> None:
        builder = FilterBuilder(self.metadata_column)
        where = self._where(builder, None, f"{self.id_column} IN :ids")
        statement = text(f"DELETE FROM {self._fqn} {where}").bindparams(
            bindparam("ids", expanding=True)
        )
        await self._execute(
            statement,
            dict(builder.params, ids=[str(i) for i in ids]),
            label=f"delete from {self._fqn}",
            begin=True,
        )

    async def _apply_vector_index(
        self, index: CSPANNIndex, prefix_columns: list[str] | None
    ) -> None:
        for column in prefix_columns or []:
            validate_identifier(column, what="prefix column")
        name = index.get_index_name(self.collection_name, self.embedding_column)
        existing = await self._execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = :schema AND tablename = :table AND indexname = :name"
            ),
            {
                "schema": self.schema.lower(),
                "table": self.collection_name.lower(),
                "name": name.lower(),
            },
            label=f"look up vector index {name}",
            fetch=True,
        )
        opclass = index.distance_strategy.get_opclass()
        if existing:
            if opclass not in existing[0][0]:
                logger.warning(
                    "Vector index %s already exists without %s, so %s searches will not use it. "
                    "Drop it with adrop_vector_index() and apply it again. Indexes created before "
                    "langchain-cockroachdb 0.4.0 always used vector_l2_ops.",
                    name,
                    opclass,
                    index.distance_strategy.name,
                )
            return
        sql = index.get_create_index_sql(
            self.collection_name,
            self.embedding_column,
            schema=self.schema,
            prefix_columns=prefix_columns,
        )
        await self._execute(text(sql), label=f"create vector index {name}", ddl=True)

    async def _drop_vector_index(self, index: CSPANNIndex) -> None:
        sql = index.get_drop_index_sql(
            self.collection_name, self.embedding_column, schema=self.schema
        )
        await self._execute(
            text(sql),
            label=f"drop vector index {index.get_index_name(self.collection_name, self.embedding_column)}",
            ddl=True,
        )

    def _select_mmr(
        self, embedding: Sequence[float], matches: list[_Match], k: int, lambda_mult: float
    ) -> list[Document]:
        """Pick k diverse documents from candidates using their stored embeddings."""
        if not matches:
            return []
        # MMR requires embeddings; they are always present when with_embeddings=True
        embeddings = []
        for m in matches:
            if m.embedding is None:
                raise ValueError(
                    "MMR requires embeddings but got None; ensure with_embeddings=True"
                )
            embeddings.append(m.embedding)
        chosen = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            lambda_mult=lambda_mult,
            k=min(k, len(matches)),
        )
        return [matches[i].document for i in chosen]

    @staticmethod
    def _engine_for(
        engine: CockroachDBEngine | None, connection_string: str | None
    ) -> CockroachDBEngine:
        if engine is not None:
            return engine
        if connection_string is None:
            raise ValueError("Either engine or connection_string must be provided")
        return CockroachDBEngine.from_connection_string(connection_string)

    # Async API

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and add texts, retrying transient errors per batch.

        Args:
            texts: Texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for texts
            **kwargs: Additional arguments (batch_size override supported)

        Returns:
            List of IDs for added texts

        Raises:
            ValueError: When using a namespace, if any id already belongs to a different
                namespace. The guarded upsert also raises if a concurrent writer claims
                an id in another namespace between the check and the write. Rows in the
                same batch that were in this namespace, and earlier batches, are already
                committed when this error is raised.
        """
        texts_list = list(texts)
        if not texts_list:
            return []
        embeddings = await self._embeddings.aembed_documents(texts_list)
        return await self.aadd_embeddings(
            texts_list, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts with precomputed embeddings.

        Args:
            texts: Texts to add
            embeddings: One embedding per text
            metadatas: Optional metadata for each text
            ids: Optional IDs for texts
            **kwargs: Additional arguments (batch_size override supported)

        Returns:
            List of IDs for added texts

        Raises:
            ValueError: When using a namespace, if any id already belongs to a different
                namespace. The guarded upsert also raises if a concurrent writer claims
                an id in another namespace between the check and the write. Rows in the
                same batch that were in this namespace, and earlier batches, are already
                committed when this error is raised.
        """
        rows = self._prepare_rows(list(texts), embeddings, metadatas, ids, kwargs)
        return await self.engine._run_as_async(self._add_embeddings(rows))

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        query_options: CSPANNQueryOptions | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with scores.

        When a hybrid_search_config is set, this runs full-text and vector
        searches in parallel and returns fused scores. Otherwise scores are
        plain vector distances.

        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
            query_options: C-SPANN query options
            **kwargs: Additional arguments (fetch_k sets the hybrid candidate
                pool size, default max(k * 4, 20))

        Returns:
            List of (document, score) tuples
        """
        embedding = await self._embeddings.aembed_query(query)
        if self.hybrid_search_config is not None:
            return await self.engine._run_as_async(
                self._hybrid_search(
                    query, embedding, k, filter, query_options, kwargs.get("fetch_k")
                )
            )
        return await self.asimilarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, query_options=query_options
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
            List of (document, distance) tuples
        """
        matches = await self.engine._run_as_async(
            self._search_by_vector(embedding, k, filter, query_options)
        )
        return [(match.document, match.distance) for match in matches]

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        """Search for similar documents."""
        results = await self.asimilarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: dict | None = None, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to embedding vector."""
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
        """Max marginal relevance search using the stored embeddings of the candidates.

        Args:
            query: Query text
            k: Number of results
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Metadata filter
            **kwargs: Additional arguments (query_options supported)

        Returns:
            List of documents
        """
        embedding = await self._embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Max marginal relevance search from a query embedding."""
        matches = await self.engine._run_as_async(
            self._search_by_vector(
                embedding, fetch_k, filter, kwargs.get("query_options"), with_embeddings=True
            )
        )
        return self._select_mmr(embedding, matches, k, lambda_mult)

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by IDs.

        Args:
            ids: Document IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        if not ids:
            return True
        await self.engine._run_as_async(self._delete(ids))
        return True

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs (order not guaranteed)."""
        return await self.engine._run_as_async(self._get_by_ids(ids))

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs (sync)."""
        return self.engine._run_as_sync(self._get_by_ids(ids))

    async def aapply_vector_index(
        self, index: CSPANNIndex, prefix_columns: list[str] | None = None
    ) -> None:
        """Apply a C-SPANN vector index.

        Warns instead of rebuilding when an index with the same name exists
        with a different operator class.

        Args:
            index: Index configuration
            prefix_columns: Optional prefix columns for multi-tenant indexes
        """
        await self.engine._run_as_async(self._apply_vector_index(index, prefix_columns))

    async def adrop_vector_index(self, index: CSPANNIndex) -> None:
        """Drop a vector index.

        Args:
            index: Index configuration
        """
        await self.engine._run_as_async(self._drop_vector_index(index))

    @classmethod
    async def afrom_texts(
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
    ) -> AsyncCockroachDBVectorStore:
        """Create the table if needed, then add texts.

        Table arguments (id_type, create_tsvector, fts_language, drop_if_exists
        and the schema and column names) go to ainit_vectorstore_table; the
        rest configure the store.

        Args:
            texts: Texts to add
            embedding: Embeddings model
            metadatas: Optional metadata
            ids: Optional IDs
            engine: CockroachDBEngine instance
            connection_string: Connection string (if engine not provided)
            collection_name: Table name
            **kwargs: Table and store arguments

        Returns:
            AsyncCockroachDBVectorStore instance
        """
        resolved = cls._engine_for(engine, connection_string)
        texts = list(texts)
        vectors = await embedding.aembed_documents(texts) if texts else []
        dimension = (
            len(vectors[0]) if vectors else len(await embedding.aembed_query("dimension probe"))
        )
        table_args, store_args = split_from_texts_args(kwargs)
        await resolved.ainit_vectorstore_table(collection_name, dimension, **table_args)
        store = cls(
            engine=resolved, embeddings=embedding, collection_name=collection_name, **store_args
        )
        if texts:
            await store.aadd_embeddings(texts, vectors, metadatas=metadatas, ids=ids)
        return store

    def add_texts(
        self, texts: Iterable[str], metadatas: list[dict] | None = None, **kwargs: Any
    ) -> list[str]:
        """Sync wrapper - not implemented for async-only store."""
        raise NotImplementedError("Use aadd_texts, or CockroachDBVectorStore for sync methods")

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Sync wrapper - not implemented for async-only store."""
        raise NotImplementedError(
            "Use asimilarity_search, or CockroachDBVectorStore for sync methods"
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> AsyncCockroachDBVectorStore:
        """Sync wrapper - not implemented for async-only store."""
        raise NotImplementedError("Use afrom_texts, or CockroachDBVectorStore for sync methods")
