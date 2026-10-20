"""SQL statement building for the CockroachDB store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_cockroachdb._sql import validate_identifier, vector_literal
from langchain_cockroachdb.store.config import ResolvedIndex
from langchain_cockroachdb.store.filters import translate_filter
from langchain_cockroachdb.store.namespaces import (
    prefix_condition,
    truncate_expression,
)

# Module-level sentinel for the query vector parameter
QUERY_VECTOR = object()

# Type aliases
ItemRow = tuple[str, str, str, float | None]  # prefix, key, value_json, ttl_minutes
VectorRow = tuple[
    str, str, str, list[float], float | None
]  # prefix, key, field_name, embedding, ttl_minutes


def _validate_paging(limit: int, offset: int) -> None:
    """Validate paging parameters.

    Raises:
        ValueError: If limit or offset is negative.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")


class StoreSQL:
    """SQL statement builder for the CockroachDB store.

    Validates identifiers and builds parameterized SQL statements for all store operations.
    """

    def __init__(
        self,
        schema: str,
        table_name: str,
        *,
        index: ResolvedIndex | None = None,
    ):
        """Initialize SQL builder.

        Args:
            schema: Database schema name.
            table_name: Base table name for items.
            index: Resolved index configuration, or None.

        Raises:
            ValueError: If any identifier is invalid.
        """
        self._schema = validate_identifier(schema, what="schema")
        self._table_name = validate_identifier(table_name, what="table name")
        self._index = index

        # Fully qualified table names
        self.items = f"{self._schema}.{self._table_name}"
        self.vectors = f"{self._schema}.{self._table_name}_vectors"
        self.migrations_table = f"{self._schema}.{self._table_name}_migrations"

        # Vector index name (public)
        self.vector_index_name = f"{self._table_name}_vectors_embedding_idx"

    def migrations(self) -> list[str]:
        """Return numbered migration statements for items table.

        Returns:
            List of 2 idempotent DDL statements: (0) migrations table, (1) items table.
        """
        return [
            # Migration 0: migrations tracking table
            f"""CREATE TABLE IF NOT EXISTS {self.migrations_table} (
    v INTEGER PRIMARY KEY
);""",
            # Migration 1: items table
            f"""CREATE TABLE IF NOT EXISTS {self.items} (
    prefix TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    ttl_minutes DECIMAL,
    PRIMARY KEY (prefix, key)
);""",
        ]

    def vector_statements(self) -> list[str]:
        """Return idempotent DDL statements for vectors table and index.

        Returns:
            Empty list when no index configured.
            [vectors table] when flat=True.
            [vectors table, vector index] when index configured with C-SPANN.
        """
        if self._index is None:
            return []

        stmts = [
            # Vectors table
            f"""CREATE TABLE IF NOT EXISTS {self.vectors} (
    prefix TEXT NOT NULL,
    key TEXT NOT NULL,
    field_name TEXT NOT NULL,
    embedding VECTOR({self._index.dims}) NOT NULL,
    expires_at TIMESTAMPTZ,
    ttl_minutes DECIMAL,
    PRIMARY KEY (prefix, key, field_name)
);"""
        ]

        # Add index statement unless flat
        if not self._index.flat:
            opclass = self._index.distance.get_opclass()
            index_sql = (
                f"CREATE VECTOR INDEX IF NOT EXISTS {self.vector_index_name} "
                f"ON {self.vectors} (prefix, embedding {opclass})"
            )

            # Add C-SPANN options if configured
            if self._index.cspann:
                min_size = self._index.cspann["min_partition_size"]
                max_size = self._index.cspann["max_partition_size"]
                index_sql += f" WITH (min_partition_size={min_size}, max_partition_size={max_size})"

            stmts.append(index_sql + ";")

        return stmts

    def ttl_statements(self, cron: str) -> list[str]:
        """Return ALTER TABLE statements to enable TTL.

        Args:
            cron: Cron expression for TTL job.

        Returns:
            List of ALTER TABLE SET statements.

        Raises:
            ValueError: If cron contains single quote.
        """
        if "'" in cron:
            raise ValueError("cron expression cannot contain single quote")

        stmts = [
            f"ALTER TABLE {self.items} SET (ttl_expiration_expression = 'expires_at', ttl_job_cron = '{cron}');"
        ]

        if self._index is not None:
            stmts.append(
                f"ALTER TABLE {self.vectors} SET (ttl_expiration_expression = 'expires_at', ttl_job_cron = '{cron}');"
            )

        return stmts

    def disable_ttl_statements(self) -> list[str]:
        """Return ALTER TABLE statements to disable TTL.

        Returns:
            List of ALTER TABLE RESET statements.
        """
        stmts = [f"ALTER TABLE {self.items} RESET (ttl);"]

        if self._index is not None:
            stmts.append(f"ALTER TABLE {self.vectors} RESET (ttl);")

        return stmts

    def get(
        self, namespace_text: str, keys: Sequence[str], *, omit_expired: bool
    ) -> tuple[str, list]:
        """Build SELECT for items by namespace and keys.

        Args:
            namespace_text: Namespace as dot-separated text.
            keys: Item keys to fetch.
            omit_expired: Whether to exclude expired items.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""SELECT key, value, created_at, updated_at
FROM {self.items}
WHERE prefix = %s AND key = ANY(%s)"""

        params: list[Any] = [namespace_text, list(keys)]

        if omit_expired:
            sql += "\nAND (expires_at IS NULL OR expires_at > now())"

        return (sql, params)

    def refresh(self, namespace_text: str, keys: Sequence[str]) -> tuple[str, list]:
        """Build UPDATE to refresh TTL for items.

        Args:
            namespace_text: Namespace as dot-separated text.
            keys: Item keys to refresh.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""UPDATE {self.items}
SET expires_at = now() + ttl_minutes * INTERVAL '1 minute'
WHERE prefix = %s AND key = ANY(%s) AND ttl_minutes IS NOT NULL"""

        params: list[Any] = [namespace_text, list(keys)]
        return (sql, params)

    def refresh_vectors(self, namespace_text: str, keys: Sequence[str]) -> tuple[str, list]:
        """Build UPDATE to refresh TTL for vectors.

        Args:
            namespace_text: Namespace as dot-separated text.
            keys: Item keys to refresh.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""UPDATE {self.vectors}
SET expires_at = now() + ttl_minutes * INTERVAL '1 minute'
WHERE prefix = %s AND key = ANY(%s) AND ttl_minutes IS NOT NULL"""

        params: list[Any] = [namespace_text, list(keys)]
        return (sql, params)

    def upsert_items(self, rows: Sequence[ItemRow]) -> tuple[str, list]:
        """Build multi-row INSERT for items with ON CONFLICT.

        Args:
            rows: Sequence of (prefix, key, value_json, ttl_minutes) tuples.
                Callers must dedupe (prefix, key) rows (ON CONFLICT cannot touch a row twice).

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If rows is empty.
        """
        if not rows:
            raise ValueError("rows cannot be empty")

        # Use UNION ALL to reference ttl_minutes only once per row, with typed placeholders
        selects = []
        for i in range(len(rows)):
            if i == 0:
                selects.append(
                    "SELECT CAST(%s AS STRING) AS prefix, CAST(%s AS STRING) AS key, "
                    "CAST(%s AS JSONB) AS value, CAST(%s AS DECIMAL) AS ttl_minutes"
                )
            else:
                selects.append(
                    "UNION ALL SELECT CAST(%s AS STRING), CAST(%s AS STRING), "
                    "CAST(%s AS JSONB), CAST(%s AS DECIMAL)"
                )

        union_sql = "\n    ".join(selects)

        sql = f"""INSERT INTO {self.items} (prefix, key, value, expires_at, ttl_minutes)
SELECT prefix, key, value, now() + CAST(ttl_minutes AS DECIMAL) * INTERVAL '1 minute', ttl_minutes
FROM ({union_sql}) AS rows_to_insert
ON CONFLICT (prefix, key) DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = now(),
    expires_at = EXCLUDED.expires_at,
    ttl_minutes = EXCLUDED.ttl_minutes"""

        # Flatten parameters: for each row, emit prefix, key, value, ttl
        params: list[Any] = []
        for prefix, key, value_json, ttl_minutes in rows:
            params.extend([prefix, key, value_json, ttl_minutes])

        return (sql, params)

    def delete_items(self, namespace_text: str, keys: Sequence[str]) -> tuple[str, list]:
        """Build DELETE for items by namespace and keys.

        Args:
            namespace_text: Namespace as dot-separated text.
            keys: Item keys to delete.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""DELETE FROM {self.items}
WHERE prefix = %s AND key = ANY(%s)"""

        params: list[Any] = [namespace_text, list(keys)]
        return (sql, params)

    def delete_vectors(self, namespace_text: str, keys: Sequence[str]) -> tuple[str, list]:
        """Build DELETE for vectors by namespace and keys.

        Args:
            namespace_text: Namespace as dot-separated text.
            keys: Item keys to delete.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""DELETE FROM {self.vectors}
WHERE prefix = %s AND key = ANY(%s)"""

        params: list[Any] = [namespace_text, list(keys)]
        return (sql, params)

    def upsert_vectors(self, rows: Sequence[VectorRow]) -> tuple[str, list]:
        """Build multi-row INSERT for vectors with ON CONFLICT.

        Args:
            rows: Sequence of (prefix, key, field_name, embedding, ttl_minutes) tuples.
                Callers must dedupe (prefix, key, field_name) rows (ON CONFLICT cannot touch a row twice).

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If rows is empty.
        """
        if not rows:
            raise ValueError("rows cannot be empty")

        # Use UNION ALL to reference ttl_minutes only once per row, with typed placeholders
        selects = []
        for i in range(len(rows)):
            if i == 0:
                selects.append(
                    "SELECT CAST(%s AS STRING) AS prefix, CAST(%s AS STRING) AS key, "
                    "CAST(%s AS STRING) AS field_name, CAST(%s AS VECTOR) AS embedding, "
                    "CAST(%s AS DECIMAL) AS ttl_minutes"
                )
            else:
                selects.append(
                    "UNION ALL SELECT CAST(%s AS STRING), CAST(%s AS STRING), "
                    "CAST(%s AS STRING), CAST(%s AS VECTOR), CAST(%s AS DECIMAL)"
                )

        union_sql = "\n    ".join(selects)

        sql = f"""INSERT INTO {self.vectors} (prefix, key, field_name, embedding, expires_at, ttl_minutes)
SELECT prefix, key, field_name, embedding, now() + CAST(ttl_minutes AS DECIMAL) * INTERVAL '1 minute', ttl_minutes
FROM ({union_sql}) AS rows_to_insert
ON CONFLICT (prefix, key, field_name) DO UPDATE SET
    embedding = EXCLUDED.embedding,
    expires_at = EXCLUDED.expires_at,
    ttl_minutes = EXCLUDED.ttl_minutes"""

        # Flatten parameters: for each row, emit prefix, key, field_name, vector literal, ttl
        params: list[Any] = []
        for prefix, key, field_name, embedding, ttl_minutes in rows:
            vec_literal = vector_literal(embedding)
            params.extend([prefix, key, field_name, vec_literal, ttl_minutes])

        return (sql, params)

    def search(
        self,
        *,
        namespace_prefix: Sequence[str],
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
    ) -> tuple[str, list]:
        """Build plain search query ordered by updated_at.

        Args:
            namespace_prefix: Namespace prefix to match.
            filter: Filter dict, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If limit or offset is negative.
        """
        _validate_paging(limit, offset)

        sql = f"SELECT prefix, key, value, created_at, updated_at\nFROM {self.items}\nWHERE "
        params: list[Any] = []

        # Namespace condition
        ns_condition, ns_params = prefix_condition("prefix", namespace_prefix)
        sql += ns_condition
        params.extend(ns_params)

        # Filter conditions
        filter_conditions, filter_params = translate_filter(filter)
        if filter_conditions:
            sql += " AND " + " AND ".join(filter_conditions)
            params.extend(filter_params)

        # Expiry clause
        if omit_expired:
            sql += " AND (expires_at IS NULL OR expires_at > now())"

        # Order and paging
        sql += "\nORDER BY updated_at DESC, prefix, key\nLIMIT %s OFFSET %s"
        params.extend([limit, offset])

        return (sql, params)

    def semantic_search(
        self,
        *,
        namespace_prefix: Sequence[str],
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        over_fetch: int,
    ) -> tuple[str, list]:
        """Build semantic search query with vector similarity.

        Args:
            namespace_prefix: Namespace prefix to match.
            filter: Filter dict, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.
            over_fetch: Limit for the vector CTE (top-k from index).

        Returns:
            Tuple of (SQL, parameters). Query vector appears as QUERY_VECTOR sentinel twice.

        Raises:
            ValueError: If limit or offset is negative.
        """
        _validate_paging(limit, offset)

        if self._index is None:
            raise ValueError("semantic_search requires an index configuration")

        from langchain_cockroachdb.indexes import DistanceStrategy

        operator = self._index.distance.get_operator()

        # Build SQL text
        sql = f"""WITH ranked AS (
    SELECT prefix, key, field_name, embedding {operator} CAST(%s AS VECTOR) AS distance
    FROM {self.vectors}
    WHERE """

        # Build params in order of placeholders in SQL text
        params: list[Any] = [QUERY_VECTOR]  # First placeholder

        # Namespace condition on vectors table
        ns_condition, ns_params = prefix_condition("prefix", namespace_prefix)
        sql += ns_condition

        # Filter expired vectors in CTE
        if omit_expired:
            sql += " AND (expires_at IS NULL OR expires_at > now())"

        sql += f"""
    ORDER BY embedding {operator} CAST(%s AS VECTOR)
    LIMIT %s
),
best AS (
    SELECT DISTINCT ON (i.prefix, i.key)
        i.prefix, i.key, i.value, i.created_at, i.updated_at,"""

        # Score calculation
        if self._index.distance == DistanceStrategy.COSINE:
            sql += "\n        1 - r.distance AS score"
        else:
            sql += "\n        -r.distance AS score"

        sql += f"""
    FROM ranked r
    JOIN {self.items} i ON r.prefix = i.prefix AND r.key = i.key
    WHERE """

        # Add namespace params after first QUERY_VECTOR
        params.extend(ns_params)

        # Add second QUERY_VECTOR and over_fetch
        params.append(QUERY_VECTOR)
        params.append(over_fetch)

        # Filter conditions on items
        filter_conditions, filter_params = translate_filter(filter, value_column="i.value")
        if filter_conditions:
            sql += " AND ".join(filter_conditions)
            sql += " AND "

        # Expiry clause on items table
        if omit_expired:
            sql += "(i.expires_at IS NULL OR i.expires_at > now())"
        else:
            sql += "TRUE"

        sql += """
    ORDER BY i.prefix, i.key, r.distance
)
SELECT prefix, key, value, created_at, updated_at, score
FROM best
ORDER BY score DESC, prefix, key
LIMIT %s OFFSET %s"""

        # Add filter params, then limit and offset
        params.extend(filter_params)
        params.extend([limit, offset])

        return (sql, params)

    def list_namespaces(
        self,
        *,
        match_conditions: Sequence[str],
        max_depth: int | None,
        limit: int,
        offset: int,
        omit_expired: bool,
    ) -> tuple[str, list]:
        """Build query to list distinct namespaces.

        Args:
            match_conditions: Regex pattern strings (from match_pattern).
            max_depth: Maximum namespace depth for truncation, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If limit or offset is negative.
        """
        _validate_paging(limit, offset)

        # Build truncation expression or use prefix directly
        ns_expr = truncate_expression("prefix", max_depth) if max_depth is not None else "prefix"

        sql = f"SELECT DISTINCT {ns_expr} AS ns\nFROM {self.items}\nWHERE "
        params: list[Any] = []

        # Expiry clause
        if omit_expired:
            sql += "(expires_at IS NULL OR expires_at > now())"
        else:
            sql += "TRUE"

        # Match conditions (regex patterns)
        for pattern in match_conditions:
            sql += " AND prefix ~ %s"
            params.append(pattern)

        sql += "\nORDER BY ns\nLIMIT %s OFFSET %s"
        params.extend([limit, offset])

        return (sql, params)

    def delete_namespace(self, namespace_prefix: Sequence[str]) -> list[tuple[str, list]]:
        """Build DELETE statements for a namespace and its descendants.

        Args:
            namespace_prefix: Namespace prefix to delete.

        Returns:
            List of (SQL, parameters) tuples. Vectors delete (if index configured),
            then items delete with RETURNING key.
        """
        deletes: list[tuple[str, list]] = []

        # Build prefix condition
        ns_condition, ns_params = prefix_condition("prefix", namespace_prefix)

        # Vectors delete (if index configured)
        if self._index is not None:
            vectors_sql = f"DELETE FROM {self.vectors}\nWHERE {ns_condition}"
            deletes.append((vectors_sql, ns_params.copy()))

        # Items delete with RETURNING
        items_sql = f"DELETE FROM {self.items}\nWHERE {ns_condition}\nRETURNING key"
        deletes.append((items_sql, ns_params.copy()))

        return deletes

    def sweep(self) -> list[tuple[str, list]]:
        """Build DELETE statements for expired rows.

        Returns:
            List of (SQL, parameters) tuples for vectors (if index), then items.
        """
        sweeps: list[tuple[str, list]] = []

        # Vectors sweep (if index configured)
        if self._index is not None:
            vectors_sql = (
                f"DELETE FROM {self.vectors}\nWHERE expires_at IS NOT NULL AND expires_at < now()"
            )
            sweeps.append((vectors_sql, []))

        # Items sweep
        items_sql = f"DELETE FROM {self.items}\nWHERE expires_at IS NOT NULL AND expires_at < now()"
        sweeps.append((items_sql, []))

        return sweeps
