"""SQL statement building for the CockroachDB store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_cockroachdb._sql import validate_identifier, vector_literal
from langchain_cockroachdb.indexes import DistanceStrategy
from langchain_cockroachdb.store.config import ResolvedIndex
from langchain_cockroachdb.store.filters import translate_filter
from langchain_cockroachdb.store.namespaces import (
    prefix_condition,
    truncate_expression,
)

# Stands in the params for the query vector until the query is embedded
QUERY_VECTOR = object()
# Stands in the params for the namespaces found by namespace_lookup
NAMESPACES = object()

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


def bind(statement: tuple[str, list], placeholder: object, value: Any) -> tuple[str, list]:
    """A copy of ``statement`` with ``value`` wherever ``placeholder`` is in its params."""
    sql, params = statement
    return (sql, [value if param is placeholder else param for param in params])


def _prefix_range(column: str, namespace_prefix: Sequence[str]) -> tuple[str, list[str]]:
    """Match a namespace and its descendants with a range the primary key can scan.

    The strings that start with "a.b." are exactly those from "a.b." up to,
    not including, "a.b/", because "/" is the byte after "." and STRING
    compares bytes. The range constrains the primary key scan on every
    supported version, while v25.4 scans the whole table for
    ``LIKE ... ESCAPE``. Labels need no escaping.
    """
    text = ".".join(namespace_prefix)
    return (
        f"({column} = %s OR ({column} >= %s AND {column} < %s))",
        [text, f"{text}.", f"{text}/"],
    )


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

    def show_create_vectors(self) -> str:
        """The statement that reads the vectors table's definition.

        setup() checks the existing vector index and VECTOR column in it.
        SHOW CREATE TABLE renders the operator class and partition sizes on
        every supported version.
        """
        return f"SELECT create_statement FROM [SHOW CREATE TABLE {self.vectors}]"

    def drop_vector_index(self) -> str:
        """The statement that drops the vector index, if it exists."""
        return f"DROP INDEX IF EXISTS {self.vectors}@{self.vector_index_name}"

    @staticmethod
    def beam_size_statement(beam_size: int) -> str:
        """The statement that sets the C-SPANN beam size for one transaction.

        psycopg cannot bind a parameter in SET, so the value is formatted
        into the text. It is a positive int checked here, never user text.

        Raises:
            ValueError: If beam_size is not a positive int.
        """
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
            raise ValueError(f"beam_size must be a positive integer, got {beam_size!r}")
        return f"SET LOCAL vector_search_beam_size = {beam_size}"

    @staticmethod
    def bind_query_vector(
        statement: tuple[str, list], embedding: Sequence[float]
    ) -> tuple[str, list]:
        """A copy of a semantic search statement with its query vector bound."""
        return bind(statement, QUERY_VECTOR, vector_literal(embedding))

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

    def _refresh_cte(self, name: str, table: str, where: str) -> str:
        """A CTE that pushes back the expiry of the rows of ``table`` matching ``where``.

        Only rows with a TTL change. Data-modifying CTEs run even when the
        statement does not read them, and the statement's SELECT reads the
        rows as they were before the update.
        """
        return (
            f"{name} AS (\n    UPDATE {table}\n"
            "    SET expires_at = now() + ttl_minutes * INTERVAL '1 minute'\n"
            f"    WHERE ttl_minutes IS NOT NULL AND {where}\n    RETURNING key\n)"
        )

    def get(
        self,
        namespace_text: str,
        keys: Sequence[str],
        *,
        omit_expired: bool,
        refresh: bool = False,
    ) -> tuple[str, list]:
        """Build SELECT for items by namespace and keys.

        Args:
            namespace_text: Namespace as dot-separated text.
            keys: Item keys to fetch.
            omit_expired: Whether to exclude expired items.
            refresh: Whether to also push back the expiry of the rows read,
                in the same statement (vectors too, with an index).

        Returns:
            Tuple of (SQL, parameters).
        """
        expiry = " AND (expires_at IS NULL OR expires_at > now())" if omit_expired else ""
        where = f"prefix = %s AND key = ANY(%s){expiry}"
        sql = f"""SELECT key, value, created_at, updated_at
FROM {self.items}
WHERE {where}"""
        params: list[Any] = [namespace_text, list(keys)]
        if not refresh:
            return (sql, params)

        ctes = [self._refresh_cte("refreshed", self.items, where)]
        cte_params: list[Any] = [namespace_text, list(keys)]
        if self._index is not None:
            ctes.append(self._refresh_cte("refreshed_vectors", self.vectors, where))
            cte_params.extend([namespace_text, list(keys)])
        return ("WITH " + ",\n".join(ctes) + "\n" + sql, cte_params + params)

    def _with_refresh(self, query: str, columns: str, order_by: str) -> str:
        """Wrap a search query so the rows it returns get their expiry pushed back.

        The query becomes the CTE ``search_results``. Its parameters stay
        first and in order, since the added CTEs take none.
        """
        where = "(prefix, key) IN (SELECT prefix, key FROM search_results)"
        ctes = [
            f"search_results AS (\n{query}\n)",
            self._refresh_cte("updated", self.items, where),
        ]
        if self._index is not None:
            ctes.append(self._refresh_cte("updated_vectors", self.vectors, where))
        return (
            "WITH "
            + ",\n".join(ctes)
            + f"\nSELECT {columns}\nFROM search_results\nORDER BY {order_by}"
        )

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
        refresh: bool = False,
    ) -> tuple[str, list]:
        """Build plain search query ordered by updated_at.

        Args:
            namespace_prefix: Namespace prefix to match.
            filter: Filter dict, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.
            refresh: Whether to also push back the expiry of the rows returned.

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
        order_by = "updated_at DESC, prefix, key"
        sql += f"\nORDER BY {order_by}\nLIMIT %s OFFSET %s"
        params.extend([limit, offset])
        if refresh:
            sql = self._with_refresh(sql, "prefix, key, value, created_at, updated_at", order_by)

        return (sql, params)

    # How semantic searches find their vectors. Measured with EXPLAIN on
    # v25.4.14, v26.2.7 and v26.3.2, on a vectors table of a few thousand rows
    # after ANALYZE, with the index on (prefix, embedding):
    #
    # - The C-SPANN index serves a search only when the prefix column is
    #   constrained to a set of values (prefix = x, or prefix = ANY(array))
    #   and the vectors CTE has no other filter. An expiry check in that CTE
    #   turns every search into a scan of the primary key, and FORCE_INDEX
    #   reports that the index cannot be used.
    # - A namespace and its descendants, (prefix = x OR prefix LIKE 'x.%'
    #   ESCAPE ...), is never served by the index on any of the three. On
    #   v25.4 the LIKE with ESCAPE is not even turned into a key span, so it
    #   scans the whole table; v26.2 and v26.3 scan the prefix's key span. A
    #   subquery, prefix IN (SELECT DISTINCT ...), does not help either.
    #
    # So a search under a namespace takes two steps. namespace_lookup finds
    # the namespaces that have vectors, seeking once per namespace, and
    # semantic_search_in ranks with prefix = ANY(those namespaces), which the
    # index serves. A search of the whole store has no prefix to constrain
    # and uses semantic_search, a scan with exact ranking. Results are the
    # same either way: expiry and filters are checked on the joined items,
    # and vectors expire with their item. An expired vector that was not yet
    # deleted can take one of the over_fetch slots, like a filtered-out item.

    def semantic_search(
        self,
        *,
        namespace_prefix: Sequence[str],
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        over_fetch: int,
        refresh: bool = False,
    ) -> tuple[str, list]:
        """Build a semantic search in one statement, ranked by vector distance.

        The store uses it for searches of the whole store, and
        semantic_search_in for searches under a namespace.

        Args:
            namespace_prefix: Namespace prefix to match.
            filter: Filter dict, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.
            over_fetch: Limit for the vector CTE (top-k from index).
            refresh: Whether to also push back the expiry of the rows returned.

        Returns:
            Tuple of (SQL, parameters). The query vector appears as the
            QUERY_VECTOR placeholder twice.

        Raises:
            ValueError: If limit or offset is negative, or there is no index.
        """
        condition, condition_params = prefix_condition("prefix", namespace_prefix)
        return self._semantic(
            condition,
            condition_params,
            filter=filter,
            limit=limit,
            offset=offset,
            omit_expired=omit_expired,
            over_fetch=over_fetch,
            refresh=refresh,
        )

    def semantic_search_in(
        self,
        *,
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        over_fetch: int,
        refresh: bool = False,
    ) -> tuple[str, list]:
        """Build a semantic search within the namespaces namespace_lookup found.

        The vectors CTE matches ``prefix = ANY(...)``, which the C-SPANN
        index serves. Bind the namespaces in place of NAMESPACES and the
        query vector in place of QUERY_VECTOR.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If limit or offset is negative, or there is no index.
        """
        return self._semantic(
            "prefix = ANY(CAST(%s AS STRING[]))",
            [NAMESPACES],
            filter=filter,
            limit=limit,
            offset=offset,
            omit_expired=omit_expired,
            over_fetch=over_fetch,
            refresh=refresh,
        )

    def namespace_lookup(self, namespace_prefix: Sequence[str]) -> tuple[str, list]:
        """Build the query for the namespaces under a prefix that have vectors.

        A recursive CTE seeks to the next namespace in the primary key until
        none is left, so it reads one row per namespace, not every vector.
        Expired vectors count; the search itself checks expiry.

        Returns:
            Tuple of (SQL, parameters). Each row has a ``prefix``.

        Raises:
            ValueError: If the prefix is empty. A search of the whole store
                needs no lookup.
        """
        if not namespace_prefix:
            raise ValueError("namespace_lookup needs a non-empty namespace prefix")
        if self._index is None:
            raise ValueError("namespace_lookup requires an index configuration")
        condition, params = _prefix_range("prefix", namespace_prefix)
        inner, inner_params = _prefix_range("v.prefix", namespace_prefix)
        sql = f"""WITH RECURSIVE found (prefix) AS (
    (SELECT prefix FROM {self.vectors} WHERE {condition} ORDER BY prefix LIMIT 1)
    UNION ALL
    SELECT (
        SELECT v.prefix FROM {self.vectors} AS v
        WHERE v.prefix > found.prefix AND {inner}
        ORDER BY v.prefix LIMIT 1
    )
    FROM found
    WHERE found.prefix IS NOT NULL
)
SELECT prefix FROM found WHERE prefix IS NOT NULL"""
        return (sql, params + inner_params)

    def _semantic(
        self,
        condition: str,
        condition_params: Sequence[Any],
        *,
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        over_fetch: int,
        refresh: bool,
    ) -> tuple[str, list]:
        """Build a semantic search whose vectors CTE matches ``condition``.

        The CTE ranks vectors by distance and keeps ``over_fetch`` of them.
        The items join applies the filter and expiry, then DISTINCT ON keeps
        the best distance of each item. Score is 1 - distance for cosine and
        the negated distance for L2 and inner product, so higher is better.
        """
        _validate_paging(limit, offset)
        if self._index is None:
            raise ValueError("semantic_search requires an index configuration")

        operator = self._index.distance.get_operator()
        if self._index.distance == DistanceStrategy.COSINE:
            score = "1 - r.distance AS score"
        else:
            score = "-r.distance AS score"

        # No expiry check here: see the notes above semantic_search
        sql = f"""WITH ranked AS (
    SELECT prefix, key, field_name, embedding {operator} CAST(%s AS VECTOR) AS distance
    FROM {self.vectors}
    WHERE {condition}
    ORDER BY embedding {operator} CAST(%s AS VECTOR)
    LIMIT %s
),
best AS (
    SELECT DISTINCT ON (i.prefix, i.key)
        i.prefix, i.key, i.value, i.created_at, i.updated_at,
        {score}
    FROM ranked r
    JOIN {self.items} i ON r.prefix = i.prefix AND r.key = i.key
    WHERE """
        params: list[Any] = [QUERY_VECTOR, *condition_params, QUERY_VECTOR, over_fetch]

        conditions, filter_params = translate_filter(filter, value_column="i.value")
        if omit_expired:
            conditions = [*conditions, "(i.expires_at IS NULL OR i.expires_at > now())"]
        sql += " AND ".join(conditions) if conditions else "TRUE"
        params.extend(filter_params)

        sql += """
    ORDER BY i.prefix, i.key, r.distance
)
SELECT prefix, key, value, created_at, updated_at, score
FROM best
ORDER BY score DESC, prefix, key
LIMIT %s OFFSET %s"""
        params.extend([limit, offset])
        if refresh:
            sql = self._with_refresh(
                sql, "prefix, key, value, created_at, updated_at, score", "score DESC, prefix, key"
            )
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

    def delete_namespace(self, namespace_prefix: Sequence[str]) -> tuple[str, list]:
        """Build one DELETE for a namespace and its descendants.

        With an index, the vectors go in a data-modifying CTE of the same
        statement.

        Args:
            namespace_prefix: Namespace prefix to delete.

        Returns:
            Tuple of (SQL, parameters). The statement returns the key of
            every item deleted.
        """
        ns_condition, ns_params = prefix_condition("prefix", namespace_prefix)
        sql = f"DELETE FROM {self.items}\nWHERE {ns_condition}\nRETURNING key"
        if self._index is None:
            return (sql, list(ns_params))
        cte = (
            f"WITH deleted_vectors AS (\n    DELETE FROM {self.vectors}\n"
            f"    WHERE {ns_condition}\n    RETURNING key\n)\n"
        )
        return (cte + sql, list(ns_params) + list(ns_params))

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
