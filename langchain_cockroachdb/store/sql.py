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

# Most namespaces a semantic search sends to the vector index in one array.
# namespace_lookup stops after one more, and a search whose lookup finds
# more uses the exact one-statement form instead, so the lookup, the array
# and the index probes stay bounded.
MAX_SEARCH_NAMESPACES = 1000

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

        SET LOCAL lasts until the transaction ends. The store runs it in the
        search group's own transaction. But if a caller searches inside a
        transaction it opened on a single connection it shares with the
        store, the setting lasts until that caller's transaction ends.

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
    # - A namespace and its descendants, as the range from prefix_condition:
    #   a ranked CTE with it scans the vectors primary key over the prefix's
    #   key span on v25.4.14, and FORCE_INDEX on the vector index fails with
    #   "cannot be used for this query". The earlier LIKE 'x.%' ESCAPE form
    #   was not served by the index on any of the three either, and v25.4
    #   scanned the whole table for it.
    # - Nor are the namespaces computed in the same statement. Measured on
    #   the vectors CTE alone, on v25.4.14 and v26.3.2: with prefix =
    #   ANY(ARRAY(SELECT DISTINCT prefix FROM <items or vectors> WHERE
    #   <range>)) the subquery runs first, then the vectors table is scanned
    #   in full; with prefix IN (SELECT DISTINCT prefix FROM <items> WHERE
    #   <range>) the plan is a lookup join into the vectors primary key. For
    #   the ANY(ARRAY(...)) form FORCE_INDEX fails with the same error.
    #
    # So a search under a namespace takes two steps, two round trips. A
    # namespace created between the lookup and the search is not searched,
    # which is the same as the search having run a moment earlier.
    # namespace_lookup finds the namespaces that have vectors, seeking once
    # per namespace, and semantic_search_in ranks with prefix = ANY(those
    # namespaces), which the index serves. Its ranked CTE keeps the
    # over_fetch nearest vectors with no filter and no expiry check, and only
    # then are the items joined, filtered and checked for expiry. A
    # candidate that fails the filter, or whose item expired and was not
    # deleted yet, still takes a place in that window. So when the page
    # comes back short, the store runs exact_semantic_search_in over the
    # same namespaces (needs_exact_search in base.py decides), unless the
    # index returned every vector those namespaces hold. A window that is
    # not full does not show that by itself: C-SPANN can return fewer than
    # LIMIT rows while more exist. On v26.3.2, with 3,000 vectors in one
    # namespace, beam size 1 returned 57 rows and beam size 4 returned 273
    # for every LIMIT above that, and small partitions can cap it the same
    # way. So every row carries window_size, the rows the window kept, and
    # namespace_vectors, a count of the namespaces' vectors that stops at
    # over_fetch, read from the primary key. A short page is exact only when
    # the window is not full and the two are equal.
    #
    # The exact forms, semantic_search and exact_semantic_search_in, join the
    # items, filter and check expiry before they rank, as PostgresStore does.
    # Their ranked CTE has no LIMIT that the vector index could serve, so
    # they scan the vectors of their namespaces and rank every match. The
    # store uses semantic_search for a search of the whole store, for every
    # search of a flat store (no lookup either, since there is no index for
    # it to serve), and when a lookup finds more than MAX_SEARCH_NAMESPACES
    # namespaces.
    #
    # EXPLAIN on v26.3.2, with 60 namespaces of 100 vectors and one of 3,000,
    # a C-SPANN index and a filter on the value, after ANALYZE:
    # - semantic_search and exact_semantic_search_in: top-k, render, group
    #   (streaming), render, lookup join (merge join without a filter),
    #   filter, scan of the items primary key, over the namespace's spans
    #   (FULL SCAN for the whole store). No vector search, though the vector
    #   index exists.
    # - semantic_search_in: top-k, render, distinct, render, sort, render,
    #   lookup join into the items, window, top-k, render, lookup join into
    #   the vectors primary key, vector search on the vector index. Where a
    #   scan is cheaper, such as 100 vectors and a window of 21, the bottom
    #   two are a scan of the vectors primary key instead, the same choice
    #   the optimizer made before the window was added. Next to it, under
    #   root, namespace_vectors is a subquery: group (scalar) over a scan of
    #   the vectors primary key, over the namespaces' spans, with limit:
    #   over_fetch.

    def semantic_search(
        self,
        *,
        namespace_prefix: Sequence[str],
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        refresh: bool = False,
    ) -> tuple[str, list]:
        """Build an exact semantic search of a namespace and its descendants.

        It filters and checks expiry before it ranks, so it returns what
        ranking every matching item would. The store uses it for a search of
        the whole store, every search of a flat store, and a search whose
        lookup finds more than MAX_SEARCH_NAMESPACES namespaces.

        Args:
            namespace_prefix: Namespace prefix to match.
            filter: Filter dict, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.
            refresh: Whether to also push back the expiry of the rows returned.

        Returns:
            Tuple of (SQL, parameters). The query vector appears as the
            QUERY_VECTOR placeholder once.

        Raises:
            ValueError: If limit or offset is negative, or there is no index.
        """
        condition, condition_params = prefix_condition("v.prefix", namespace_prefix)
        return self._exact(
            condition,
            condition_params,
            filter=filter,
            limit=limit,
            offset=offset,
            omit_expired=omit_expired,
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
        """Build an indexed semantic search within the namespaces namespace_lookup found.

        The ranked CTE matches ``prefix = ANY(...)``, which the C-SPANN
        index serves, and keeps the ``over_fetch`` nearest vectors. Every row
        carries ``window_size``, the number of vectors the CTE kept, and
        ``namespace_vectors``, the number of vectors the namespaces hold,
        counted from the primary key up to ``over_fetch``. The store compares
        them to tell an exact short result from one the index cut short (see
        needs_exact_search). Bind the namespaces in place of NAMESPACES and
        the query vector in place of QUERY_VECTOR.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If limit or offset is negative, or there is no index.
        """
        _validate_paging(limit, offset)
        if self._index is None:
            raise ValueError("semantic_search requires an index configuration")

        operator = self._index.distance.get_operator()
        conditions, filter_params = self._item_conditions(filter, omit_expired)
        where = " AND ".join(conditions) if conditions else "TRUE"
        # No filter or expiry check in ranked: see the notes above semantic_search.
        # count(*) OVER () runs on the rows the inner LIMIT kept. The count in
        # the outer SELECT reads at most over_fetch rows of the primary key
        sql = f"""WITH ranked AS (
    SELECT prefix, key, distance, count(*) OVER () AS window_size
    FROM (
        SELECT prefix, key, embedding {operator} CAST(%s AS VECTOR) AS distance
        FROM {self.vectors}
        WHERE prefix = ANY(CAST(%s AS STRING[]))
        ORDER BY embedding {operator} CAST(%s AS VECTOR)
        LIMIT %s
    ) AS nearest
),
best AS (
    SELECT DISTINCT ON (i.prefix, i.key)
        i.prefix, i.key, i.value, i.created_at, i.updated_at,
        {self._score("r.distance")} AS score, r.window_size
    FROM ranked AS r
    JOIN {self.items} AS i ON r.prefix = i.prefix AND r.key = i.key
    WHERE {where}
    ORDER BY i.prefix, i.key, r.distance
)
SELECT prefix, key, value, created_at, updated_at, score, window_size, (
    SELECT count(*) FROM (
        SELECT 1 FROM {self.vectors} WHERE prefix = ANY(CAST(%s AS STRING[])) LIMIT %s
    ) AS c
) AS namespace_vectors
FROM best
ORDER BY score DESC, prefix, key
LIMIT %s OFFSET %s"""
        params: list[Any] = [
            QUERY_VECTOR,
            NAMESPACES,
            QUERY_VECTOR,
            over_fetch,
            *filter_params,
            NAMESPACES,
            over_fetch,
            limit,
            offset,
        ]
        if refresh:
            sql = self._with_refresh(
                sql,
                "prefix, key, value, created_at, updated_at, score, window_size, namespace_vectors",
                "score DESC, prefix, key",
            )
        return (sql, params)

    def exact_semantic_search_in(
        self,
        *,
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        refresh: bool = False,
    ) -> tuple[str, list]:
        """Build an exact semantic search within the namespaces namespace_lookup found.

        The store runs it when semantic_search_in comes back short from a
        full window. Bind the namespaces in place of NAMESPACES and the query
        vector in place of QUERY_VECTOR.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If limit or offset is negative, or there is no index.
        """
        return self._exact(
            "v.prefix = ANY(CAST(%s AS STRING[]))",
            [NAMESPACES],
            filter=filter,
            limit=limit,
            offset=offset,
            omit_expired=omit_expired,
            refresh=refresh,
        )

    def namespace_lookup(self, namespace_prefix: Sequence[str]) -> tuple[str, list]:
        """Build the query for the namespaces under a prefix that have vectors.

        A recursive CTE seeks to the next namespace in the primary key until
        none is left, so it reads one row per namespace, not every vector.
        It counts them and stops after MAX_SEARCH_NAMESPACES + 1, enough for
        the store to know it must fall back to semantic_search. Expired
        vectors count; the search itself checks expiry.

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
        condition, params = prefix_condition("prefix", namespace_prefix)
        inner, inner_params = prefix_condition("v.prefix", namespace_prefix)
        sql = f"""WITH RECURSIVE found (prefix, n) AS (
    (SELECT prefix, 1 FROM {self.vectors} WHERE {condition} ORDER BY prefix LIMIT 1)
    UNION ALL
    SELECT (
        SELECT v.prefix FROM {self.vectors} AS v
        WHERE v.prefix > found.prefix AND {inner}
        ORDER BY v.prefix LIMIT 1
    ), found.n + 1
    FROM found
    WHERE found.prefix IS NOT NULL AND found.n <= %s
)
SELECT prefix FROM found WHERE prefix IS NOT NULL"""
        return (sql, [*params, *inner_params, MAX_SEARCH_NAMESPACES])

    def _score(self, distance: str) -> str:
        """The score of a distance expression, higher is better.

        1 - distance for cosine, the negated distance for L2 and inner product.
        """
        assert self._index is not None
        if self._index.distance == DistanceStrategy.COSINE:
            return f"1 - {distance}"
        return f"-{distance}"

    @staticmethod
    def _item_conditions(filter: Mapping | None, omit_expired: bool) -> tuple[list[str], list]:
        """The filter and expiry conditions on the joined items, ``i``."""
        conditions, params = translate_filter(filter, value_column="i.value")
        if omit_expired:
            conditions = [*conditions, "(i.expires_at IS NULL OR i.expires_at > now())"]
        return conditions, params

    def _exact(
        self,
        condition: str,
        condition_params: Sequence[Any],
        *,
        filter: Mapping | None,
        limit: int,
        offset: int,
        omit_expired: bool,
        refresh: bool,
    ) -> tuple[str, list]:
        """Build an exact semantic search over the vectors that match ``condition``.

        ``condition`` is on ``v.prefix``. The ranked CTE joins the items and
        applies the filter and the expiry check before anything is cut, then
        keeps the best distance of each item. Only the outer query has a
        LIMIT, so no vector index can serve it and every match is ranked.
        """
        _validate_paging(limit, offset)
        if self._index is None:
            raise ValueError("semantic_search requires an index configuration")

        operator = self._index.distance.get_operator()
        conditions, filter_params = self._item_conditions(filter, omit_expired)
        where = " AND ".join([condition, *conditions])
        # No LIMIT in ranked: see the notes above semantic_search
        sql = f"""WITH ranked AS (
    SELECT i.prefix, i.key, i.value, i.created_at, i.updated_at,
        min(v.embedding {operator} CAST(%s AS VECTOR)) AS distance
    FROM {self.vectors} AS v
    JOIN {self.items} AS i ON v.prefix = i.prefix AND v.key = i.key
    WHERE {where}
    GROUP BY i.prefix, i.key
)
SELECT prefix, key, value, created_at, updated_at, {self._score("distance")} AS score
FROM ranked
ORDER BY score DESC, prefix, key
LIMIT %s OFFSET %s"""
        params: list[Any] = [QUERY_VECTOR, *condition_params, *filter_params, limit, offset]
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
        prefixes: Sequence[Sequence[str]] = (),
    ) -> tuple[str, list]:
        """Build query to list distinct namespaces.

        The regexes decide which namespaces match. Each of ``prefixes`` adds
        the range from prefix_condition next to them, so a listing under
        one user reads only that user's key span, whatever the optimizer
        makes of the regex. (v26.3.2 already derives a wider span from an
        anchored regex: [user.u007, user.u008) for user.u007.) A namespace a
        prefix regex matches always starts with that condition's literal
        labels, so the range leaves out nothing it would match.

        Args:
            match_conditions: Regex pattern strings (from match_pattern).
            max_depth: Maximum namespace depth for truncation, or None.
            limit: Maximum results.
            offset: Results to skip.
            omit_expired: Whether to exclude expired items.
            prefixes: The literal leading labels of each prefix condition
                (from literal_prefix). Empty ones add nothing.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If limit or offset is negative.
        """
        _validate_paging(limit, offset)

        # Build truncation expression or use prefix directly
        ns_expr = truncate_expression("prefix", max_depth) if max_depth is not None else "prefix"

        conditions: list[str] = []
        params: list[Any] = []
        if omit_expired:
            conditions.append("(expires_at IS NULL OR expires_at > now())")
        for labels in prefixes:
            if labels:
                condition, condition_params = prefix_condition("prefix", labels)
                conditions.append(condition)
                params.extend(condition_params)
        for pattern in match_conditions:
            conditions.append("prefix ~ %s")
            params.append(pattern)

        where = " AND ".join(conditions) if conditions else "TRUE"
        sql = (
            f"SELECT DISTINCT {ns_expr} AS ns\nFROM {self.items}\nWHERE {where}"
            "\nORDER BY ns\nLIMIT %s OFFSET %s"
        )
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
