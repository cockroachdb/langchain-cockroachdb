"""SQL statement building for the CockroachDB store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from langgraph.store.base import NOT_PROVIDED, NotProvided

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

# The migration that adds the columns reconcile, history and confirm use.
# Until setup() applies it, the store writes with the 0.5 statements.
MIGRATION_RECONCILE = 2
# The columns migration 2 adds. A store whose role cannot read the migrations
# table, or that finds no migrations table, looks for all of them in the
# items table instead.
_RECONCILE_COLUMNS = (
    "version",
    "provenance",
    "confirmed_at",
    "confirmations",
    "write_id",
    "valid_at",
    "confirm_id",
)

# Most namespaces a semantic search sends to the vector index in one array.
# namespace_lookup stops after one more, and a search whose lookup finds
# more uses the exact one-statement form instead, so the lookup, the array
# and the index probes stay bounded.
MAX_SEARCH_NAMESPACES = 1000

# Type aliases
ItemRow = tuple[str, str, str, float | None]  # prefix, key, value_json, ttl_minutes
ItemRowV2 = tuple[
    str, str, str, float | None, str | None, datetime | None
]  # prefix, key, value_json, ttl_minutes, provenance_json, valid_at
VectorRow = tuple[
    str, str, str, list[float], float | None
]  # prefix, key, field_name, embedding, ttl_minutes

# The columns of a candidate row, as the candidate statements return them
_CANDIDATE_NAMES = (
    "key",
    "value",
    "created_at",
    "updated_at",
    "version",
    "valid_at",
    "provenance",
    "confirmed_at",
    "confirmations",
    "confirm_id",
)
_CANDIDATE_OUTPUT = ", ".join(_CANDIDATE_NAMES)


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
        self.history = f"{self._schema}.{self._table_name}_history"

        # Vector index name (public)
        self.vector_index_name = f"{self._table_name}_vectors_embedding_idx"

    def migrations(self) -> list[str]:
        """Return numbered migration statements for items table.

        Returns:
            List of 3 idempotent DDL statements: (0) migrations table, (1) items
            table, (2) the columns reconcile, history and confirm use.
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
            # Migration 2 (MIGRATION_RECONCILE): one statement, so it is one
            # schema change in its own implicit transaction, as a schema_locked
            # table needs. Existing rows get version 1 and no confirmations.
            # write_id marks the rows a write wrote and confirm_id the rows a
            # confirmation confirmed, so a retry of either finds its own work.
            f"""ALTER TABLE {self.items} ADD COLUMN IF NOT EXISTS version INT8 NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS provenance JSONB,
    ADD COLUMN IF NOT EXISTS confirmed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS confirmations INT8 NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS write_id UUID,
    ADD COLUMN IF NOT EXISTS valid_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS confirm_id UUID;""",
        ]

    def migration_version(self) -> str:
        """The statement that reads the latest migration recorded, if any.

        setup() reads it to find the migrations it still has to run.
        """
        return f"SELECT v FROM {self.migrations_table} ORDER BY v DESC LIMIT 1"

    def reconcile_migration_recorded(self) -> tuple[str, list]:
        """The statement that tells whether setup() has recorded migration 2, as ``ready``.

        The schema check reads it first. setup() records a migration only
        after its schema change has settled on every node, so once the row
        exists every node can use the new columns.
        """
        return (
            f"SELECT EXISTS (SELECT 1 FROM {self.migrations_table} WHERE v >= %s) AS ready",
            [MIGRATION_RECONCILE],
        )

    def reconcile_columns_exist(self) -> tuple[str, list]:
        """The statement that tells whether the items table has migration 2's columns, as ``ready``.

        The schema check reads it instead of the migrations table when the
        store's role has no grant on that table. pg_attribute lists only
        public columns, so columns that a schema change is still adding do
        not count. A missing items table has none of them.
        """
        return (
            "SELECT count(*) = %s AS ready FROM pg_catalog.pg_attribute\n"
            "WHERE attrelid = to_regclass(%s) AND NOT attisdropped AND attname = ANY(%s)",
            [len(_RECONCILE_COLUMNS), self.items, list(_RECONCILE_COLUMNS)],
        )

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

    def history_statements(self, ttl_days: float | None) -> list[str]:
        """Return idempotent DDL statements for the history table.

        setup() runs them only on a store with history, each on its own.

        Args:
            ttl_days: The history's TTL in days, or None to keep it. With a
                TTL, CockroachDB's row-level TTL deletes a version once its
                ``expires_at`` passes. The archive statement sets that.

        Returns:
            [history table], and the TTL setting with ``ttl_days``.
        """
        stmts = [
            f"""CREATE TABLE IF NOT EXISTS {self.history} (
    prefix TEXT NOT NULL,
    key TEXT NOT NULL,
    version INT8 NOT NULL,
    value JSONB NOT NULL,
    provenance JSONB,
    valid_at TIMESTAMPTZ NOT NULL,
    invalid_at TIMESTAMPTZ,
    written_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL,
    superseded_by TEXT,
    confirmations INT8 NOT NULL DEFAULT 0,
    confirmed_at TIMESTAMPTZ,
    write_id UUID,
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (prefix, key, version)
);"""
        ]
        if ttl_days is not None:
            stmts.append(
                f"ALTER TABLE {self.history} SET "
                "(ttl_expiration_expression = 'expires_at', ttl_job_cron = '@hourly');"
            )
        return stmts

    def history_exists(self) -> tuple[str, list]:
        """The statement that tells whether the history table exists, as ``ready``."""
        return ("SELECT to_regclass(%s) IS NOT NULL AS ready", [self.history])

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

    def upsert_items(
        self,
        rows: Sequence[ItemRow] | Sequence[ItemRowV2],
        *,
        write_id: str | None = None,
        history: bool = False,
        return_version: bool = False,
    ) -> tuple[str, list]:
        """Build multi-row INSERT for items with ON CONFLICT.

        Without ``write_id`` this is the 0.5 statement, for a table that
        setup() has not given migration 2's columns yet. With it, the rows'
        provenance, valid time and the write id are written too, and an
        update bumps the row's version. A row that already carries the write
        id is left as it is, so a batch retried after its commit changes nothing.

        Confirmations belong to the fact, not the key: an update that changes
        the value starts confirmations and confirmed_at again from none,
        while an exact re-save of the live value keeps them. A memory that
        expired counts as gone, so a write over it starts from none as well.
        The id of the last confirmation stays on the row, so a confirmation
        retried after it committed still finds it and counts nothing again.

        Args:
            rows: (prefix, key, value_json, ttl_minutes) tuples without
                ``write_id``, (prefix, key, value_json, ttl_minutes,
                provenance_json, valid_at) tuples with it. Callers must dedupe
                (prefix, key) rows (ON CONFLICT cannot touch a row twice).
            write_id: The batch's write id, chosen before its first attempt.
            history: For a store with history. A key's versions then stay
                unique across its live row and its history: every new
                version, inserted or updated, is one more than the larger of
                the live version and the highest version the key has in the
                history table. So a key written again after a delete, or
                after reconcile recorded a late fact under it straight into
                history, never reuses a version kept there. Needs ``write_id``.
            return_version: Whether the statement returns the version of
                each row it inserts or updates, as ``version``. A row it
                leaves as it is returns nothing. Needs ``write_id``.

        Returns:
            Tuple of (SQL, parameters).

        Raises:
            ValueError: If rows is empty, a row has the wrong number of
                fields, or ``history`` or ``return_version`` is set without
                ``write_id``.
        """
        if not rows:
            raise ValueError("rows cannot be empty")
        if history and write_id is None:
            raise ValueError("upsert_items with history needs a write id")
        if return_version and write_id is None:
            raise ValueError("upsert_items needs a write id to return the version")

        columns = [
            ("prefix", "STRING"),
            ("key", "STRING"),
            ("value", "JSONB"),
            ("ttl_minutes", "DECIMAL"),
        ]
        if write_id is not None:
            columns += [("provenance", "JSONB"), ("valid_at", "TIMESTAMPTZ")]

        # Use UNION ALL to reference ttl_minutes only once per row, with typed placeholders
        first = "SELECT " + ", ".join(f"CAST(%s AS {kind}) AS {name}" for name, kind in columns)
        others = "UNION ALL SELECT " + ", ".join(f"CAST(%s AS {kind})" for _, kind in columns)
        union_sql = "\n    ".join([first] + [others] * (len(rows) - 1))

        # The write id's placeholder comes first, in the outer SELECT. Then
        # each row's fields, in the order of columns
        params: list[Any] = [] if write_id is None else [write_id]
        for row in rows:
            if len(row) != len(columns):
                raise ValueError(
                    f"item rows have {len(columns)} fields "
                    f"{'without' if write_id is None else 'with'} a write id, got {len(row)}"
                )
            params.extend(row)

        if write_id is None:
            sql = f"""INSERT INTO {self.items} (prefix, key, value, expires_at, ttl_minutes)
SELECT prefix, key, value, now() + CAST(ttl_minutes AS DECIMAL) * INTERVAL '1 minute', ttl_minutes
FROM ({union_sql}) AS rows_to_insert
ON CONFLICT (prefix, key) DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = now(),
    expires_at = EXCLUDED.expires_at,
    ttl_minutes = EXCLUDED.ttl_minutes"""
            return (sql, params)

        # Without history a new row starts at version 1, the column's
        # default, and an update adds one. With history both take the next
        # version over the live row and the key's history, computed once here
        version_column, version_value = "", ""
        version_update = f"{self.items}.version + 1"
        if history:
            version_column = ", version"
            version_value = ",\n    " + self._next_version("rows_to_insert")
            version_update = "EXCLUDED.version"
        # The SET expressions read the row as it was before the update
        same_fact = (
            f"{self.items}.value = EXCLUDED.value AND "
            f"({self.items}.expires_at IS NULL OR {self.items}.expires_at > now())"
        )
        sql = f"""INSERT INTO {self.items} (prefix, key, value, expires_at, ttl_minutes, provenance, valid_at, write_id{version_column})
SELECT prefix, key, value, now() + CAST(ttl_minutes AS DECIMAL) * INTERVAL '1 minute', ttl_minutes, provenance, valid_at, CAST(%s AS UUID){version_value}
FROM ({union_sql}) AS rows_to_insert
ON CONFLICT (prefix, key) DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = now(),
    expires_at = EXCLUDED.expires_at,
    ttl_minutes = EXCLUDED.ttl_minutes,
    provenance = EXCLUDED.provenance,
    valid_at = EXCLUDED.valid_at,
    write_id = EXCLUDED.write_id,
    version = {version_update},
    confirmations = CASE WHEN {same_fact} THEN {self.items}.confirmations ELSE 0 END,
    confirmed_at = CASE WHEN {same_fact} THEN {self.items}.confirmed_at ELSE NULL END
WHERE {self.items}.write_id IS DISTINCT FROM EXCLUDED.write_id"""
        if return_version:
            sql += "\nRETURNING version"
        return (sql, params)

    def _next_version(self, source: str) -> str:
        """The next version of the key that ``source`` names, on a store with history.

        One more than the larger of the key's version in the items table,
        live or not, and its highest version in the history table, so a
        key's versions stay unique across both. ``source`` is the alias of
        a relation with ``prefix`` and ``key`` columns.
        """
        return (
            "GREATEST(\n"
            f"        COALESCE((SELECT i.version FROM {self.items} AS i "
            f"WHERE i.prefix = {source}.prefix AND i.key = {source}.key), 0),\n"
            f"        COALESCE((SELECT max(h.version) FROM {self.history} AS h "
            f"WHERE h.prefix = {source}.prefix AND h.key = {source}.key), 0)\n"
            "    ) + 1"
        )

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

        The store runs it when semantic_search_in comes back short and its
        window may have missed vectors, as needs_exact_search decides. Bind
        the namespaces in place of NAMESPACES and the query vector in place
        of QUERY_VECTOR.

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

    # How find_conflicts, and reconcile in its transaction, find the memories
    # a new one may conflict with, in two round trips. The first sends now(),
    # key_slot_candidates and, when the similar read must choose a scan,
    # namespace_vector_count. The second sends, for each vector of the new
    # memory, candidate_scores for the key and slot candidates and
    # similar_candidates for the others. Candidates come only from the new
    # memory's own namespace, never its descendants, so every statement
    # matches prefix = %s. Expired memories are never candidates, whatever
    # omit_expired says, and none of these statements pushes back an expiry.
    #
    # A namespace with at most reconcile_scan_limit vectors gets the exact
    # form of similar_candidates: it ranks every vector of the namespace,
    # read by primary key, which also lets a serializable transaction notice
    # a concurrent insert. A larger one gets the index form: a window of the
    # nearest vectors from the C-SPANN index, as in semantic_search_in, then
    # the join, filter and expiry check. Rows that the filter, the expiry
    # check or the excluded keys remove still take places in the window, so
    # a match beyond it would be missed. The index form therefore reports
    # the window's size and lowest score, over every vector in it, and the
    # namespace's vectors counted up to over_fetch, as semantic_search_in
    # does. When it returned limit rows, or the window is whole and its
    # lowest score is below the threshold, no vector beyond the window can
    # change the result, up to the index's approximation, and the result
    # stands. Otherwise matches may lie beyond the window: the store runs
    # the exact form for that vector (needs_exact_candidates in base.py).
    # The window is not whole when it holds fewer than over_fetch vectors
    # while the namespace holds more: C-SPANN cut it short, as a small
    # beam size or small partitions can. The exact form costs one more
    # round trip, and only when fewer than limit rows pass.
    #
    # Measured with EXPLAIN on v26.3.3, with 3,000 vectors in one namespace
    # and a C-SPANN index: the index form computes nearest once, as a buffer,
    # with a vector search on the vector index over the namespace's prefix
    # span under a top-k of over_fetch; the window's count and lowest score
    # read that buffer, and so do the excluded keys' filter and the items
    # lookup join; the namespace's count is a scan of the vectors primary
    # key over the namespace's span, with limit over_fetch. The exact form
    # is a merge join of the items and vectors primary keys over the
    # namespace's span. candidate_scores reads its keys from the vectors
    # primary key. test_only_the_index_form_uses_the_vector_index checks
    # the index form's plan.

    @staticmethod
    def _candidate_columns(alias: str = "") -> str:
        """The columns of a candidate row, read from the items table under ``alias``.

        A row without a valid time, from 0.5.x or a plain put, reads as valid
        from its updated_at. confirm_id, the id of the memory's last
        confirmation, tells a confirm retried after a lost commit that its
        earlier attempt confirmed the memory.
        """
        a = f"{alias}." if alias else ""
        return (
            f"{a}key, {a}value, {a}created_at, {a}updated_at, {a}version, "
            f"COALESCE({a}valid_at, {a}updated_at) AS valid_at, {a}provenance, "
            f"{a}confirmed_at, {a}confirmations, {a}confirm_id"
        )

    @staticmethod
    def now() -> tuple[str, list]:
        """The statement that reads the database's clock, as ``now``.

        In a transaction it reads the transaction's timestamp, which is also
        the updated_at of every row the transaction writes.
        """
        return ("SELECT now() AS now", [])

    def key_slot_candidates(
        self,
        prefix: str,
        key: str,
        *,
        slot_field: str | None,
        slot_value_json: str | None,
        filter: Mapping | None,
    ) -> tuple[str, list]:
        """Build the read of a new memory's target and of the memories in its slot.

        The live row under ``key`` comes back whatever ``filter`` says. With a
        slot, so do the live rows whose value holds the same value in the
        slot field, compared as JSON, if they also match ``filter``. The slot
        field and value are bound, never formatted into the text.

        Args:
            prefix: Namespace text. Its descendants are not read.
            key: The new memory's key.
            slot_field: The top-level field that holds the slot, or None.
            slot_value_json: The slot's value serialized as JSON, or None.
            filter: Narrows the slot matches, as in search.

        Returns:
            Tuple of (SQL, parameters). Each row has the candidate columns.

        Raises:
            ValueError: If the filter has an unknown operator.
        """
        params: list[Any] = [prefix, key]
        match = "key = %s"
        if slot_field is not None:
            conditions, filter_params = translate_filter(filter)
            slot = " AND ".join(["value -> %s = CAST(%s AS JSONB)", *conditions])
            match = f"(key = %s OR ({slot}))"
            params.extend([slot_field, slot_value_json, *filter_params])
        sql = f"""SELECT {self._candidate_columns()}
FROM {self.items}
WHERE prefix = %s AND {match}
    AND (expires_at IS NULL OR expires_at > now())"""
        return (sql, params)

    def similar_candidates(
        self,
        prefix: str,
        *,
        threshold: float,
        limit: int,
        filter: Mapping | None,
        scan: Literal["exact", "index"],
        over_fetch: int,
        exclude: Sequence[str] = (),
    ) -> tuple[str, list]:
        """Build the read of the memories similar to one vector of a new memory.

        Every row is live, matches ``filter`` and scores at or above
        ``threshold`` with the store's score function. Rows come best first,
        then by key, up to ``limit``. The ``exclude`` keys are left out: the
        target and the slot matches, which candidate_scores scores.

        The index form also reports the window it ranked: ``window_size``,
        the vectors the C-SPANN index returned, ``lowest_score``, the lowest
        score among them, before the filter, the expiry check and the
        excluded keys, and ``namespace_vectors``, the namespace's vectors
        counted from the primary key up to ``over_fetch``. Every row carries
        all three, and when no row passes, one row still does, with NULL
        candidate columns. See needs_exact_candidates in base.py.

        Args:
            prefix: Namespace text. Its descendants are not read.
            threshold: The lowest score a row may have.
            limit: Most rows.
            filter: Narrows the rows by value, as in search.
            scan: "exact" ranks every vector of the namespace. "index" ranks
                the ``over_fetch`` nearest vectors the C-SPANN index returns,
                then filters them.
            over_fetch: The window of the index scan. The exact scan has none.
            exclude: Keys to leave out.

        Returns:
            Tuple of (SQL, parameters). Each row has the candidate columns
            and ``score``, and with the index scan ``window_size``,
            ``lowest_score`` and ``namespace_vectors``. Bind the query vector
            in place of QUERY_VECTOR.

        Raises:
            ValueError: If limit is negative, the scan is unknown, there is no
                index, or the filter has an unknown operator.
        """
        _validate_paging(limit, 0)
        if self._index is None:
            raise ValueError("similar_candidates requires an index configuration")
        if scan not in ("exact", "index"):
            raise ValueError(f"scan must be 'exact' or 'index', got {scan!r}")
        operator = self._index.distance.get_operator()
        conditions, filter_params = self._item_conditions(filter, omit_expired=True)
        score = self._score("distance")
        if scan == "exact":
            # No LIMIT in ranked: see the notes above _candidate_columns
            where = " AND ".join(
                ["v.prefix = %s", "NOT (v.key = ANY(CAST(%s AS STRING[])))", *conditions]
            )
            sql = f"""WITH ranked AS (
    SELECT {self._candidate_columns("i")},
        min(v.embedding {operator} CAST(%s AS VECTOR)) AS distance
    FROM {self.vectors} AS v
    JOIN {self.items} AS i ON v.prefix = i.prefix AND v.key = i.key
    WHERE {where}
    GROUP BY i.prefix, i.key
)
SELECT {_CANDIDATE_OUTPUT}, {score} AS score
FROM ranked
WHERE {score} >= CAST(%s AS FLOAT8)
ORDER BY score DESC, key
LIMIT %s"""
            params: list[Any] = [
                QUERY_VECTOR,
                prefix,
                list(exclude),
                *filter_params,
                threshold,
                limit,
            ]
            return (sql, params)

        # No filter in nearest, so the C-SPANN index serves it: see the notes
        # above semantic_search. window_stats reads every vector it kept, and
        # the LEFT JOIN returns its row even when no candidate passes
        where = " AND ".join(["NOT (i.key = ANY(CAST(%s AS STRING[])))", *conditions])
        hit_columns = ", ".join(f"h.{name}" for name in _CANDIDATE_NAMES)
        sql = f"""WITH nearest AS (
    SELECT prefix, key, embedding {operator} CAST(%s AS VECTOR) AS distance
    FROM {self.vectors}
    WHERE prefix = %s
    ORDER BY embedding {operator} CAST(%s AS VECTOR)
    LIMIT %s
),
window_stats AS (
    SELECT count(*) AS window_size, min({score}) AS lowest_score, (
        SELECT count(*) FROM (
            SELECT 1 FROM {self.vectors} WHERE prefix = %s LIMIT %s
        ) AS c
    ) AS namespace_vectors
    FROM nearest
),
best AS (
    SELECT DISTINCT ON (i.key)
        {self._candidate_columns("i")}, r.distance
    FROM nearest AS r
    JOIN {self.items} AS i ON r.prefix = i.prefix AND r.key = i.key
    WHERE {where}
    ORDER BY i.key, r.distance
),
hits AS (
    SELECT {_CANDIDATE_OUTPUT}, {score} AS score
    FROM best
    WHERE {score} >= CAST(%s AS FLOAT8)
    ORDER BY score DESC, key
    LIMIT %s
)
SELECT {hit_columns}, h.score, w.window_size, w.lowest_score, w.namespace_vectors
FROM window_stats AS w
LEFT JOIN hits AS h ON TRUE
ORDER BY h.score DESC, h.key"""
        params = [
            QUERY_VECTOR,
            prefix,
            QUERY_VECTOR,
            over_fetch,
            prefix,
            over_fetch,
            list(exclude),
            *filter_params,
            threshold,
            limit,
        ]
        return (sql, params)

    def candidate_scores(self, prefix: str, keys: Sequence[str]) -> tuple[str, list]:
        """Build the scores of some memories against one vector of a new memory.

        A memory scores as its best vector, with the store's score function,
        and one without vectors returns no row. No threshold or filter
        applies: these are the target and the slot matches, found already.

        Returns:
            Tuple of (SQL, parameters). Each row has ``key`` and ``score``.
            Bind the query vector in place of QUERY_VECTOR.

        Raises:
            ValueError: If there is no index.
        """
        if self._index is None:
            raise ValueError("candidate_scores requires an index configuration")
        operator = self._index.distance.get_operator()
        best = self._score(f"min(embedding {operator} CAST(%s AS VECTOR))")
        sql = f"""SELECT key, {best} AS score
FROM {self.vectors}
WHERE prefix = %s AND key = ANY(CAST(%s AS STRING[]))
GROUP BY key"""
        return (sql, [QUERY_VECTOR, prefix, list(keys)])

    def namespace_vector_count(self, prefix: str, cap: int) -> tuple[str, list]:
        """Build the count of one namespace's vectors, up to ``cap``.

        It reads at most ``cap`` rows of the vectors primary key, so a count
        of ``cap`` means at least that many. Expired vectors count, and the
        namespace's descendants do not.

        Returns:
            Tuple of (SQL, parameters). The row has the count as ``n``.

        Raises:
            ValueError: If there is no index.
        """
        if self._index is None:
            raise ValueError("namespace_vector_count requires an index configuration")
        sql = f"""SELECT count(*) AS n FROM (
    SELECT 1 FROM {self.vectors} WHERE prefix = %s LIMIT %s
) AS c"""
        return (sql, [prefix, cap])

    def record(self, prefix: str, key: str) -> tuple[str, list]:
        """Build the read of one live memory with its version, provenance and confirmations.

        It never pushes back the expiry: reading a memory does not confirm it.

        Returns:
            Tuple of (SQL, parameters). The row, if any, has the candidate columns.
        """
        sql = f"""SELECT {self._candidate_columns()}
FROM {self.items}
WHERE prefix = %s AND key = %s AND (expires_at IS NULL OR expires_at > now())"""
        return (sql, [prefix, key])

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

    def archive(
        self,
        prefix: str,
        keys: Sequence[str],
        *,
        reason: str,
        superseded_by: str | None,
        invalid_at: datetime | None,
        write_id: str,
        ttl_days: float | None,
    ) -> tuple[str, list]:
        """Build the INSERT that copies live items into the history table.

        A write runs it first, in the write's transaction, before it
        overwrites or deletes the items. It copies each live item of ``keys``
        with its version, value, provenance, valid time (``updated_at`` when
        the row has none), confirmations and confirmed_at, and records when
        the store wrote it, when the version ended (now), why, and what
        replaced it. Rows that
        already carry ``write_id`` were written by this batch, in an attempt
        that committed before its result was lost, so they are not old
        versions and are left out. A version already in history is left as
        it is, so a retried attempt never adds a row twice.

        Args:
            prefix: Namespace text.
            keys: Item keys whose live versions end.
            reason: Why they end: "overwritten", "superseded" or "deleted".
            superseded_by: The key that replaces them, or None.
            invalid_at: When the facts stopped being true, or None for now.
                A version never ends before its own valid time: an earlier
                time is raised to it, as when a fact written with an older
                valid time overwrites a newer one.
            write_id: The write id of the batch that ends them.
            ttl_days: Days the versions stay in history, or None to keep them.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""INSERT INTO {self.history} (prefix, key, version, value, provenance, valid_at, invalid_at, written_at, ended_at, reason, superseded_by, confirmations, confirmed_at, write_id, expires_at)
SELECT prefix, key, version, value, provenance, COALESCE(valid_at, updated_at),
    GREATEST(COALESCE(CAST(%s AS TIMESTAMPTZ), now()), COALESCE(valid_at, updated_at)),
    updated_at, now(),
    CAST(%s AS STRING), CAST(%s AS STRING), confirmations, confirmed_at, CAST(%s AS UUID),
    now() + CAST(%s AS DECIMAL) * INTERVAL '1 day'
FROM {self.items}
WHERE prefix = %s AND key = ANY(%s)
    AND (expires_at IS NULL OR expires_at > now())
    AND write_id IS DISTINCT FROM CAST(%s AS UUID)
ON CONFLICT (prefix, key, version) DO NOTHING"""
        params: list[Any] = [
            invalid_at,
            reason,
            superseded_by,
            write_id,
            ttl_days,
            prefix,
            list(keys),
            write_id,
        ]
        return (sql, params)

    def history_insert_superseded(
        self,
        prefix: str,
        key: str,
        *,
        value_json: str,
        provenance_json: str | None,
        valid_at: datetime,
        invalid_at: datetime,
        superseded_by: str,
        write_id: str,
        ttl_days: float | None,
    ) -> tuple[str, list]:
        """Build the INSERT that records a new memory straight into history, as superseded.

        reconcile runs it when the policy decides that a candidate whose
        fact is newer already supersedes the new memory, so the new memory
        is never live. The row takes the key's next version, as the item
        upsert numbers them on a store with history, so it never takes a
        version the key has or gets later. It is valid from ``valid_at``
        until ``invalid_at``, never earlier, and the store held it for no
        time: written_at and ended_at are both now, and it has no
        confirmations. If a superseded row of
        the key already carries ``write_id``, this write inserted it in an
        attempt that committed before its result was lost, and nothing is
        inserted. The same write's archive of the memory under the key, as
        when it deletes that memory, has another reason and never counts,
        whatever order the statements run in.

        Args:
            prefix: Namespace text.
            key: The new memory's key.
            value_json: Its value, serialized.
            provenance_json: Its provenance, serialized, or None.
            valid_at: When its fact became true, as the policy saw it.
            invalid_at: When the newer fact became true.
            superseded_by: The key of the newer fact.
            write_id: The write id of the reconcile.
            ttl_days: Days the row stays in history, or None to keep it.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""INSERT INTO {self.history} (prefix, key, version, value, provenance, valid_at, invalid_at, written_at, ended_at, reason, superseded_by, write_id, expires_at)
SELECT p.prefix, p.key,
    {self._next_version("p")},
    p.value, p.provenance, p.valid_at, GREATEST(p.invalid_at, p.valid_at), now(), now(),
    p.reason, p.superseded_by, p.write_id, now() + p.ttl_days * INTERVAL '1 day'
FROM (
    SELECT CAST(%s AS STRING) AS prefix, CAST(%s AS STRING) AS key, CAST(%s AS JSONB) AS value,
        CAST(%s AS JSONB) AS provenance, CAST(%s AS TIMESTAMPTZ) AS valid_at,
        CAST(%s AS TIMESTAMPTZ) AS invalid_at, CAST(%s AS STRING) AS reason,
        CAST(%s AS STRING) AS superseded_by, CAST(%s AS UUID) AS write_id,
        CAST(%s AS DECIMAL) AS ttl_days
) AS p
WHERE NOT EXISTS (
    SELECT 1 FROM {self.history} AS h
    WHERE h.prefix = p.prefix AND h.key = p.key AND h.write_id = p.write_id
        AND h.reason = p.reason
)
ON CONFLICT (prefix, key, version) DO NOTHING"""
        params: list[Any] = [
            prefix,
            key,
            value_json,
            provenance_json,
            valid_at,
            invalid_at,
            "superseded",
            superseded_by,
            write_id,
            ttl_days,
        ]
        return (sql, params)

    def confirm(
        self,
        prefix: str,
        key: str,
        *,
        confirm_id: str,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> tuple[str, list]:
        """Build the UPDATE that confirms one live memory.

        It sets confirmed_at to now(), adds one to confirmations and stores
        ``confirm_id``, the id of the confirmation: confirm's own, or the
        write id of the reconcile or resolve that confirms. Without ``ttl``
        the memory's expiry is renewed from its own TTL, and a memory without
        a TTL keeps none. A number of minutes becomes its TTL, and None
        removes its expiry. Its vectors get the same expiry, in the same
        statement and only when the memory is confirmed: the CTE reads the
        item as it was before the update. The version, updated_at and the
        write id stay as they are, so confirming never invalidates a pending
        find_conflicts, and a write retried after a lost commit still finds
        its write id on the memory. A memory that already carries
        ``confirm_id`` was confirmed by this confirmation, in an attempt that
        committed before its result was lost, and is left as it is, with
        its vectors.

        Returns:
            Tuple of (SQL, parameters). The statement updates one row when it
            confirms the memory, and none when no live memory has the key or
            the memory already carries ``confirm_id``.
        """
        match = (
            "prefix = %s AND key = %s AND (expires_at IS NULL OR expires_at > now())\n"
            "{indent}AND confirm_id IS DISTINCT FROM CAST(%s AS UUID)"
        )
        match_params: list[Any] = [prefix, key, confirm_id]
        ttl_params: list[Any] = []
        if isinstance(ttl, NotProvided):
            item_expiry = (
                "expires_at = COALESCE(now() + ttl_minutes * INTERVAL '1 minute', expires_at)"
            )
            vector_expiry = "expires_at = now() + ttl_minutes * INTERVAL '1 minute'"
            vector_filter = " AND ttl_minutes IS NOT NULL"
        else:
            new_ttl = (
                "ttl_minutes = CAST(%s AS DECIMAL),\n"
                "{indent}expires_at = now() + CAST(%s AS DECIMAL) * INTERVAL '1 minute'"
            )
            item_expiry = new_ttl.format(indent="    ")
            vector_expiry = new_ttl.format(indent="        ")
            vector_filter = ""
            ttl_params = [ttl, ttl]
        sql = (
            f"UPDATE {self.items} SET confirmed_at = now(), confirmations = confirmations + 1,\n"
            f"    confirm_id = CAST(%s AS UUID),\n    {item_expiry}\n"
            f"WHERE {match.format(indent='    ')}"
        )
        params: list[Any] = [confirm_id, *ttl_params, *match_params]
        if self._index is None:
            return (sql, params)
        renewed = (
            "WITH renewed AS (\n"
            f"    UPDATE {self.vectors} SET {vector_expiry}\n"
            f"    WHERE prefix = %s AND key = %s{vector_filter} AND EXISTS (\n"
            f"        SELECT 1 FROM {self.items}\n"
            f"        WHERE {match.format(indent='            ')}\n"
            "    )\n"
            "    RETURNING key\n"
            ")\n"
        )
        return (renewed + sql, [*ttl_params, prefix, key, *match_params, *params])

    def confirm_check(self, prefix: str, key: str, *, confirm_id: str) -> tuple[str, list]:
        """Build the read of the live memory under ``key`` if it carries ``confirm_id``.

        confirm sends it with its UPDATE. An UPDATE that changed no row
        either found no live memory, or found one that an earlier attempt of
        the same confirm already confirmed, in an attempt that committed
        before its result was lost. Only then does this find the memory.

        Returns:
            Tuple of (SQL, parameters). The statement returns a row if the
            live memory carries ``confirm_id``, or no rows otherwise.
        """
        sql = (
            f"SELECT 1 FROM {self.items} WHERE prefix = %s AND key = %s "
            "AND (expires_at IS NULL OR expires_at > now()) "
            "AND confirm_id = CAST(%s AS UUID)"
        )
        return (sql, [prefix, key, confirm_id])

    def applied_items(self, prefix: str, keys: Sequence[str], *, write_id: str) -> tuple[str, list]:
        """Build the read of the items under ``keys`` that carry a call's id.

        A reconcile or resolve retried after a commit whose result was lost
        reads them, with applied_history, to tell what its earlier attempt
        did: the memory it wrote carries ``write_id`` as its write id, and the
        memories it confirmed carry it as their confirm id. put_if reads them
        for the memory it wrote. ``keys`` are the keys that the decisions of
        earlier attempts named, looked up by primary key, so the retry's
        transaction reads only their rows, not the whole namespace. Expired
        rows count, since they were written all the same.

        Returns:
            Tuple of (SQL, parameters). Each row has ``key``, ``version``,
            ``written``, whether it carries the id as its write id, and
            ``confirmed``, whether it carries it as its confirm id.
        """
        sql = (
            "SELECT key, version, write_id IS NOT DISTINCT FROM CAST(%s AS UUID) AS written,\n"
            "    confirm_id IS NOT DISTINCT FROM CAST(%s AS UUID) AS confirmed\n"
            f"FROM {self.items}\n"
            "WHERE prefix = %s AND key = ANY(CAST(%s AS STRING[]))\n"
            "    AND (write_id = CAST(%s AS UUID) OR confirm_id = CAST(%s AS UUID))"
        )
        return (sql, [write_id, write_id, prefix, list(keys), write_id, write_id])

    def applied_history(
        self, prefix: str, keys: Sequence[str], *, write_id: str
    ) -> tuple[str, list]:
        """Build the read of the history rows under ``keys`` that carry a write id.

        They are the versions a reconcile archived, and a new memory it
        recorded straight into history. History's primary key starts with
        (prefix, key), so only the versions of ``keys`` are read. See
        applied_items.

        Returns:
            Tuple of (SQL, parameters). Each row has ``key`` and ``reason``.
        """
        sql = (
            f"SELECT key, reason FROM {self.history}\n"
            "WHERE prefix = %s AND key = ANY(CAST(%s AS STRING[])) AND write_id = CAST(%s AS UUID)"
        )
        return (sql, [prefix, list(keys), write_id])

    def tokens(self, prefix: str, keys: Sequence[str]) -> tuple[str, list]:
        """Build the read of the version and updated_at of the live items under ``keys``.

        resolve() compares them with what find_conflicts read, to tell
        whether a memory changed in between (base.changed_keys). A put
        bumps the version, and a 0.5.x store, which does not, still sets
        updated_at, so both are read. An expired item is not read: it counts
        as gone. The keys are looked up by primary key, and the read pushes
        back no expiry.

        Returns:
            Tuple of (SQL, parameters). Each row has ``key``, ``version`` and
            ``updated_at``.
        """
        sql = (
            f"SELECT key, version, updated_at FROM {self.items}\n"
            "WHERE prefix = %s AND key = ANY(CAST(%s AS STRING[]))\n"
            "    AND (expires_at IS NULL OR expires_at > now())"
        )
        return (sql, [prefix, list(keys)])

    def history_versions(self, prefix: str, key: str, *, limit: int) -> tuple[str, list]:
        """Build the SELECT for the versions of one key in history, newest first.

        Versions past their history TTL are left out, though CockroachDB's
        TTL job may not have deleted them yet.

        Returns:
            Tuple of (SQL, parameters).
        """
        sql = f"""SELECT version, value, provenance, valid_at, invalid_at, written_at, ended_at, reason, superseded_by, confirmed_at, confirmations
FROM {self.history}
WHERE prefix = %s AND key = %s AND (expires_at IS NULL OR expires_at > now())
ORDER BY version DESC
LIMIT %s"""
        return (sql, [prefix, key, limit])

    def purge(self, prefix: str, key: str) -> tuple[str, list]:
        """Build one statement that deletes an item, its vectors and its history.

        Each table's rows go in a data-modifying CTE of the same statement,
        so they are deleted together.

        Returns:
            Tuple of (SQL, parameters). The statement returns one row whose
            ``removed`` says whether any row was deleted.
        """
        where = "WHERE prefix = %s AND key = %s RETURNING 1"
        tables = [("deleted_items", self.items)]
        if self._index is not None:
            tables.append(("deleted_vectors", self.vectors))
        tables.append(("deleted_history", self.history))
        ctes = [f"{name} AS (\n    DELETE FROM {table} {where}\n)" for name, table in tables]
        found = "\n    OR ".join(f"EXISTS (SELECT 1 FROM {name})" for name, _ in tables)
        sql = "WITH " + ",\n".join(ctes) + f"\nSELECT {found} AS removed"
        return (sql, [prefix, key] * len(tables))

    def delete_namespace(
        self, namespace_prefix: Sequence[str], *, history: bool = False
    ) -> tuple[str, list]:
        """Build one DELETE for a namespace and its descendants.

        With an index, the vectors go in a data-modifying CTE of the same
        statement, and with ``history`` so do the history rows.

        Args:
            namespace_prefix: Namespace prefix to delete.
            history: Whether to delete the namespace's history rows too.

        Returns:
            Tuple of (SQL, parameters). The statement returns the key of
            every item deleted.
        """
        ns_condition, ns_params = prefix_condition("prefix", namespace_prefix)
        sql = f"DELETE FROM {self.items}\nWHERE {ns_condition}\nRETURNING key"
        tables = []
        if self._index is not None:
            tables.append(("deleted_vectors", self.vectors))
        if history:
            tables.append(("deleted_history", self.history))
        if not tables:
            return (sql, list(ns_params))
        ctes = [
            f"{name} AS (\n    DELETE FROM {table}\n    WHERE {ns_condition}\n    RETURNING key\n)"
            for name, table in tables
        ]
        return ("WITH " + ",\n".join(ctes) + "\n" + sql, list(ns_params) * (len(tables) + 1))

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
