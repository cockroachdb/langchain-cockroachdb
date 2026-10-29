"""Tests for store SQL statement building."""

import re
from datetime import datetime, timezone

import pytest

from langchain_cockroachdb.indexes import DistanceStrategy
from langchain_cockroachdb.store.config import ResolvedIndex
from langchain_cockroachdb.store.sql import (
    MAX_SEARCH_NAMESPACES,
    MIGRATION_RECONCILE,
    NAMESPACES,
    QUERY_VECTOR,
    StoreSQL,
    bind,
)


def _make_index(
    *,
    dims: int = 128,
    distance: DistanceStrategy = DistanceStrategy.COSINE,
    cspann: dict | None = None,
    flat: bool = False,
) -> ResolvedIndex:
    """Helper to create a minimal ResolvedIndex."""
    from unittest.mock import Mock

    return ResolvedIndex(
        dims=dims,
        embeddings=Mock(),
        fields=(("$", "$"),),
        estimated_vectors=1,
        distance=distance,
        cspann=cspann or {},
        flat=flat,
        beam_size=None,
    )


def candidate_statements(sql: StoreSQL) -> list[tuple[str, list]]:
    """Every statement that finds candidates or reads a record, in each of its forms."""
    return [
        sql.now(),
        sql.key_slot_candidates("a.b", "k1", slot_field=None, slot_value_json=None, filter=None),
        sql.key_slot_candidates(
            "a.b", "k1", slot_field="seat", slot_value_json='"x"', filter={"x": 1, "y": {"$gt": 2}}
        ),
        *(
            sql.similar_candidates(
                "a.b",
                threshold=0.8,
                limit=3,
                filter={"x": 1},
                scan=scan,  # type: ignore[arg-type]
                over_fetch=7,
                exclude=["k1", "k2"],
            )
            for scan in ("exact", "index")
        ),
        sql.candidate_scores("a.b", ["k1", "k2"]),
        sql.namespace_vector_count("a.b", 2001),
        sql.record("a.b", "k1"),
    ]


def test_identifiers_are_validated():
    """StoreSQL validates identifiers and raises ValueError for invalid ones."""
    with pytest.raises(ValueError, match="Invalid"):
        StoreSQL("public", "store; DROP TABLE x")

    with pytest.raises(ValueError, match="Invalid"):
        StoreSQL("public; --", "store")


def test_vector_statements_are_separate_and_idempotent():
    """migrations() returns only 3 statements; vector_statements() is separate."""
    # Without index: migrations has 3 statements, vector_statements is empty
    sql_no_index = StoreSQL("public", "store", index=None)
    assert len(sql_no_index.migrations()) == 3
    assert len(sql_no_index.vector_statements()) == 0

    # With index (not flat): migrations still has 3, vector_statements has 2
    sql_with_index = StoreSQL("public", "store", index=_make_index())
    assert len(sql_with_index.migrations()) == 3
    assert len(sql_with_index.vector_statements()) == 2

    # With flat index: migrations has 3, vector_statements has 1 (no index statement)
    sql_flat = StoreSQL("public", "store", index=_make_index(flat=True))
    assert len(sql_flat.migrations()) == 3
    assert len(sql_flat.vector_statements()) == 1

    # All vector statements must have IF NOT EXISTS
    for stmt in sql_with_index.vector_statements():
        assert "IF NOT EXISTS" in stmt


def test_vector_index_has_prefix_column_and_options():
    """Vector index includes prefix column and C-SPANN options when configured."""
    # With C-SPANN options
    sql_cspann = StoreSQL(
        "public",
        "store",
        index=_make_index(
            distance=DistanceStrategy.COSINE,
            cspann={"min_partition_size": 16, "max_partition_size": 128},
        ),
    )
    stmts = sql_cspann.vector_statements()
    # Find the index statement (second one)
    index_stmt = stmts[1]
    assert "(prefix, embedding vector_cosine_ops)" in index_stmt
    assert "WITH (min_partition_size=16, max_partition_size=128)" in index_stmt

    # Without C-SPANN (flat)
    sql_flat = StoreSQL("public", "store", index=_make_index(flat=True))
    stmts = sql_flat.vector_statements()
    # Only table statement, no index
    assert len(stmts) == 1
    assert "CREATE TABLE" in stmts[0]


def test_search_sql_rejects_negative_paging():
    """search and semantic_search raise ValueError for negative paging."""
    sql = StoreSQL("public", "store", index=None)

    with pytest.raises(ValueError, match="limit"):
        sql.search(
            namespace_prefix=["a"],
            filter=None,
            limit=-1,
            offset=0,
            omit_expired=True,
        )

    with pytest.raises(ValueError, match="offset"):
        sql.search(
            namespace_prefix=["a"],
            filter=None,
            limit=10,
            offset=-1,
            omit_expired=True,
        )


def test_semantic_search_param_order():
    """semantic_search parameters match placeholder order in SQL text."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.semantic_search(
        namespace_prefix=["a", "b"],
        filter={"x": 1},
        limit=5,
        offset=10,
        omit_expired=True,
    )

    # Order: QUERY_VECTOR, namespace params, filter params, limit, offset
    assert params[0] is QUERY_VECTOR
    # Namespace params (exact match "a.b", then the range "a.b." to "a.b/")
    assert params[1:4] == ["a.b", "a.b.", "a.b/"]
    # Filter params (key "x", value "1")
    assert params[4:6] == ["x", "1"]
    # limit, offset
    assert params[6:] == [5, 10]
    assert query_sql.count("%s") == len(params)
    assert "(v.prefix = %s OR (v.prefix >= %s AND v.prefix < %s))" in query_sql


def test_no_cast_shorthand():
    """No SQL statement uses :: cast shorthand."""
    sql = StoreSQL("public", "store", index=_make_index())

    # Check all migration and vector statements
    for stmt in sql.migrations():
        assert "::" not in stmt

    for stmt in sql.vector_statements():
        assert "::" not in stmt

    # Check various query methods
    query_sql, _ = sql.get(["a"], ["k1"], omit_expired=True)
    assert "::" not in query_sql

    query_sql, _ = sql.search(
        namespace_prefix=["a"],
        filter=None,
        limit=10,
        offset=0,
        omit_expired=True,
    )
    assert "::" not in query_sql

    query_sql, _ = sql.semantic_search(
        namespace_prefix=["a"],
        filter=None,
        limit=10,
        offset=0,
        omit_expired=True,
    )
    assert "::" not in query_sql

    query_sql, _ = sql.semantic_search_in(
        filter={"x": 1}, limit=10, offset=0, omit_expired=True, over_fetch=21
    )
    assert "::" not in query_sql

    query_sql, _ = sql.exact_semantic_search_in(
        filter={"x": 1}, limit=10, offset=0, omit_expired=True
    )
    assert "::" not in query_sql

    for stmt in sql.history_statements(30.0):
        assert "::" not in stmt
    for query_sql, _ in (
        sql.archive(
            "a",
            ["k1"],
            reason="deleted",
            superseded_by=None,
            invalid_at=None,
            write_id="w",
            ttl_days=1.0,
        ),
        sql.history_versions("a", "k1", limit=1),
        sql.history_exists(),
        sql.reconcile_columns_exist(),
        sql.purge("a", "k1"),
        sql.delete_namespace(["a"], history=True),
        sql.upsert_items([("a", "k1", "{}", None, None, None)], write_id="w", history=True),
        *candidate_statements(sql),
    ):
        assert "::" not in query_sql


def test_get():
    """get() builds SELECT with prefix and key list."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.get("a.b", ["k1", "k2"], omit_expired=True)

    assert "SELECT" in query_sql
    assert "FROM public.store" in query_sql
    assert "prefix = %s" in query_sql
    assert "key = ANY(%s)" in query_sql
    assert "(expires_at IS NULL OR expires_at > now())" in query_sql

    assert params == ["a.b", ["k1", "k2"]]


def test_get_with_refresh_is_one_statement():
    """get(refresh=True) updates expiry in CTEs and reads in the same statement."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.get("a", ["k1"], omit_expired=True, refresh=True)

    assert query_sql.startswith("WITH refreshed AS (")
    assert "UPDATE public.store" in query_sql
    assert "expires_at = now() + ttl_minutes * INTERVAL '1 minute'" in query_sql
    assert "ttl_minutes IS NOT NULL" in query_sql
    assert "refreshed_vectors" not in query_sql
    assert query_sql.rstrip().endswith("AND (expires_at IS NULL OR expires_at > now())")
    assert query_sql.count("%s") == len(params)
    assert params == ["a", ["k1"], "a", ["k1"]]

    # With an index the vectors get a CTE of their own
    indexed = StoreSQL("public", "store", index=_make_index())
    query_sql, params = indexed.get("a", ["k1"], omit_expired=False, refresh=True)
    assert "refreshed_vectors AS (" in query_sql
    assert "UPDATE public.store_vectors" in query_sql
    assert "expires_at > now()" not in query_sql
    assert params == ["a", ["k1"]] * 3


def test_search_with_refresh_wraps_results():
    """search(refresh=True) keeps its params and updates the rows it returns."""
    sql = StoreSQL("public", "store", index=_make_index())
    plain_sql, plain_params = sql.search(
        namespace_prefix=["a"], filter={"x": 1}, limit=10, offset=5, omit_expired=True
    )
    query_sql, params = sql.search(
        namespace_prefix=["a"],
        filter={"x": 1},
        limit=10,
        offset=5,
        omit_expired=True,
        refresh=True,
    )
    assert params == plain_params
    assert query_sql.startswith("WITH search_results AS (\n" + plain_sql + "\n)")
    assert "updated AS (" in query_sql
    assert "updated_vectors AS (" in query_sql
    assert "(prefix, key) IN (SELECT prefix, key FROM search_results)" in query_sql
    assert query_sql.endswith("FROM search_results\nORDER BY updated_at DESC, prefix, key")

    semantic_sql, semantic_params = sql.semantic_search(
        namespace_prefix=["a"],
        filter=None,
        limit=3,
        offset=0,
        omit_expired=True,
        refresh=True,
    )
    assert semantic_params[0] is QUERY_VECTOR
    assert semantic_params[-2:] == [3, 0]
    assert semantic_sql.count("%s") == len(semantic_params)
    assert semantic_sql.endswith("score\nFROM search_results\nORDER BY score DESC, prefix, key")


def test_upsert_items():
    """upsert_items() builds multi-row INSERT with ON CONFLICT."""
    sql = StoreSQL("public", "store", index=None)
    rows = [
        ("a", "k1", '{"x": 1}', 60.0),
        ("a", "k2", '{"x": 2}', None),
    ]
    query_sql, params = sql.upsert_items(rows)

    assert "INSERT INTO public.store" in query_sql
    assert "ON CONFLICT (prefix, key) DO UPDATE" in query_sql
    assert "value = EXCLUDED.value" in query_sql
    # created_at keeps first value, not updated
    assert "created_at = EXCLUDED.created_at" not in query_sql

    # Check params: prefix, key, value, ttl for each row
    assert params[:4] == ["a", "k1", '{"x": 1}', 60.0]
    assert params[4:8] == ["a", "k2", '{"x": 2}', None]


def test_migration_2_adds_the_reconcile_columns_in_one_statement() -> None:
    statements = StoreSQL("public", "store").migrations()
    assert len(statements) == 3
    migration = statements[MIGRATION_RECONCILE]
    assert migration.startswith("ALTER TABLE public.store ")
    for column in (
        "version INT8 NOT NULL DEFAULT 1",
        "provenance JSONB",
        "confirmed_at TIMESTAMPTZ",
        "confirmations INT8 NOT NULL DEFAULT 0",
        "write_id UUID",
        "valid_at TIMESTAMPTZ",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in migration


def test_the_column_check_binds_the_items_table_and_migration_2_columns() -> None:
    sql = StoreSQL("memory", "facts")
    query, params = sql.reconcile_columns_exist()
    assert query == (
        "SELECT count(*) = %s AS ready FROM pg_catalog.pg_attribute\n"
        "WHERE attrelid = to_regclass(%s) AND NOT attisdropped AND attname = ANY(%s)"
    )
    columns = ["version", "provenance", "confirmed_at", "confirmations", "write_id", "valid_at"]
    assert params == [6, "memory.facts", columns]
    for value in ("memory", "facts", *columns):
        assert value not in query
    # The check requires exactly the columns migration 2 adds
    migration = sql.migrations()[MIGRATION_RECONCILE]
    assert re.findall(r"ADD COLUMN IF NOT EXISTS (\w+) ", migration) == columns


def test_upsert_with_a_write_id_bumps_the_version_only_for_a_new_write() -> None:
    sql, params = StoreSQL("public", "store").upsert_items(
        [("user.a", "k", '{"text": "x"}', None, '{"source": "user"}', None)], write_id="w-1"
    )
    assert "version = public.store.version + 1" in sql
    assert "WHERE public.store.write_id IS DISTINCT FROM EXCLUDED.write_id" in sql
    assert "w-1" in params and '{"source": "user"}' in params
    assert "w-1" not in sql and "user" not in sql


def test_upsert_with_a_write_id_binds_every_value() -> None:
    when = datetime(2026, 1, 2, tzinfo=timezone.utc)
    sql, params = StoreSQL("public", "store").upsert_items(
        [
            ("a", "k1", '{"x": 1}', 60.0, '{"source": "chat"}', when),
            ("a", "k2", '{"x": 2}', None, None, None),
        ],
        write_id="w-2",
    )
    # The write id's placeholder is in the outer SELECT, before the rows'
    assert params == [
        "w-2",
        *("a", "k1", '{"x": 1}', 60.0, '{"source": "chat"}', when),
        *("a", "k2", '{"x": 2}', None, None, None),
    ]
    assert sql.count("%s") == len(params)
    assert "CAST(%s AS UUID)" in sql
    assert "CAST(%s AS JSONB) AS provenance, CAST(%s AS TIMESTAMPTZ) AS valid_at" in sql
    assert "::" not in sql

    builder = StoreSQL("public", "store")
    with pytest.raises(ValueError, match="6 fields with a write id, got 4"):
        builder.upsert_items([("a", "k", "{}", None)], write_id="w-3")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="4 fields without a write id, got 6"):
        builder.upsert_items([("a", "k", "{}", None, None, None)])


def test_upsert_without_a_write_id_is_the_0_5_statement() -> None:
    sql, _ = StoreSQL("public", "store").upsert_items([("user.a", "k", "{}", None)])
    assert "version" not in sql and "write_id" not in sql and "provenance" not in sql


def test_upsert_without_a_write_id_keeps_the_0_5_1_text() -> None:
    """Until setup() applies migration 2, puts send the 0.5.1 statement byte for byte."""
    sql, params = StoreSQL("public", "store").upsert_items(
        [("user.a", "k", "{}", None), ("user.a", "j", "{}", 5.0)]
    )
    assert sql == (
        "INSERT INTO public.store (prefix, key, value, expires_at, ttl_minutes)\n"
        "SELECT prefix, key, value, now() + CAST(ttl_minutes AS DECIMAL) * INTERVAL '1 minute', ttl_minutes\n"
        "FROM (SELECT CAST(%s AS STRING) AS prefix, CAST(%s AS STRING) AS key, CAST(%s AS JSONB) AS value, CAST(%s AS DECIMAL) AS ttl_minutes\n"
        "    UNION ALL SELECT CAST(%s AS STRING), CAST(%s AS STRING), CAST(%s AS JSONB), CAST(%s AS DECIMAL)) AS rows_to_insert\n"
        "ON CONFLICT (prefix, key) DO UPDATE SET\n"
        "    value = EXCLUDED.value,\n"
        "    updated_at = now(),\n"
        "    expires_at = EXCLUDED.expires_at,\n"
        "    ttl_minutes = EXCLUDED.ttl_minutes"
    )
    assert params == ["user.a", "k", "{}", None, "user.a", "j", "{}", 5.0]


def test_delete_items():
    """delete_items() builds DELETE with prefix and key list."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.delete_items("a", ["k1", "k2"])

    assert "DELETE FROM public.store" in query_sql
    assert "prefix = %s" in query_sql
    assert "key = ANY(%s)" in query_sql

    assert params == ["a", ["k1", "k2"]]


def test_delete_vectors():
    """delete_vectors() mirrors delete_items for vectors table."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.delete_vectors("a", ["k1"])

    assert "DELETE FROM public.store_vectors" in query_sql

    assert params == ["a", ["k1"]]


def test_upsert_vectors():
    """upsert_vectors() builds multi-row INSERT with vector literals."""
    sql = StoreSQL("public", "store", index=_make_index())
    rows = [
        ("a", "k1", "field1", [0.1, 0.2], 60.0),
        ("a", "k1", "field2", [0.3, 0.4], 60.0),
    ]
    query_sql, params = sql.upsert_vectors(rows)

    assert "INSERT INTO public.store_vectors" in query_sql
    assert "ON CONFLICT (prefix, key, field_name) DO UPDATE" in query_sql
    assert "CAST(%s AS VECTOR)" in query_sql

    # First row params: prefix, key, field_name, vector literal, ttl
    assert params[0] == "a"
    assert params[1] == "k1"
    assert params[2] == "field1"
    assert params[3] == "[0.1,0.2]"  # vector_literal format
    assert params[4] == 60.0


def test_search():
    """search() builds ORDER BY updated_at DESC with paging."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.search(
        namespace_prefix=["a"],
        filter={"x": 1},
        limit=10,
        offset=5,
        omit_expired=True,
    )

    assert "FROM public.store" in query_sql
    assert "ORDER BY updated_at DESC, prefix, key" in query_sql
    assert "LIMIT %s OFFSET %s" in query_sql
    assert "(expires_at IS NULL OR expires_at > now())" in query_sql

    # namespace params, filter params, limit, offset
    assert params[-2:] == [10, 5]


def test_list_namespaces():
    """list_namespaces() builds DISTINCT query with truncation."""
    sql = StoreSQL("public", "store", index=None)

    # With max_depth
    query_sql, params = sql.list_namespaces(
        match_conditions=[],
        max_depth=2,
        limit=10,
        offset=0,
        omit_expired=True,
    )

    assert "SELECT DISTINCT" in query_sql
    assert "regexp_replace" in query_sql  # truncation expression
    assert "ORDER BY" in query_sql
    assert "LIMIT %s OFFSET %s" in query_sql

    # Without max_depth
    query_sql, params = sql.list_namespaces(
        match_conditions=[],
        max_depth=None,
        limit=10,
        offset=0,
        omit_expired=True,
    )

    assert "SELECT DISTINCT" in query_sql
    assert "regexp_replace" not in query_sql  # no truncation


def test_list_namespaces_bounds_prefixes_with_a_range():
    """Each literal prefix adds the range next to the regexes; empty ones add nothing."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.list_namespaces(
        match_conditions=[r"^user\.alice(\.|\z)", r"(^|\.)notes\z"],
        max_depth=None,
        limit=10,
        offset=2,
        omit_expired=True,
        prefixes=[("user", "alice"), ()],
    )
    assert query_sql == (
        "SELECT DISTINCT prefix AS ns\nFROM public.store\n"
        "WHERE (expires_at IS NULL OR expires_at > now())"
        " AND (prefix = %s OR (prefix >= %s AND prefix < %s))"
        " AND prefix ~ %s AND prefix ~ %s"
        "\nORDER BY ns\nLIMIT %s OFFSET %s"
    )
    assert params == [
        "user.alice",
        "user.alice.",
        "user.alice/",
        r"^user\.alice(\.|\z)",
        r"(^|\.)notes\z",
        10,
        2,
    ]

    # Nothing to filter on
    query_sql, params = sql.list_namespaces(
        match_conditions=[], max_depth=None, limit=10, offset=0, omit_expired=False
    )
    assert "WHERE TRUE\n" in query_sql
    assert params == [10, 0]


def test_delete_namespace():
    """delete_namespace() is one statement; vectors go in a CTE with an index."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.delete_namespace(["a", "b"])

    assert query_sql.startswith("WITH deleted_vectors AS (")
    assert "DELETE FROM public.store_vectors" in query_sql
    assert "DELETE FROM public.store\n" in query_sql
    assert "(prefix = %s OR (prefix >= %s AND prefix < %s))" in query_sql
    assert query_sql.endswith("RETURNING key")
    assert params == ["a.b", "a.b.", "a.b/"] * 2

    # Without index: only items delete
    sql_no_index = StoreSQL("public", "store", index=None)
    query_sql, params = sql_no_index.delete_namespace(["a"])
    assert query_sql.startswith("DELETE FROM public.store\n")
    assert "store_vectors" not in query_sql
    assert params == ["a", "a.", "a/"]


def test_sweep():
    """sweep() returns DELETE statements for expired rows."""
    # With index: vectors, then items
    sql = StoreSQL("public", "store", index=_make_index())
    sweeps = sql.sweep()

    assert len(sweeps) == 2

    vectors_sql, vectors_params = sweeps[0]
    assert "DELETE FROM public.store_vectors" in vectors_sql
    assert "expires_at IS NOT NULL AND expires_at < now()" in vectors_sql
    assert vectors_params == []

    items_sql, items_params = sweeps[1]
    assert "DELETE FROM public.store" in items_sql
    assert "expires_at IS NOT NULL AND expires_at < now()" in items_sql
    assert items_params == []

    # Without index: only items
    sql_no_index = StoreSQL("public", "store", index=None)
    sweeps = sql_no_index.sweep()

    assert len(sweeps) == 1


def test_ttl_statements():
    """ttl_statements() builds ALTER TABLE SET for TTL configuration."""
    sql = StoreSQL("public", "store", index=_make_index())
    stmts = sql.ttl_statements("@hourly")

    # With index: items and vectors
    assert len(stmts) == 2
    for stmt in stmts:
        assert "ALTER TABLE" in stmt
        assert "SET (ttl_expiration_expression" in stmt
        assert "@hourly" in stmt


def test_disable_ttl_statements():
    """disable_ttl_statements() builds ALTER TABLE RESET."""
    sql = StoreSQL("public", "store", index=_make_index())
    stmts = sql.disable_ttl_statements()

    assert len(stmts) == 2
    for stmt in stmts:
        assert "ALTER TABLE" in stmt
        assert "RESET (ttl)" in stmt


def test_upsert_empty_rows_raises():
    """Empty rows should raise ValueError."""
    sql = StoreSQL("public", "store", index=None)

    with pytest.raises(ValueError, match="rows cannot be empty"):
        sql.upsert_items([])

    sql_with_index = StoreSQL("public", "store", index=_make_index())
    with pytest.raises(ValueError, match="rows cannot be empty"):
        sql_with_index.upsert_vectors([])


def test_ttl_cron_validation():
    """ttl_statements should reject cron with single quote."""
    sql = StoreSQL("public", "store", index=None)

    with pytest.raises(ValueError, match="single quote"):
        sql.ttl_statements("0 0 * * *'; DROP TABLE")


def test_every_builder_has_one_param_per_placeholder():
    """Every %s in SQL must have exactly one parameter."""
    # Use a mixed case: namespace with 3 labels, filter with two keys, index configured
    sql = StoreSQL("public", "store", index=_make_index())

    # get
    query_sql, params = sql.get("a.b.c", ["k1"], omit_expired=True)
    assert query_sql.count("%s") == len(params)

    # get with refresh
    query_sql, params = sql.get("a.b.c", ["k1", "k2"], omit_expired=True, refresh=True)
    assert query_sql.count("%s") == len(params)

    # upsert_items
    rows = [("a.b.c", "k1", '{"x": 1}', 60.0), ("a.b.c", "k2", '{"y": 2}', None)]
    query_sql, params = sql.upsert_items(rows)
    assert query_sql.count("%s") == len(params)

    # delete_items
    query_sql, params = sql.delete_items("a.b.c", ["k1", "k2"])
    assert query_sql.count("%s") == len(params)

    # delete_vectors
    query_sql, params = sql.delete_vectors("a.b.c", ["k1"])
    assert query_sql.count("%s") == len(params)

    # upsert_vectors
    vec_rows = [("a.b.c", "k1", "f1", [0.1, 0.2], 60.0)]
    query_sql, params = sql.upsert_vectors(vec_rows)
    assert query_sql.count("%s") == len(params)

    # search
    query_sql, params = sql.search(
        namespace_prefix=["a", "b", "c"],
        filter={"x": 1, "y": 2},
        limit=10,
        offset=5,
        omit_expired=True,
    )
    assert query_sql.count("%s") == len(params)

    # semantic searches, exact and indexed
    query_sql, params = sql.semantic_search(
        namespace_prefix=["a", "b", "c"],
        filter={"x": 1, "y": 2},
        limit=10,
        offset=5,
        omit_expired=True,
    )
    assert query_sql.count("%s") == len(params)
    for refresh in (False, True):
        query_sql, params = sql.semantic_search_in(
            filter={"x": 1, "y": 2},
            limit=10,
            offset=5,
            omit_expired=True,
            over_fetch=100,
            refresh=refresh,
        )
        assert query_sql.count("%s") == len(params)
        query_sql, params = sql.exact_semantic_search_in(
            filter={"x": 1, "y": 2}, limit=10, offset=5, omit_expired=True, refresh=refresh
        )
        assert query_sql.count("%s") == len(params)

    # list_namespaces
    query_sql, params = sql.list_namespaces(
        match_conditions=["^a", "^b"],
        max_depth=2,
        limit=10,
        offset=0,
        omit_expired=True,
    )
    assert query_sql.count("%s") == len(params)

    # delete_namespace
    query_sql, params = sql.delete_namespace(["a", "b", "c"])
    assert query_sql.count("%s") == len(params)
    query_sql, params = sql.delete_namespace(["a", "b", "c"], history=True)
    assert query_sql.count("%s") == len(params)

    # history
    query_sql, params = sql.archive(
        "a.b.c",
        ["k1", "k2"],
        reason="deleted",
        superseded_by=None,
        invalid_at=None,
        write_id="w",
        ttl_days=None,
    )
    assert query_sql.count("%s") == len(params)
    for query_sql, params in (
        sql.history_versions("a.b.c", "k1", limit=3),
        sql.history_exists(),
        sql.reconcile_columns_exist(),
        sql.purge("a.b.c", "k1"),
        sql.upsert_items([("a.b.c", "k1", "{}", None, None, None)], write_id="w", history=True),
        *candidate_statements(sql),
    ):
        assert query_sql.count("%s") == len(params)

    # sweep
    sweeps = sql.sweep()
    for query_sql, params in sweeps:
        assert query_sql.count("%s") == len(params)


def test_vector_index_name_is_public():
    """vector_index_name is exposed as a public attribute."""
    sql = StoreSQL("public", "my_store", index=_make_index())
    assert sql.vector_index_name == "my_store_vectors_embedding_idx"


def test_indexed_window_carries_no_filter_or_expiry():
    """The indexed form's window ranks with no filter or expiry check, so C-SPANN can serve it.

    The items join applies both, and every row reports the window size,
    counted over the rows the inner LIMIT kept.
    """
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.semantic_search_in(
        filter={"x": 1}, limit=3, offset=0, omit_expired=True, over_fetch=7
    )
    ranked, rest = query_sql.split("\n),\nbest AS (", 1)
    assert "expires_at" not in ranked and "value" not in ranked
    assert "count(*) OVER () AS window_size" in ranked
    # The count wraps the inner LIMIT, so it counts the window, not the namespaces
    assert ranked.index("count(*) OVER ()") < ranked.index("LIMIT %s")
    assert "(i.expires_at IS NULL OR i.expires_at > now())" in rest
    assert "i.value" in rest
    assert rest.rstrip().endswith("LIMIT %s OFFSET %s")
    assert "SELECT prefix, key, value, created_at, updated_at, score, window_size, (" in rest
    # The namespaces' vectors, counted from the primary key up to over_fetch,
    # with the same namespaces and bound as the window
    assert (
        "    SELECT count(*) FROM (\n"
        "        SELECT 1 FROM public.store_vectors WHERE prefix = ANY(CAST(%s AS STRING[])) LIMIT %s\n"
        "    ) AS c\n"
        ") AS namespace_vectors"
    ) in rest
    assert params == [QUERY_VECTOR, NAMESPACES, QUERY_VECTOR, 7, "x", "1", NAMESPACES, 7, 3, 0]
    assert query_sql.count("%s") == len(params)


@pytest.mark.parametrize("form", ["semantic_search", "exact_semantic_search_in"])
def test_exact_forms_filter_before_they_rank(form: str):
    """The exact forms join, filter and check expiry in the ranked CTE, with no LIMIT there.

    Only the outer query cuts, so every match is ranked and no vector index
    can serve the statement.
    """
    sql = StoreSQL("public", "store", index=_make_index())
    kwargs = {"filter": {"x": 1}, "limit": 3, "offset": 2, "omit_expired": True}
    if form == "semantic_search":
        query_sql, params = sql.semantic_search(namespace_prefix=["a"], **kwargs)  # type: ignore[arg-type]
        assert params == [QUERY_VECTOR, "a", "a.", "a/", "x", "1", 3, 2]
    else:
        query_sql, params = sql.exact_semantic_search_in(**kwargs)  # type: ignore[arg-type]
        assert "v.prefix = ANY(CAST(%s AS STRING[]))" in query_sql
        assert params == [QUERY_VECTOR, NAMESPACES, "x", "1", 3, 2]
    ranked, outer = query_sql.split("\n)\nSELECT ", 1)
    assert ranked.startswith("WITH ranked AS (")
    assert "JOIN public.store AS i ON v.prefix = i.prefix AND v.key = i.key" in ranked
    assert "(i.expires_at IS NULL OR i.expires_at > now())" in ranked
    assert "i.value" in ranked
    assert "min(v.embedding <=> CAST(%s AS VECTOR)) AS distance" in ranked
    assert "GROUP BY i.prefix, i.key" in ranked
    assert "LIMIT" not in ranked and "ORDER BY" not in ranked
    assert outer.endswith("ORDER BY score DESC, prefix, key\nLIMIT %s OFFSET %s")
    assert "1 - distance AS score" in outer
    assert query_sql.count("%s") == len(params)

    no_expiry, _ = sql.exact_semantic_search_in(filter=None, limit=3, offset=0, omit_expired=False)
    assert "expires_at" not in no_expiry

    l2 = StoreSQL("public", "store", index=_make_index(distance=DistanceStrategy.EUCLIDEAN))
    l2_sql, _ = l2.semantic_search(
        namespace_prefix=[], filter=None, limit=3, offset=0, omit_expired=False
    )
    assert "min(v.embedding <-> CAST(%s AS VECTOR))" in l2_sql
    assert "-distance AS score" in l2_sql
    assert "WHERE TRUE" in l2_sql

    with pytest.raises(ValueError, match="index"):
        StoreSQL("public", "store").exact_semantic_search_in(
            filter=None, limit=1, offset=0, omit_expired=True
        )
    with pytest.raises(ValueError, match="limit"):
        sql.exact_semantic_search_in(filter=None, limit=-1, offset=0, omit_expired=True)


def test_semantic_search_in_ranks_within_found_namespaces():
    """semantic_search_in matches the namespaces from namespace_lookup with ANY."""
    sql = StoreSQL("public", "store", index=_make_index(distance=DistanceStrategy.EUCLIDEAN))
    query_sql, params = sql.semantic_search_in(
        filter={"x": 1}, limit=5, offset=2, omit_expired=False, over_fetch=11
    )
    assert "WHERE prefix = ANY(CAST(%s AS STRING[]))" in query_sql
    assert "LIKE" not in query_sql
    assert "-r.distance AS score" in query_sql
    assert "embedding <-> CAST(%s AS VECTOR)" in query_sql
    assert params == [QUERY_VECTOR, NAMESPACES, QUERY_VECTOR, 11, "x", "1", NAMESPACES, 11, 5, 2]
    assert query_sql.count("%s") == len(params)
    assert "::" not in query_sql

    refreshed_sql, refreshed_params = sql.semantic_search_in(
        filter=None, limit=5, offset=0, omit_expired=True, over_fetch=11, refresh=True
    )
    assert refreshed_params == [QUERY_VECTOR, NAMESPACES, QUERY_VECTOR, 11, NAMESPACES, 11, 5, 0]
    assert refreshed_sql.startswith("WITH search_results AS (")
    assert refreshed_sql.count("%s") == len(refreshed_params)
    # The window size and the count reach the caller through the refresh wrapper too
    assert refreshed_sql.endswith(
        "SELECT prefix, key, value, created_at, updated_at, score, window_size, "
        "namespace_vectors\nFROM search_results\nORDER BY score DESC, prefix, key"
    )

    with pytest.raises(ValueError, match="index"):
        StoreSQL("public", "store").semantic_search_in(
            filter=None, limit=1, offset=0, omit_expired=True, over_fetch=3
        )


def test_namespace_lookup_seeks_each_distinct_namespace():
    """namespace_lookup finds the namespaces under a prefix, one index seek each.

    The range ">= 'a.b.' AND < 'a.b/'" holds exactly the strings that start
    with "a.b.", because "/" is the byte after ".". It constrains the primary
    key scan on every supported version; LIKE with ESCAPE does not on v25.4.
    """
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.namespace_lookup(["a", "b"])
    assert query_sql.startswith("WITH RECURSIVE found (prefix, n) AS (")
    assert "FROM public.store_vectors" in query_sql
    assert "LIKE" not in query_sql
    assert "ORDER BY prefix LIMIT 1" in query_sql
    # The lookup stops after MAX_SEARCH_NAMESPACES + 1 namespaces
    assert "found.n + 1" in query_sql
    assert "WHERE found.prefix IS NOT NULL AND found.n <= %s" in query_sql
    assert params == ["a.b", "a.b.", "a.b/"] * 2 + [MAX_SEARCH_NAMESPACES]
    assert MAX_SEARCH_NAMESPACES == 1000
    assert query_sql.count("%s") == len(params)
    assert query_sql.rstrip().endswith("SELECT prefix FROM found WHERE prefix IS NOT NULL")
    # Labels may hold LIKE wildcards; the range form needs no escaping
    _, params = sql.namespace_lookup(["50%_off"])
    assert params[:3] == ["50%_off", "50%_off.", "50%_off/"]

    with pytest.raises(ValueError, match="namespace"):
        sql.namespace_lookup([])


def test_bind_replaces_placeholders_only():
    statement = ("SELECT %s, %s, %s", [QUERY_VECTOR, 1, QUERY_VECTOR])
    assert bind(statement, QUERY_VECTOR, "[1.0]") == ("SELECT %s, %s, %s", ["[1.0]", 1, "[1.0]"])
    # The original statement is left as it was
    assert statement[1][0] is QUERY_VECTOR
    assert StoreSQL.bind_query_vector(statement, [1.0, 2.5]) == (
        "SELECT %s, %s, %s",
        ["[1.0,2.5]", 1, "[1.0,2.5]"],
    )


def test_beam_size_statement_formats_a_validated_int():
    sql = StoreSQL("public", "store", index=_make_index())
    assert sql.beam_size_statement(16) == "SET LOCAL vector_search_beam_size = 16"
    for bad in (0, -1, True, 1.5, "8; DROP TABLE x"):
        with pytest.raises(ValueError, match="beam_size"):
            sql.beam_size_statement(bad)  # type: ignore[arg-type]


def test_vector_index_statements():
    sql = StoreSQL("app", "mem", index=_make_index())
    assert (
        sql.drop_vector_index() == "DROP INDEX IF EXISTS app.mem_vectors@mem_vectors_embedding_idx"
    )
    assert sql.show_create_vectors() == (
        "SELECT create_statement FROM [SHOW CREATE TABLE app.mem_vectors]"
    )


# The history table's columns, in order, as setup() creates them
HISTORY_COLUMNS = [
    "prefix TEXT NOT NULL",
    "key TEXT NOT NULL",
    "version INT8 NOT NULL",
    "value JSONB NOT NULL",
    "provenance JSONB",
    "valid_at TIMESTAMPTZ NOT NULL",
    "invalid_at TIMESTAMPTZ",
    "written_at TIMESTAMPTZ NOT NULL",
    "ended_at TIMESTAMPTZ NOT NULL",
    "reason TEXT NOT NULL",
    "superseded_by TEXT",
    "write_id UUID",
    "expires_at TIMESTAMPTZ",
]


def test_history_table_statement() -> None:
    sql = StoreSQL("app", "mem")
    assert sql.history == "app.mem_history"
    [create] = sql.history_statements(None)
    lines = create.splitlines()
    assert lines[0] == "CREATE TABLE IF NOT EXISTS app.mem_history ("
    assert lines[-1] == ");"
    assert [line.strip().rstrip(",") for line in lines[1:-1]] == [
        *HISTORY_COLUMNS,
        "PRIMARY KEY (prefix, key, version)",
    ]


def test_history_ttl_statement_only_with_ttl_days() -> None:
    sql = StoreSQL("public", "store")
    [create] = sql.history_statements(None)
    statements = sql.history_statements(30.0)
    assert statements == [
        create,
        "ALTER TABLE public.store_history SET "
        "(ttl_expiration_expression = 'expires_at', ttl_job_cron = '@hourly');",
    ]
    # Each archived row carries its own expires_at, so the days are not in the DDL
    assert "30" not in statements[1]


def test_archive_statement_binds_everything() -> None:
    sql = StoreSQL("public", "store")
    when = datetime(2026, 1, 2, tzinfo=timezone.utc)
    write_id = "8d6f1c9e-0000-4000-8000-000000000001"
    query, params = sql.archive(
        "user.alice",
        ["seat-key", "food-key"],
        reason="overwritten",
        superseded_by="newer-key",
        invalid_at=when,
        write_id=write_id,
        ttl_days=30.0,
    )
    for value in ("user.alice", "seat-key", "food-key", "overwritten", "newer-key", write_id):
        assert value not in query
    assert query.startswith("INSERT INTO public.store_history (")
    assert "\nFROM public.store\n" in query
    assert "ON CONFLICT (prefix, key, version) DO NOTHING" in query
    # Only live rows, and not the rows this batch already wrote
    assert "(expires_at IS NULL OR expires_at > now())" in query
    assert "write_id IS DISTINCT FROM CAST(%s AS UUID)" in query
    assert "COALESCE(valid_at, updated_at)" in query
    assert query.count("%s") == len(params)
    assert "::" not in query
    assert params == [
        when,
        "overwritten",
        "newer-key",
        write_id,
        30.0,
        "user.alice",
        ["seat-key", "food-key"],
        write_id,
    ]

    # No invalid time means now, and no ttl_days means no expiry: both bind NULL
    _, params = sql.archive(
        "user.alice",
        ["seat-key"],
        reason="deleted",
        superseded_by=None,
        invalid_at=None,
        write_id=write_id,
        ttl_days=None,
    )
    assert params == [None, "deleted", None, write_id, None, "user.alice", ["seat-key"], write_id]


def test_history_reads_bind_the_key_and_skip_expired_versions() -> None:
    sql = StoreSQL("public", "store")
    query, params = sql.history_versions("user.alice", "seat-key", limit=5)
    assert "FROM public.store_history\n" in query
    assert "prefix = %s AND key = %s" in query
    assert "(expires_at IS NULL OR expires_at > now())" in query
    assert "ORDER BY version DESC\nLIMIT %s" in query
    assert params == ["user.alice", "seat-key", 5]

    # The schema check asks for the history table by its validated name
    query, params = sql.history_exists()
    assert query == "SELECT to_regclass(%s) IS NOT NULL AS ready"
    assert params == ["public.store_history"]


def test_purge_deletes_the_item_its_vectors_and_history_in_one_statement() -> None:
    query, params = StoreSQL("public", "store", index=_make_index()).purge("user.alice", "k1")
    assert query.startswith("WITH deleted_items AS (")
    for table in ("public.store", "public.store_vectors", "public.store_history"):
        assert f"DELETE FROM {table} WHERE prefix = %s AND key = %s RETURNING 1" in query
    assert query.rstrip().endswith("AS removed")
    assert params == ["user.alice", "k1"] * 3

    # Without an index there are no vectors to delete
    query, params = StoreSQL("public", "store").purge("user.alice", "k1")
    assert "store_vectors" not in query
    assert params == ["user.alice", "k1"] * 2


def test_delete_namespace_with_history_deletes_its_history_too() -> None:
    sql = StoreSQL("public", "store", index=_make_index())
    without = sql.delete_namespace(["a", "b"])
    query, params = sql.delete_namespace(["a", "b"], history=True)
    assert "DELETE FROM public.store_history\n" in query
    # The same prefix condition as the items, in the same statement
    assert query.count("(prefix = %s OR (prefix >= %s AND prefix < %s))") == 3
    assert query.endswith("RETURNING key")
    assert params == ["a.b", "a.b.", "a.b/"] * 3
    assert sql.delete_namespace(["a", "b"], history=False) == without

    query, params = StoreSQL("public", "store").delete_namespace(["a"], history=True)
    assert query.startswith("WITH deleted_history AS (")
    assert "store_vectors" not in query
    assert params == ["a", "a.", "a/"] * 2


def test_upsert_with_history_continues_versions_after_the_history() -> None:
    sql = StoreSQL("public", "store")
    rows = [("user.a", "k", '{"text": "x"}', None, None, None)]
    query, params = sql.upsert_items(rows, write_id="w-1", history=True)
    # A key written again after a delete never reuses a version it has in history
    assert (
        "COALESCE((SELECT max(h.version) FROM public.store_history AS h "
        "WHERE h.prefix = rows_to_insert.prefix AND h.key = rows_to_insert.key), 0) + 1"
    ) in query
    assert "version = public.store.version + 1" in query
    assert "WHERE public.store.write_id IS DISTINCT FROM EXCLUDED.write_id" in query
    assert query.count("%s") == len(params)
    assert params == ["w-1", *rows[0]]
    # Without history the statement is the one Task 1 sends
    assert "store_history" not in sql.upsert_items(rows, write_id="w-1")[0]
    with pytest.raises(ValueError, match="write id"):
        sql.upsert_items([("user.a", "k", "{}", None)], history=True)


def test_delete_namespace_without_history_keeps_the_0_5_1_text() -> None:
    """With no history table to clear, delete_namespace sends the 0.5.1 statement."""
    condition = "(prefix = %s OR (prefix >= %s AND prefix < %s))"
    assert StoreSQL("public", "store").delete_namespace(["user", "a"]) == (
        f"DELETE FROM public.store\nWHERE {condition}\nRETURNING key",
        ["user.a", "user.a.", "user.a/"],
    )
    assert StoreSQL("public", "store", index=_make_index()).delete_namespace(["user", "a"]) == (
        "WITH deleted_vectors AS (\n"
        "    DELETE FROM public.store_vectors\n"
        f"    WHERE {condition}\n"
        "    RETURNING key\n"
        ")\n"
        f"DELETE FROM public.store\nWHERE {condition}\nRETURNING key",
        ["user.a", "user.a.", "user.a/"] * 2,
    )


CANDIDATE_COLUMNS = (
    "key, value, created_at, updated_at, version, COALESCE(valid_at, updated_at) AS valid_at, "
    "provenance, confirmed_at, confirmations, write_id"
)


def test_key_slot_candidates_bind_the_slot() -> None:
    """The target comes by its key alone; slot matches compare the bound slot field and value.

    The filter narrows the slot matches only, and expired rows never come back.
    """
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.key_slot_candidates(
        "user.alice",
        "k",
        slot_field="slot-field-xyz",
        slot_value_json='"slot-value-xyz"',
        filter={"source": "chat"},
    )
    assert "slot-field-xyz" not in query_sql and "slot-value-xyz" not in query_sql
    assert "chat" not in query_sql and "alice" not in query_sql
    assert query_sql.startswith(f"SELECT {CANDIDATE_COLUMNS}\nFROM public.store\n")
    assert (
        "WHERE prefix = %s AND (key = %s OR (value -> %s = CAST(%s AS JSONB) "
        "AND value -> %s = CAST(%s AS JSONB)))"
    ) in query_sql
    assert query_sql.rstrip().endswith("AND (expires_at IS NULL OR expires_at > now())")
    assert params == ["user.alice", "k", "slot-field-xyz", '"slot-value-xyz"', "source", '"chat"']
    assert query_sql.count("%s") == len(params)

    # Without a slot only the target is read, and the filter does not apply to it
    query_sql, params = sql.key_slot_candidates(
        "user.alice", "k", slot_field=None, slot_value_json=None, filter={"source": "chat"}
    )
    assert "WHERE prefix = %s AND key = %s\n" in query_sql
    assert "->" not in query_sql
    assert params == ["user.alice", "k"]

    with pytest.raises(ValueError, match="Unknown filter operator"):
        sql.key_slot_candidates(
            "a", "k", slot_field="seat", slot_value_json='"x"', filter={"x": {"$bad": 1}}
        )


def test_similar_candidates_exact_and_index_shapes() -> None:
    """The exact form ranks every vector of the namespace; the index form keeps a window.

    Both skip the excluded keys, filter, check expiry and keep the scores at
    or above the threshold, best first, up to the limit. The index form also
    reports the window's size, its lowest score and the namespace's vectors,
    for needs_exact_candidates.
    """
    sql = StoreSQL("public", "store", index=_make_index())
    kwargs = {"threshold": 0.8, "limit": 5, "filter": {"x": 1}, "over_fetch": 11}
    exact_sql, exact_params = sql.similar_candidates(
        "user.alice",
        scan="exact",
        exclude=["k", "s"],
        **kwargs,  # type: ignore[arg-type]
    )
    ranked, outer = exact_sql.split("\n)\nSELECT ", 1)
    assert ranked.startswith("WITH ranked AS (")
    assert "JOIN public.store AS i ON v.prefix = i.prefix AND v.key = i.key" in ranked
    assert (
        "WHERE v.prefix = %s AND NOT (v.key = ANY(CAST(%s AS STRING[]))) "
        "AND i.value -> %s = CAST(%s AS JSONB) AND (i.expires_at IS NULL OR i.expires_at > now())"
    ) in ranked
    assert "min(v.embedding <=> CAST(%s AS VECTOR)) AS distance" in ranked
    assert "GROUP BY i.prefix, i.key" in ranked
    # No inner LIMIT on the vectors, so no vector index can serve it
    assert "LIMIT" not in ranked and "ORDER BY" not in ranked
    assert outer.endswith(
        "1 - distance AS score\nFROM ranked\nWHERE 1 - distance >= CAST(%s AS FLOAT8)\n"
        "ORDER BY score DESC, key\nLIMIT %s"
    )
    assert exact_params == [QUERY_VECTOR, "user.alice", ["k", "s"], "x", "1", 0.8, 5]
    assert exact_sql.count("%s") == len(exact_params)

    index_sql, index_params = sql.similar_candidates(
        "user.alice",
        scan="index",
        exclude=["k", "s"],
        **kwargs,  # type: ignore[arg-type]
    )
    nearest, rest = index_sql.split("\n),\nwindow_stats AS (\n", 1)
    # The window orders by distance with a LIMIT, which the C-SPANN index serves,
    # and has no other filter
    assert nearest.startswith("WITH nearest AS (")
    assert "FROM public.store_vectors\n    WHERE prefix = %s\n" in nearest
    assert nearest.endswith("ORDER BY embedding <=> CAST(%s AS VECTOR)\n    LIMIT %s")
    assert "expires_at" not in nearest and "value" not in nearest and "ANY(" not in nearest
    # The window's size and lowest score, over every vector in it: before the
    # filter, the expiry check and the excluded keys. And the namespace's
    # vectors, counted from the primary key up to over_fetch, which tells a
    # window the index cut short
    stats, rest = rest.split("\n),\nbest AS (\n", 1)
    assert stats == (
        "    SELECT count(*) AS window_size, min(1 - distance) AS lowest_score, (\n"
        "        SELECT count(*) FROM (\n"
        "            SELECT 1 FROM public.store_vectors WHERE prefix = %s LIMIT %s\n"
        "        ) AS c\n"
        "    ) AS namespace_vectors\n"
        "    FROM nearest"
    )
    best, rest = rest.split("\n),\nhits AS (\n", 1)
    assert "JOIN public.store AS i ON r.prefix = i.prefix AND r.key = i.key" in best
    assert (
        "WHERE NOT (i.key = ANY(CAST(%s AS STRING[]))) AND i.value -> %s = CAST(%s AS JSONB) "
        "AND (i.expires_at IS NULL OR i.expires_at > now())"
    ) in best
    assert "SELECT DISTINCT ON (i.key)" in best and "ORDER BY i.key, r.distance" in best
    hits, outer = rest.split("\n)\nSELECT ", 1)
    assert hits.endswith(
        "    FROM best\n    WHERE 1 - distance >= CAST(%s AS FLOAT8)\n"
        "    ORDER BY score DESC, key\n    LIMIT %s"
    )
    # Every row carries the window's figures. Without a candidate one row
    # still does, with NULL candidate columns
    assert outer == (
        "h.key, h.value, h.created_at, h.updated_at, h.version, h.valid_at, h.provenance, "
        "h.confirmed_at, h.confirmations, h.write_id, h.score, w.window_size, w.lowest_score, "
        "w.namespace_vectors\n"
        "FROM window_stats AS w\nLEFT JOIN hits AS h ON TRUE\nORDER BY h.score DESC, h.key"
    )
    assert index_params == [
        QUERY_VECTOR,
        "user.alice",
        QUERY_VECTOR,
        11,
        "user.alice",
        11,
        ["k", "s"],
        "x",
        "1",
        0.8,
        5,
    ]
    assert index_sql.count("%s") == len(index_params)

    # An l2 store scores by the negated distance, as search does
    l2 = StoreSQL("public", "store", index=_make_index(distance=DistanceStrategy.EUCLIDEAN))
    for scan in ("exact", "index"):
        l2_sql, _ = l2.similar_candidates(
            "a",
            scan=scan,  # type: ignore[arg-type]
            exclude=[],
            **kwargs,  # type: ignore[arg-type]
        )
        assert "<->" in l2_sql and "<=>" not in l2_sql
        assert "-distance AS score" in l2_sql and "WHERE -distance >= CAST(%s AS FLOAT8)" in l2_sql
        assert ("min(-distance) AS lowest_score" in l2_sql) == (scan == "index")

    with pytest.raises(ValueError, match="scan"):
        sql.similar_candidates("a", scan="fast", exclude=[], **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        sql.similar_candidates(
            "a", threshold=0.8, limit=-1, filter=None, scan="exact", over_fetch=1, exclude=[]
        )
    with pytest.raises(ValueError, match="index"):
        StoreSQL("public", "store").similar_candidates("a", scan="exact", exclude=[], **kwargs)  # type: ignore[arg-type]


def test_candidate_scores_count_record_and_now_bind_everything() -> None:
    sql = StoreSQL("public", "store", index=_make_index())

    # The key and slot candidates are scored by their vectors alone: primary
    # key lookups, no threshold and no filter
    query_sql, params = sql.candidate_scores("user.alice", ["k", "s"])
    assert query_sql == (
        "SELECT key, 1 - min(embedding <=> CAST(%s AS VECTOR)) AS score\n"
        "FROM public.store_vectors\n"
        "WHERE prefix = %s AND key = ANY(CAST(%s AS STRING[]))\n"
        "GROUP BY key"
    )
    assert params == [QUERY_VECTOR, "user.alice", ["k", "s"]]
    l2 = StoreSQL("public", "store", index=_make_index(distance=DistanceStrategy.EUCLIDEAN))
    assert "-min(embedding <-> CAST(%s AS VECTOR)) AS score" in l2.candidate_scores("a", ["k"])[0]

    # The namespace's own vectors, counted up to the cap from the primary key
    query_sql, params = sql.namespace_vector_count("user.alice", 2001)
    assert query_sql == (
        "SELECT count(*) AS n FROM (\n"
        "    SELECT 1 FROM public.store_vectors WHERE prefix = %s LIMIT %s\n"
        ") AS c"
    )
    assert params == ["user.alice", 2001]

    # A record is one live row with its metadata. It never pushes back the expiry
    query_sql, params = sql.record("user.alice", "k")
    assert query_sql == (
        f"SELECT {CANDIDATE_COLUMNS}\nFROM public.store\n"
        "WHERE prefix = %s AND key = %s AND (expires_at IS NULL OR expires_at > now())"
    )
    assert params == ["user.alice", "k"]
    assert "UPDATE" not in query_sql

    assert sql.now() == ("SELECT now() AS now", [])

    for builder in (
        lambda s: s.candidate_scores("a", ["k"]),
        lambda s: s.namespace_vector_count("a", 1),
    ):
        with pytest.raises(ValueError, match="index"):
            builder(StoreSQL("public", "store"))
