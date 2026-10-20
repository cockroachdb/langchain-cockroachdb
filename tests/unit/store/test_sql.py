"""Tests for store SQL statement building."""

import pytest

from langchain_cockroachdb.indexes import DistanceStrategy
from langchain_cockroachdb.store.config import ResolvedIndex
from langchain_cockroachdb.store.sql import QUERY_VECTOR, StoreSQL


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


def test_identifiers_are_validated():
    """StoreSQL validates identifiers and raises ValueError for invalid ones."""
    with pytest.raises(ValueError, match="Invalid"):
        StoreSQL("public", "store; DROP TABLE x")

    with pytest.raises(ValueError, match="Invalid"):
        StoreSQL("public; --", "store")


def test_vector_statements_are_separate_and_idempotent():
    """migrations() returns only 2 statements; vector_statements() is separate."""
    # Without index: migrations has 2 statements, vector_statements is empty
    sql_no_index = StoreSQL("public", "store", index=None)
    assert len(sql_no_index.migrations()) == 2
    assert len(sql_no_index.vector_statements()) == 0

    # With index (not flat): migrations still has 2, vector_statements has 2
    sql_with_index = StoreSQL("public", "store", index=_make_index())
    assert len(sql_with_index.migrations()) == 2
    assert len(sql_with_index.vector_statements()) == 2

    # With flat index: migrations has 2, vector_statements has 1 (no index statement)
    sql_flat = StoreSQL("public", "store", index=_make_index(flat=True))
    assert len(sql_flat.migrations()) == 2
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
    """semantic_search parameters appear in correct order: query vector twice, namespace, filter, over_fetch, limit, offset."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.semantic_search(
        namespace_prefix=["a", "b"],
        filter={"x": 1},
        limit=5,
        offset=10,
        omit_expired=True,
        over_fetch=100,
    )

    # QUERY_VECTOR appears twice (CTE and score calculation)
    assert params[0] is QUERY_VECTOR
    assert params[1] is QUERY_VECTOR

    # Then namespace params (exact match "a.b", then pattern "a.b.%")
    assert params[2] == "a.b"
    assert params[3] == "a.b.%"

    # Then filter params (key "x", value "1")
    assert params[4] == "x"
    assert params[5] == "1"

    # Then over_fetch, limit, offset
    assert params[6] == 100
    assert params[7] == 5
    assert params[8] == 10


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
        over_fetch=100,
    )
    assert "::" not in query_sql


def test_get():
    """get() builds SELECT with prefix and key list."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.get(["a", "b"], ["k1", "k2"], omit_expired=True)

    assert "SELECT" in query_sql
    assert "FROM public.store" in query_sql
    assert "prefix = %s" in query_sql
    assert "key = ANY(%s)" in query_sql
    assert "(expires_at IS NULL OR expires_at > now())" in query_sql

    assert params == ["a.b", ["k1", "k2"]]


def test_refresh():
    """refresh() builds UPDATE to extend TTL."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.refresh(["a"], ["k1"])

    assert "UPDATE public.store" in query_sql
    assert "expires_at = now() + ttl_minutes * INTERVAL '1 minute'" in query_sql
    assert "ttl_minutes IS NOT NULL" in query_sql

    assert params == ["a", ["k1"]]


def test_refresh_vectors():
    """refresh_vectors() mirrors refresh for vectors table."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.refresh_vectors(["a"], ["k1"])

    assert "UPDATE public.store_vectors" in query_sql
    assert "expires_at = now() + ttl_minutes * INTERVAL '1 minute'" in query_sql

    assert params == ["a", ["k1"]]


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


def test_delete_items():
    """delete_items() builds DELETE with prefix and key list."""
    sql = StoreSQL("public", "store", index=None)
    query_sql, params = sql.delete_items(["a"], ["k1", "k2"])

    assert "DELETE FROM public.store" in query_sql
    assert "prefix = %s" in query_sql
    assert "key = ANY(%s)" in query_sql

    assert params == ["a", ["k1", "k2"]]


def test_delete_vectors():
    """delete_vectors() mirrors delete_items for vectors table."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.delete_vectors(["a"], ["k1"])

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


def test_delete_namespace():
    """delete_namespace() returns list of (sql, params) tuples."""
    # With index: vectors delete, then items delete
    sql = StoreSQL("public", "store", index=_make_index())
    deletes = sql.delete_namespace(["a", "b"])

    assert len(deletes) == 2

    # Vectors delete
    vectors_sql, vectors_params = deletes[0]
    assert "DELETE FROM public.store_vectors" in vectors_sql
    assert "(prefix = %s OR prefix LIKE %s ESCAPE '\\')" in vectors_sql
    assert vectors_params == ["a.b", "a.b.%"]

    # Items delete with RETURNING
    items_sql, items_params = deletes[1]
    assert "DELETE FROM public.store" in items_sql
    assert "RETURNING key" in items_sql
    assert items_params == ["a.b", "a.b.%"]

    # Without index: only items delete
    sql_no_index = StoreSQL("public", "store", index=None)
    deletes = sql_no_index.delete_namespace(["a"])

    assert len(deletes) == 1
    assert "RETURNING key" in deletes[0][0]


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
