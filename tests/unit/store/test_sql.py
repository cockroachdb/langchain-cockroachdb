"""Tests for store SQL statement building."""

import pytest

from langchain_cockroachdb.indexes import DistanceStrategy
from langchain_cockroachdb.store.config import ResolvedIndex
from langchain_cockroachdb.store.sql import NAMESPACES, QUERY_VECTOR, StoreSQL, bind


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
    """semantic_search parameters match placeholder order in SQL text."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.semantic_search(
        namespace_prefix=["a", "b"],
        filter={"x": 1},
        limit=5,
        offset=10,
        omit_expired=True,
        over_fetch=100,
    )

    # Order: QUERY_VECTOR, namespace params, QUERY_VECTOR, over_fetch, filter params, limit, offset
    # First QUERY_VECTOR (in ranked CTE)
    assert params[0] is QUERY_VECTOR

    # Namespace params (exact match "a.b", then pattern "a.b.%")
    assert params[1] == "a.b"
    assert params[2] == "a.b.%"

    # Second QUERY_VECTOR (ORDER BY in ranked CTE)
    assert params[3] is QUERY_VECTOR

    # over_fetch
    assert params[4] == 100

    # Filter params (key "x", value "1")
    assert params[5] == "x"
    assert params[6] == "1"

    # limit, offset
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
        over_fetch=7,
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


def test_delete_namespace():
    """delete_namespace() is one statement; vectors go in a CTE with an index."""
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, params = sql.delete_namespace(["a", "b"])

    assert query_sql.startswith("WITH deleted_vectors AS (")
    assert "DELETE FROM public.store_vectors" in query_sql
    assert "DELETE FROM public.store\n" in query_sql
    assert "(prefix = %s OR prefix LIKE %s ESCAPE '\\')" in query_sql
    assert query_sql.endswith("RETURNING key")
    assert params == ["a.b", "a.b.%", "a.b", "a.b.%"]

    # Without index: only items delete
    sql_no_index = StoreSQL("public", "store", index=None)
    query_sql, params = sql_no_index.delete_namespace(["a"])
    assert query_sql.startswith("DELETE FROM public.store\n")
    assert "store_vectors" not in query_sql
    assert params == ["a", "a.%"]


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

    # semantic_search
    query_sql, params = sql.semantic_search(
        namespace_prefix=["a", "b", "c"],
        filter={"x": 1, "y": 2},
        limit=10,
        offset=5,
        omit_expired=True,
        over_fetch=100,
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

    # sweep
    sweeps = sql.sweep()
    for query_sql, params in sweeps:
        assert query_sql.count("%s") == len(params)


def test_vector_index_name_is_public():
    """vector_index_name is exposed as a public attribute."""
    sql = StoreSQL("public", "my_store", index=_make_index())
    assert sql.vector_index_name == "my_store_vectors_embedding_idx"


def test_ranked_vectors_carry_no_expiry_condition():
    """The vector CTE ranks without the expiry check, so C-SPANN can serve it.

    The items join checks expiry, and vectors expire with their item.
    """
    sql = StoreSQL("public", "store", index=_make_index())
    query_sql, _ = sql.semantic_search(
        namespace_prefix=["a"], filter=None, limit=3, offset=0, omit_expired=True, over_fetch=7
    )
    ranked, rest = query_sql.split("\n),\nbest AS (", 1)
    assert "expires_at" not in ranked
    assert "(i.expires_at IS NULL OR i.expires_at > now())" in rest


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
    assert params == [QUERY_VECTOR, NAMESPACES, QUERY_VECTOR, 11, "x", "1", 5, 2]
    assert query_sql.count("%s") == len(params)
    assert "::" not in query_sql

    refreshed_sql, refreshed_params = sql.semantic_search_in(
        filter=None, limit=5, offset=0, omit_expired=True, over_fetch=11, refresh=True
    )
    assert refreshed_params == [QUERY_VECTOR, NAMESPACES, QUERY_VECTOR, 11, 5, 0]
    assert refreshed_sql.startswith("WITH search_results AS (")
    assert refreshed_sql.count("%s") == len(refreshed_params)

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
    assert query_sql.startswith("WITH RECURSIVE found (prefix) AS (")
    assert "FROM public.store_vectors" in query_sql
    assert "LIKE" not in query_sql
    assert "ORDER BY prefix LIMIT 1" in query_sql
    assert params == ["a.b", "a.b.", "a.b/"] * 2
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
