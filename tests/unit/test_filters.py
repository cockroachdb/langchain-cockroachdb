"""Unit tests for the metadata filter builder."""

import json

import pytest

from langchain_cockroachdb._filters import FilterBuilder


def build(filter: dict) -> tuple[str, dict]:
    builder = FilterBuilder("metadata")
    return builder.build(filter), builder.params


def test_shorthand_equality_binds_key_and_value() -> None:
    sql, params = build({"merchant": "Trader Joe's"})
    assert sql == "metadata->CAST(:f0 AS STRING) = CAST(:f1 AS JSONB)"
    assert params == {"f0": "merchant", "f1": json.dumps("Trader Joe's")}


@pytest.mark.parametrize("value", [True, False, None, 3, 2.5, "x", ["a", 1], {"n": 1}])
def test_equality_json_encodes_every_type(value) -> None:
    _, params = build({"k": {"$eq": value}})
    assert params["f1"] == json.dumps(value)


def test_not_equal() -> None:
    sql, params = build({"k": {"$ne": 1}})
    assert sql == "metadata->CAST(:f0 AS STRING) != CAST(:f1 AS JSONB)"
    assert params == {"f0": "k", "f1": "1"}


@pytest.mark.parametrize("op,sql_op", [("$lt", "<"), ("$lte", "<="), ("$gt", ">"), ("$gte", ">=")])
def test_numeric_range_ignores_other_json_types(op: str, sql_op: str) -> None:
    sql, params = build({"year": {op: 2020}})
    assert sql == (
        "(CASE WHEN jsonb_typeof(metadata->CAST(:f0 AS STRING)) = 'number' "
        "THEN CAST(metadata->>CAST(:f0 AS STRING) AS DECIMAL) END) "
        f"{sql_op} CAST(:f1 AS DECIMAL)"
    )
    assert params == {"f0": "year", "f1": 2020}


def test_string_range_compares_text() -> None:
    sql, params = build({"day": {"$gte": "2024-01-01"}})
    assert "= 'string' THEN metadata->>CAST(:f0 AS STRING) END) >= :f1" in sql
    assert params["f1"] == "2024-01-01"


def test_between() -> None:
    sql, params = build({"n": {"$between": [1, 5]}})
    assert sql.startswith("(") and " AND " in sql
    assert params == {"f0": "n", "f1": 1, "f2": 5}


def test_in_and_not_in() -> None:
    sql, params = build({"tag": {"$in": ["a", 2]}})
    assert sql == "metadata->CAST(:f0 AS STRING) IN (CAST(:f1 AS JSONB), CAST(:f2 AS JSONB))"
    assert params == {"f0": "tag", "f1": '"a"', "f2": "2"}
    sql, _ = build({"tag": {"$nin": ["a"]}})
    assert " NOT IN (" in sql


def test_empty_in_lists() -> None:
    assert build({"k": {"$in": []}})[0] == "FALSE"
    assert build({"k": {"$nin": []}})[0] == "TRUE"


def test_exists() -> None:
    assert build({"k": {"$exists": True}})[0] == "metadata ? CAST(:f0 AS STRING)"
    assert build({"k": {"$exists": False}})[0] == "NOT (metadata ? CAST(:f0 AS STRING))"


def test_like_and_ilike() -> None:
    sql, params = build({"name": {"$like": "Tra%"}})
    assert sql == "metadata->>CAST(:f0 AS STRING) LIKE :f1"
    assert params["f1"] == "Tra%"
    assert " ILIKE " in build({"name": {"$ilike": "%joe%"}})[0]


def test_and_or_not_nesting() -> None:
    sql, _ = build({"$or": [{"a": 1}, {"$and": [{"b": {"$gt": 2}}, {"c": "x"}]}]})
    assert sql.startswith("(") and " OR " in sql and " AND " in sql
    assert build({"$not": {"a": 1}})[0].startswith("NOT (")


def test_top_level_keys_next_to_operators_are_kept() -> None:
    sql, params = build({"$or": [{"a": 1}, {"b": 2}], "c": 3})
    assert " AND " in sql
    assert "c" in params.values()


def test_several_operators_on_one_key() -> None:
    assert " AND " in build({"n": {"$gte": 1, "$lte": 5}})[0]


def test_injection_and_colons_stay_in_parameters() -> None:
    key = "x') OR ('1'='1"
    sql, params = build({key: {"$ne": "a:b"}, "a:b": 1})
    assert key not in sql
    assert "a:b" not in sql
    assert key in params.values() and "a:b" in params.values()


def test_empty_filter_is_true() -> None:
    assert build({}) == ("TRUE", {})


@pytest.mark.parametrize(
    "bad",
    [
        {"$and": []},
        {"$or": "x"},
        {"k": {}},
        {"k": {"$nope": 1}},
        {"$where": 1},
        {"k": {"$in": "abc"}},
        {"k": {"$between": [1]}},
        {"k": {"$exists": "yes"}},
        {"k": {"$gt": True}},
        {"k": {"$gt": [1]}},
        {"k": {"$like": 5}},
    ],
)
def test_malformed_filters_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        build(bad)


def test_non_dict_filter_raises() -> None:
    with pytest.raises(ValueError):
        FilterBuilder("metadata").build(["a"])  # type: ignore[arg-type]


def test_bind_with_custom_prefix() -> None:
    builder = FilterBuilder("metadata", prefix="q")
    assert builder.bind("tenant") == ":q0"
    assert builder.params == {"q0": "tenant"}
