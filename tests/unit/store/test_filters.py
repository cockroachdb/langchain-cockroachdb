"""Tests for filter translation to SQL."""

import pytest

from langchain_cockroachdb.store.filters import translate_filter


def test_plain_equality():
    """Plain scalar maps to jsonb equality."""
    conditions, params = translate_filter({"status": "active"})
    assert conditions == ["value -> %s = CAST(%s AS JSONB)"]
    assert params == ["status", '"active"']


def test_list_equality_whole():
    """List values become jsonb equality on the whole array."""
    conditions, params = translate_filter({"tags": ["a", "b"]})
    assert conditions == ["value -> %s = CAST(%s AS JSONB)"]
    assert params == ["tags", '["a", "b"]']


def test_nested_dict_leaf_paths():
    """Nested dict without $ operators becomes leaf path conditions."""
    conditions, params = translate_filter({"profile": {"lang": "en", "n": 1}})
    assert len(conditions) == 2
    # Check both conditions use #> operator
    assert all(c == "value #> %s = CAST(%s AS JSONB)" for c in conditions)
    # Check parameters (order might vary, so check both possibilities)
    assert params in [
        [["profile", "lang"], '"en"', ["profile", "n"], "1"],
        [["profile", "n"], "1", ["profile", "lang"], '"en"'],
    ]


def test_numeric_comparison_guarded():
    """Numeric comparisons check jsonb_typeof and cast to DECIMAL."""
    conditions, params = translate_filter({"score": {"$gt": 9}})
    assert len(conditions) == 1
    assert "jsonb_typeof" in conditions[0]
    assert "AS DECIMAL" in conditions[0]
    assert params == ["score", "score", 9]


def test_text_comparison_for_strings():
    """String comparisons use ->> (text extraction) without type checking."""
    conditions, params = translate_filter({"ts": {"$gte": "2026-01-01"}})
    assert len(conditions) == 1
    assert conditions[0] == "value ->> %s >= %s"
    assert params == ["ts", "2026-01-01"]


def test_bool_is_not_numeric():
    """Boolean values are not numeric and raise ValueError for comparison operators."""
    with pytest.raises(ValueError, match="comparison"):
        translate_filter({"n": {"$gt": True}})


def test_filter_missing_key_semantics():
    """$ne uses IS DISTINCT FROM to match missing keys, $eq uses =."""
    # $ne should use IS DISTINCT FROM
    conditions, params = translate_filter({"x": {"$ne": 1}})
    assert len(conditions) == 1
    assert "IS DISTINCT FROM" in conditions[0]
    assert params == ["x", "1"]

    # $eq should use = (same as plain equality)
    conditions, params = translate_filter({"x": {"$eq": 1}})
    assert len(conditions) == 1
    assert conditions[0] == "value -> %s = CAST(%s AS JSONB)"
    assert params == ["x", "1"]


def test_unknown_operator():
    """Unknown $ operators raise ValueError naming the operator."""
    with pytest.raises(ValueError, match=r"\$in"):
        translate_filter({"n": {"$in": [1]}})


def test_unknown_operator_below_nested_path_raises():
    """Unknown $ operators below nested paths raise ValueError naming the operator."""
    with pytest.raises(ValueError, match=r"\$regex"):
        translate_filter({"p": {"n": {"$regex": "x"}}})


def test_none_filter():
    """None and empty dict return no conditions."""
    conditions, params = translate_filter(None)
    assert conditions == []
    assert params == []

    conditions, params = translate_filter({})
    assert conditions == []
    assert params == []


def test_all_comparison_operators():
    """Test all supported comparison operators."""
    # $eq
    conditions, params = translate_filter({"x": {"$eq": 5}})
    assert "=" in conditions[0]

    # $ne
    conditions, params = translate_filter({"x": {"$ne": 5}})
    assert "IS DISTINCT FROM" in conditions[0]

    # $gt
    conditions, params = translate_filter({"x": {"$gt": 5}})
    assert ">" in conditions[0]

    # $gte
    conditions, params = translate_filter({"x": {"$gte": 5}})
    assert ">=" in conditions[0]

    # $lt
    conditions, params = translate_filter({"x": {"$lt": 5}})
    assert "<" in conditions[0]

    # $lte
    conditions, params = translate_filter({"x": {"$lte": 5}})
    assert "<=" in conditions[0]


def test_custom_value_column():
    """value_column parameter changes the column name in conditions."""
    conditions, params = translate_filter({"x": 1}, value_column="data")
    assert conditions == ["data -> %s = CAST(%s AS JSONB)"]
    assert params == ["x", "1"]


def test_nested_path_with_numeric_operator():
    """Numeric operator at nested path uses #> and #>>."""
    conditions, params = translate_filter({"profile": {"n": {"$gt": 1}}})
    assert len(conditions) == 1
    assert "jsonb_typeof" in conditions[0]
    assert "#>" in conditions[0]
    assert "#>>" in conditions[0]
    assert "AS DECIMAL" in conditions[0]
    assert params == [["profile", "n"], ["profile", "n"], 1]


def test_nested_path_with_ne_operator():
    """$ne at nested path uses #> with IS DISTINCT FROM."""
    conditions, params = translate_filter({"profile": {"lang": {"$ne": "en"}}})
    assert len(conditions) == 1
    assert "#>" in conditions[0]
    assert "IS DISTINCT FROM" in conditions[0]
    assert params == [["profile", "lang"], '"en"']


def test_value_column_validation():
    """value_column must be a valid identifier or alias.identifier."""
    # Valid single identifier
    translate_filter({"x": 1}, value_column="data")

    # Valid qualified identifier
    translate_filter({"x": 1}, value_column="i.value")

    # Invalid: too many dots
    with pytest.raises(ValueError, match="value_column"):
        translate_filter({"x": 1}, value_column="a.b.c")

    # Invalid: not an identifier
    with pytest.raises(ValueError, match="value_column"):
        translate_filter({"x": 1}, value_column="foo; DROP")
