"""Unit tests for namespace helpers."""

import pytest
from langgraph.store.base import InvalidNamespaceError

from langchain_cockroachdb.store.namespaces import (
    from_text,
    match_pattern,
    prefix_condition,
    to_text,
    truncate_expression,
    validate_labels,
)


def test_validate_labels():
    """Test label validation."""
    # Valid labels
    assert validate_labels(("a", "b")) == ("a", "b")

    # Label with dot
    with pytest.raises(InvalidNamespaceError):
        validate_labels(("a.b",))

    # Empty label
    with pytest.raises(InvalidNamespaceError):
        validate_labels(("",))

    # Non-string label
    with pytest.raises(InvalidNamespaceError):
        validate_labels((1,))  # type: ignore


def test_text_round_trip():
    """Test namespace to text and back."""
    assert to_text(("a", "b")) == "a.b"
    assert from_text("a.b") == ("a", "b")


def test_prefix_condition_empty_matches_all():
    """Test that empty prefix matches all namespaces."""
    sql, params = prefix_condition("namespace", ())
    assert sql == "TRUE"
    assert params == []


def test_prefix_condition_is_a_plain_range():
    """Labels with LIKE metacharacters give plain range bounds, with no escaping.

    The integration test test_search_scope_with_metacharacter_labels runs
    these against the database.
    """
    sql, params = prefix_condition("namespace", ("user_1",))
    assert sql == "(namespace = %s OR (namespace >= %s AND namespace < %s))"
    assert params == ["user_1", "user_1.", "user_1/"]
    assert "LIKE" not in sql

    _, params = prefix_condition("namespace", ("50%off", "x"))
    assert params == ["50%off.x", "50%off.x.", "50%off.x/"]

    _, params = prefix_condition("namespace", ("back\\slash",))
    assert params == ["back\\slash", "back\\slash.", "back\\slash/"]


def matches(namespace: str, prefix: tuple[str, ...]) -> bool:
    """The condition evaluated in Python, comparing bytes like CockroachDB."""
    _, (exact, low, high) = prefix_condition("namespace", prefix)
    value = namespace.encode()
    return value == exact.encode() or low.encode() <= value < high.encode()


def test_prefix_condition_matches_descendants_only():
    assert matches("a", ("a",))
    assert matches("a.b", ("a",))
    assert matches("a.b.c", ("a",))
    # Siblings that share the text prefix, and labels starting with bytes
    # just before and after "."
    for other in ("ab", "a-b", "a/b", "a0", "b", "", "a\u00e9"):
        assert not matches(other, ("a",)), other
    assert matches("a.\u00e9", ("a",))


def test_match_pattern_escapes_regex_metacharacters():
    """Test that regex metacharacters are escaped."""
    # Plus sign (regex metacharacter)
    pattern = match_pattern(("a+b",), "prefix")
    assert pattern == r"^a\+b(\.|\z)"

    # Wildcard in the middle
    pattern = match_pattern(("uid", "*", "alice"), "prefix")
    assert pattern == r"^uid\.[^.]+\.alice(\.|\z)"

    # Suffix pattern
    pattern = match_pattern(("v1",), "suffix")
    assert pattern == r"(^|\.)v1\z"


def test_truncate_expression():
    """Test SQL truncate expression generation."""
    # max_depth 2
    expr = truncate_expression("namespace", 2)
    assert "regexp_replace" in expr
    assert "{1}" in expr

    # max_depth 1
    expr = truncate_expression("namespace", 1)
    assert "{0}" in expr

    # Invalid max_depth
    with pytest.raises(ValueError):
        truncate_expression("namespace", 0)

    with pytest.raises(ValueError):
        truncate_expression("namespace", -1)
