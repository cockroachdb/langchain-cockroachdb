"""Unit tests for SQL helpers."""

import pytest

from langchain_cockroachdb._sql import parse_vector, validate_identifier, vector_literal


@pytest.mark.parametrize("name", ["documents", "_t", "Docs2", "public"])
def test_valid_identifiers(name: str) -> None:
    assert validate_identifier(name, what="table name") == name


@pytest.mark.parametrize(
    "name", ["", "2docs", "my-docs", "a b", "t;drop", 'x"y', "a.b", "docs\n", None]
)
def test_invalid_identifiers(name) -> None:
    with pytest.raises(ValueError, match="table name"):
        validate_identifier(name, what="table name")


def test_vector_literal_round_trip() -> None:
    literal = vector_literal([0.1, 2, -3.5])
    assert literal == "[0.1,2.0,-3.5]"
    assert parse_vector(literal) == [0.1, 2.0, -3.5]


def test_parse_vector_accepts_sequences() -> None:
    assert parse_vector((1, 2)) == [1.0, 2.0]
