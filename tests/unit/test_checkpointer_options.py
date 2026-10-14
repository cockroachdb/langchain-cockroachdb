"""Unit tests for checkpointer connection option helpers."""

import pytest
from psycopg.rows import dict_row

from langchain_cockroachdb.checkpointer.base import connection_options


def test_connection_options_adds_default_connect_timeout() -> None:
    """Default adds connect_timeout=10."""
    options = connection_options("postgresql://localhost/db", None, 10.0, 5)
    assert options["connect_timeout"] == 10
    assert options["autocommit"] is True
    assert options["row_factory"] is dict_row
    assert options["prepare_threshold"] == 5


def test_connection_options_url_connect_timeout_wins() -> None:
    """A URL connect_timeout wins."""
    options = connection_options("postgresql://localhost/db?connect_timeout=3", None, 10.0, 5)
    assert "connect_timeout" not in options
    assert options["autocommit"] is True


def test_connection_options_kwargs_connect_timeout_wins() -> None:
    """connect_kwargs connect_timeout wins."""
    options = connection_options("postgresql://localhost/db", {"connect_timeout": 5}, 10.0, 5)
    assert options["connect_timeout"] == 5
    assert options["autocommit"] is True


def test_connection_options_none_adds_none() -> None:
    """connect_timeout=None adds none."""
    options = connection_options("postgresql://localhost/db", None, None, 5)
    assert "connect_timeout" not in options
    assert options["autocommit"] is True


@pytest.mark.parametrize("key", ["autocommit", "row_factory", "prepare_threshold"])
def test_connection_options_rejects_managed_keys(key: str) -> None:
    """Each managed key raises ValueError."""
    with pytest.raises(ValueError, match=key):
        connection_options("postgresql://localhost/db", {key: "dummy"}, 10.0, 5)


def test_connection_options_adds_gssencmode_by_default() -> None:
    """Default adds gssencmode=disable."""
    options = connection_options("postgresql://localhost/db", None, 10.0, 5)
    assert options["gssencmode"] == "disable"


def test_connection_options_url_gssencmode_wins() -> None:
    """A URL gssencmode wins."""
    options = connection_options("postgresql://localhost/db?gssencmode=prefer", None, 10.0, 5)
    assert "gssencmode" not in options


def test_connection_options_kwargs_gssencmode_wins() -> None:
    """connect_kwargs gssencmode wins."""
    options = connection_options("postgresql://localhost/db", {"gssencmode": "require"}, 10.0, 5)
    assert options["gssencmode"] == "require"
