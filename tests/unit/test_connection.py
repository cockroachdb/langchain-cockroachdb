"""Unit tests for connection string helpers."""

import pytest

from langchain_cockroachdb._connection import to_libpq_url, to_sqlalchemy_url, with_connect_timeout

REST = "root@localhost:26257/defaultdb?sslmode=disable"


@pytest.mark.parametrize(
    "scheme",
    [
        "cockroachdb",
        "cockroachdb+psycopg",
        "cockroachdb+psycopg2",
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
        "postgres",
        "postgres+psycopg",
        "postgres+psycopg2",
    ],
)
def test_every_spelling_maps_to_the_psycopg3_dialect(scheme: str) -> None:
    assert to_sqlalchemy_url(f"{scheme}://{REST}") == f"cockroachdb+psycopg://{REST}"
    assert to_libpq_url(f"{scheme}://{REST}") == f"postgresql://{REST}"


def test_asyncpg_and_unknown_schemes_pass_through() -> None:
    assert to_sqlalchemy_url(f"cockroachdb+asyncpg://{REST}") == f"cockroachdb+asyncpg://{REST}"
    assert to_sqlalchemy_url(f"custom+driver://{REST}") == f"custom+driver://{REST}"


def test_key_value_dsn_passes_through_libpq() -> None:
    assert to_libpq_url("host=localhost port=26257") == "host=localhost port=26257"


def test_connect_timeout_is_added_when_missing() -> None:
    assert with_connect_timeout(f"postgresql://{REST}", None, 10.0) == {"connect_timeout": 10}
    assert with_connect_timeout(f"postgresql://{REST}", {"a": 1}, 2.5) == {
        "a": 1,
        "connect_timeout": 3,
    }


def test_existing_connect_timeout_wins() -> None:
    url = f"postgresql://{REST}&connect_timeout=4"
    assert with_connect_timeout(url, None, 10.0) == {}
    assert with_connect_timeout(f"postgresql://{REST}", {"connect_timeout": 7}, 10.0) == {
        "connect_timeout": 7
    }
    assert with_connect_timeout("host=h connect_timeout=5", None, 10.0) == {}


def test_none_disables_and_options_are_copied() -> None:
    options = {"application_name": "x"}
    merged = with_connect_timeout(f"postgresql://{REST}", options, None)
    assert merged == options and merged is not options


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        with_connect_timeout(f"postgresql://{REST}", None, 0)
