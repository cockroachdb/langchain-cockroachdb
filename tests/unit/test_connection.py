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


def test_gssencmode_is_added_when_missing() -> None:
    from langchain_cockroachdb._connection import with_libpq_defaults

    assert with_libpq_defaults(f"postgresql://{REST}", None, 10.0) == {
        "connect_timeout": 10,
        "gssencmode": "disable",
    }


def test_gssencmode_in_url_wins() -> None:
    from langchain_cockroachdb._connection import with_libpq_defaults

    url = f"postgresql://{REST}&gssencmode=prefer"
    assert with_libpq_defaults(url, None, 10.0) == {"connect_timeout": 10}


def test_gssencmode_in_options_wins() -> None:
    from langchain_cockroachdb._connection import with_libpq_defaults

    options = with_libpq_defaults(f"postgresql://{REST}", {"gssencmode": "require"}, 10.0)
    assert options["gssencmode"] == "require"


def test_gssencmode_added_for_key_value_dsn() -> None:
    from langchain_cockroachdb._connection import with_libpq_defaults

    options = with_libpq_defaults("host=localhost port=26257", None, 10.0)
    assert options["gssencmode"] == "disable"
    assert options["connect_timeout"] == 10


def test_engine_asyncpg_url_gets_no_gssencmode(monkeypatch) -> None:
    """CockroachDBEngine with asyncpg URL must not pass gssencmode to driver."""
    import contextlib

    import langchain_cockroachdb.engine as engine_module
    from langchain_cockroachdb import CockroachDBEngine

    captured = {}
    real = engine_module.create_async_engine

    def spy(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        try:
            return real(url, **kwargs)
        except ImportError:
            pass

    monkeypatch.setattr(engine_module, "create_async_engine", spy)
    with contextlib.suppress(ImportError):
        CockroachDBEngine.from_connection_string("cockroachdb+asyncpg://root@localhost:1/db")
    assert "connect_args" not in captured["kwargs"], "asyncpg must not get libpq options"
    url_query = captured["url"].split("?", 1)[1] if "?" in captured["url"] else ""
    assert "gssencmode" not in url_query, "asyncpg URL must not have gssencmode in query"


def test_key_value_dsn_with_gssencmode_keeps_it() -> None:
    """A key/value DSN that already sets gssencmode should not override it."""
    from langchain_cockroachdb._connection import with_libpq_defaults

    dsn = "host=localhost port=26257 dbname=defaultdb gssencmode=require"
    options = with_libpq_defaults(dsn, None, 10.0)
    # Should not add gssencmode to options when DSN already has it
    assert "gssencmode" not in options
    # But connect_timeout should still be added
    assert options["connect_timeout"] == 10
