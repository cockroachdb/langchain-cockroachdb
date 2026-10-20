"""Connection string helpers shared by the engine, chat history and savers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

SQLALCHEMY_SCHEME = "cockroachdb+psycopg"

# CockroachDB has no GSS encryption; disable it to avoid Kerberos lookups on macOS that crash forked workers
DEFAULT_GSSENCMODE = "disable"

# Spellings already found in env files and compose files. All of them land on
# the CockroachDB dialect with the psycopg 3 driver.
_KNOWN_SCHEMES = frozenset(
    {
        "cockroachdb",
        "cockroachdb+psycopg",
        "cockroachdb+psycopg2",
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
        "postgres",
        "postgres+psycopg",
        "postgres+psycopg2",
        "cockroachdb+asyncpg",
    }
)


def _split(url: str) -> tuple[str, str]:
    scheme, _, rest = url.partition("://")
    return scheme.lower(), rest


def to_sqlalchemy_url(url: str) -> str:
    """Return the SQLAlchemy URL for ``url``: the CockroachDB dialect on psycopg 3.

    ``cockroachdb+asyncpg://`` and unknown schemes pass through unchanged.
    """
    scheme, rest = _split(url)
    if scheme not in _KNOWN_SCHEMES or scheme == "cockroachdb+asyncpg":
        return url
    return f"{SQLALCHEMY_SCHEME}://{rest}"


def to_libpq_url(url: str) -> str:
    """Return a ``postgresql://`` URL for code that opens psycopg connections itself.

    Key/value DSNs such as ``host=localhost port=26257`` and unknown schemes
    pass through unchanged.
    """
    if "://" not in url:
        return url
    scheme, rest = _split(url)
    return f"postgresql://{rest}" if scheme in _KNOWN_SCHEMES else url


def _url_sets_connect_timeout(url: str) -> bool:
    if "://" in url:
        return "connect_timeout" in parse_qs(urlsplit(url).query)
    return "connect_timeout" in conninfo_to_dict(url)


def _url_sets_gssencmode(url: str) -> bool:
    if "://" in url:
        return "gssencmode" in parse_qs(urlsplit(url).query)
    return "gssencmode" in conninfo_to_dict(url)


def with_connect_timeout(
    url: str, options: Mapping[str, Any] | None, connect_timeout: float | None
) -> dict[str, Any]:
    """Return connection options with ``connect_timeout`` added when nothing sets it.

    A timeout in the URL or in ``options`` wins. libpq takes whole seconds, so
    the value is rounded up.

    Args:
        url: Connection URL or key/value DSN
        options: Existing connection options (connect_args), or None
        connect_timeout: Seconds, or None to leave the timeout unset

    Returns:
        A new dict of connection options
    """
    merged = dict(options or {})
    if connect_timeout is None or "connect_timeout" in merged or _url_sets_connect_timeout(url):
        return merged
    if connect_timeout <= 0:
        raise ValueError("connect_timeout must be positive or None")
    merged["connect_timeout"] = max(1, math.ceil(connect_timeout))
    return merged


def with_libpq_defaults(
    url: str, options: Mapping[str, Any] | None, connect_timeout: float | None
) -> dict[str, Any]:
    """Return connection options with libpq defaults added when nothing sets them.

    Values in the URL or in ``options`` always win.

    Args:
        url: Connection URL or key/value DSN
        options: Existing connection options (connect_args), or None
        connect_timeout: Seconds, or None to leave the timeout unset

    Returns:
        A new dict of connection options
    """
    merged = with_connect_timeout(url, options, connect_timeout)
    if "gssencmode" not in merged and not _url_sets_gssencmode(url):
        merged["gssencmode"] = DEFAULT_GSSENCMODE
    return merged


# The psycopg components (checkpointers, store) depend on these connection settings
_MANAGED_OPTIONS = frozenset({"autocommit", "row_factory", "prepare_threshold"})


def connection_options(
    conn_string: str,
    connect_kwargs: Mapping[str, Any] | None,
    connect_timeout: float | None,
    prepare_threshold: int | None,
    *,
    component: str = "checkpointer",
) -> dict[str, Any]:
    """Keyword arguments for psycopg's connect(), shared by the psycopg components.

    ``component`` names the caller in the error message.

    Raises:
        ValueError: If connect_kwargs sets an option the component manages
    """
    managed = _MANAGED_OPTIONS & set(connect_kwargs or {})
    if managed:
        raise ValueError(
            f"{', '.join(sorted(managed))} cannot go in connect_kwargs: the {component} sets "
            "autocommit and row_factory itself, and prepare_threshold has its own argument"
        )
    options = with_libpq_defaults(conn_string, connect_kwargs, connect_timeout)
    options.update(autocommit=True, row_factory=dict_row, prepare_threshold=prepare_threshold)
    return options
