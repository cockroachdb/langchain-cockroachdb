"""Connection string helpers shared by the engine, chat history and savers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

from psycopg.conninfo import conninfo_to_dict

SQLALCHEMY_SCHEME = "cockroachdb+psycopg"

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
