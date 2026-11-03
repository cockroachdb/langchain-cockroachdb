"""Connection factories preserve self-hosted cluster-routing startup options.

A self-hosted virtual cluster is selected with the URL's `options=-ccluster=...`
startup option. These tests check that each connection factory passes that
option, and other startup options, through to psycopg unchanged. They inspect
driver arguments without opening a database connection. A managed Continuum
virtual cluster uses its own generated connection string instead, with no
`-ccluster` option; that path is covered by the opt-in integration suite in
tests/integration/test_virtual_cluster_deployment.py.
"""

from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from langchain_cockroachdb import (
    AsyncCockroachDBSaver,
    CockroachDBChatMessageHistory,
    CockroachDBEngine,
    CockroachDBSaver,
)
from langchain_cockroachdb.store import CockroachDBStore, PoolConfig

OPTIONS = "-ccluster=main -cstatement_timeout=5000"
REST = (
    "app:p%40ss%2Bword@localhost:26257/agents?sslmode=verify-full"
    "&options=-ccluster%3Dmain%20-cstatement_timeout%3D5000"
    "&application_name=agent%2Bresearch&connect_timeout=7"
)
SCHEMES = pytest.mark.parametrize("scheme", ["postgresql", "cockroachdb", "cockroachdb+psycopg"])


def assert_self_hosted_options(options: dict) -> None:
    assert options["options"] == OPTIONS
    assert options["dbname"] == "agents"
    assert options["user"] == "app"
    assert options["password"] == "p@ss+word"
    assert options["sslmode"] == "verify-full"
    assert options["application_name"] == "agent+research"
    assert str(options["connect_timeout"]) == "7"


@SCHEMES
@pytest.mark.parametrize("client", ["engine", "history"])
async def test_sqlalchemy_factories_preserve_self_hosted_cluster_options(
    scheme: str, client: str
) -> None:
    url = f"{scheme}://{REST}"
    history = None
    if client == "history":
        history = CockroachDBChatMessageHistory("conversation-1", connection_string=url)
        engine = history.engine
    else:
        engine = CockroachDBEngine.from_connection_string(url).engine
    try:
        _, options = engine.dialect.create_connect_args(engine.url)
        assert_self_hosted_options(options)
    finally:
        if history is not None:
            await history.aclose()
        else:
            await engine.dispose()


@SCHEMES
def test_sync_saver_passes_self_hosted_cluster_options_to_psycopg(
    scheme: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = MagicMock(spec=psycopg.Connection)
    conn.__enter__.return_value = conn
    connect = MagicMock(return_value=conn)
    monkeypatch.setattr(psycopg.Connection, "connect", connect)

    with CockroachDBSaver.from_conn_string(f"{scheme}://{REST}") as saver:
        assert saver.conn is conn

    connect.assert_called_once()
    args, kwargs = connect.call_args
    assert args == (f"postgresql://{REST}",)
    assert_self_hosted_options(conninfo_to_dict(args[0], **kwargs))
    assert "connect_timeout" not in kwargs
    assert kwargs["autocommit"] is True
    assert kwargs["row_factory"] is dict_row
    conn.__exit__.assert_called_once()


@SCHEMES
async def test_async_saver_passes_self_hosted_cluster_options_to_psycopg(
    scheme: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = MagicMock(spec=psycopg.AsyncConnection)
    conn.__aenter__.return_value = conn
    connect = AsyncMock(return_value=conn)
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)

    async with AsyncCockroachDBSaver.from_conn_string(f"{scheme}://{REST}") as saver:
        assert saver.conn is conn

    connect.assert_awaited_once()
    args, kwargs = connect.call_args
    assert args == (f"postgresql://{REST}",)
    assert_self_hosted_options(conninfo_to_dict(args[0], **kwargs))
    assert "connect_timeout" not in kwargs
    assert kwargs["autocommit"] is True
    assert kwargs["row_factory"] is dict_row
    conn.__aexit__.assert_awaited_once()


@SCHEMES
def test_store_normalizes_self_hosted_cluster_url(
    scheme: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = MagicMock(spec=psycopg.Connection)
    conn.__enter__.return_value = conn
    connect = MagicMock(return_value=conn)
    monkeypatch.setattr(psycopg.Connection, "connect", connect)

    with CockroachDBStore.from_conn_string(f"{scheme}://{REST}"):
        pass

    connect.assert_called_once()
    (url,), _ = connect.call_args
    assert url == f"postgresql://{REST}"
    assert conninfo_to_dict(url)["options"] == OPTIONS


@SCHEMES
def test_store_pool_config_normalizes_self_hosted_cluster_url(
    scheme: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    class FakePool:
        def __init__(self, conninfo: str, **kwargs: object) -> None:
            captured["conninfo"] = conninfo

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def wait(self, *a: object, **k: object) -> None:
            pass

        def open(self, *a: object, **k: object) -> None:
            pass

        def close(self, *a: object, **k: object) -> None:
            pass

    monkeypatch.setattr("langchain_cockroachdb.store.store.ConnectionPool", FakePool)

    with CockroachDBStore.from_conn_string(f"{scheme}://{REST}", pool_config=PoolConfig()):
        pass

    assert captured["conninfo"] == f"postgresql://{REST}"
    assert conninfo_to_dict(captured["conninfo"])["options"] == OPTIONS
