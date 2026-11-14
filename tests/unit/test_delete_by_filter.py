"""Metadata deletion rejects empty predicates and binds all caller values."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CockroachDBVectorStore,
)

MODES = pytest.mark.parametrize("mode", ["sync", "async"])


def make_store(mode: str, namespace: str | None):
    engine = CockroachDBEngine.from_connection_string("cockroachdb://root@localhost:1/none")
    store_type = CockroachDBVectorStore if mode == "sync" else AsyncCockroachDBVectorStore
    store = store_type(
        engine,
        DeterministicFakeEmbedding(size=8),
        "documents",
        schema="custom",
        metadata_column="attributes",
        namespace_column="tenant",
        namespace=namespace,
    )
    return engine, store


async def delete(store, mode: str, filter):
    if mode == "sync":
        return store.delete_by_filter(filter)
    return await store.adelete_by_filter(filter)


@MODES
@pytest.mark.parametrize(
    "filter",
    [
        None,
        {},
        [],
        {"$and": [{}]},
        {"$or": [{"source": "keep"}, {}]},
        {"$not": {}},
        {"$and": []},
        {"source": {"$unsupported": 1}},
    ],
)
async def test_invalid_delete_filter_never_executes_sql(mode: str, filter, monkeypatch) -> None:
    engine, store = make_store(mode, None)
    execute = AsyncMock(return_value=[])
    monkeypatch.setattr(store, "_execute", execute)
    try:
        with pytest.raises(ValueError):
            await delete(store, mode, filter)
        execute.assert_not_called()
    finally:
        await engine.aclose()


@MODES
@pytest.mark.parametrize("namespace", [None, "", "tenant-a'; DELETE FROM documents; --"])
async def test_delete_filter_binds_metadata_and_namespace(
    mode: str, namespace: str | None, monkeypatch
) -> None:
    engine, store = make_store(mode, namespace)
    value = "O'Brien'); DELETE FROM documents; --"
    filter = {"$or": [{"source": value}, {"details": {"$eq": {}}}]}
    original = copy.deepcopy(filter)
    execute = AsyncMock(return_value=[])
    monkeypatch.setattr(store, "_execute", execute)
    try:
        assert await delete(store, mode, filter) is True
        execute.assert_awaited_once()
        (statement, params), options = execute.call_args
        sql = str(statement)
        assert sql.startswith("DELETE FROM custom.documents WHERE ")
        assert "attributes" in sql
        assert value not in sql
        assert json.dumps(value) in params.values()
        assert "{}" in params.values()
        assert options["begin"] is True
        if namespace is None:
            assert "tenant =" not in sql
        else:
            assert "tenant =" in sql
            assert namespace in params.values()
            assert namespace == "" or namespace not in sql
        assert filter == original
    finally:
        await engine.aclose()
