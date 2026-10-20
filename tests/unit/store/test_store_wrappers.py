"""AsyncCockroachDBStore construction and its inherited sync methods. No database."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import GetOp, SearchOp
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb import AsyncCockroachDBStore
from langchain_cockroachdb.store import AsyncCockroachDBStore as StoreExport


def test_construction_outside_a_running_loop_raises() -> None:
    pool = AsyncConnectionPool("", open=False)
    with pytest.raises(RuntimeError):
        AsyncCockroachDBStore(pool)


async def test_construction_inside_a_loop_with_a_pool() -> None:
    assert StoreExport is AsyncCockroachDBStore
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False), table_name="memories")
    assert store.supports_ttl is True
    assert store.loop is asyncio.get_running_loop()
    assert store.sql.migrations_table.endswith("memories_migrations")
    assert store.pipe is None


async def test_pool_with_pipeline_is_rejected_and_index_is_validated() -> None:
    pool = AsyncConnectionPool("", open=False)
    with pytest.raises(ValueError, match="Pipeline"):
        AsyncCockroachDBStore(pool, pipe=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embed"):
        AsyncCockroachDBStore(pool, index={"dims": 3, "embed": None})  # type: ignore[typeddict-item]
    store = AsyncCockroachDBStore(
        pool, index={"dims": 3, "embed": DeterministicFakeEmbedding(size=3)}
    )
    assert store.index is not None and store.index.dims == 3


async def test_sync_batch_from_the_stores_own_loop_raises() -> None:
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False))
    with pytest.raises(asyncio.InvalidStateError):
        store.batch([])
    with pytest.raises(asyncio.InvalidStateError):
        store.get(("a",), "k")


class RecordingAsyncCursor:
    """An async cursor that records executes and fetches in a shared log."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.connection = self

    def cursor(self, **kwargs: object) -> RecordingAsyncCursor:
        return RecordingAsyncCursor(self.log)

    async def execute(self, query: str, params: list) -> None:
        self.log.append("execute")

    async def fetchall(self) -> list:
        self.log.append("fetch")
        return []

    async def close(self) -> None:
        pass


async def test_read_group_queues_every_statement_before_fetching() -> None:
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False))
    log: list[str] = []

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[RecordingAsyncCursor]:
        log.append("cursor")
        yield RecordingAsyncCursor(log)

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    results = await store.abatch([GetOp(("a",), "k"), GetOp(("b",), "k"), GetOp(("c",), "k")])
    assert results == [None, None, None]
    # One cursor context, three statements, every fetch after every execute
    assert log == ["cursor"] + ["execute"] * 3 + ["fetch"] * 3

    log.clear()
    assert await store.abatch([SearchOp(("a",)), SearchOp(("b",), limit=0)]) == [[], []]
    assert log == ["cursor", "execute", "fetch"]
