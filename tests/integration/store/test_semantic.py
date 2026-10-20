"""Semantic search on the store: the write path.

Indexed items get one vector row per text, written in the same transaction
as the item. Re-puts replace the vectors, ``index=False`` and deletes remove
them, and vectors carry the item's TTL.
"""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import PutOp
from psycopg import AsyncConnection, Connection

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore
from langchain_cockroachdb._sql import parse_vector
from tests.integration.faults import Interruption
from tests.integration.store.conftest import FAST, drop_store_tables

DIMS = 16
FIELDS = ["text", "messages[*].content"]
TWO_MESSAGES = {"text": "hello", "messages": [{"content": "first"}, {"content": "second"}]}


def index_config(**extra: Any) -> dict[str, Any]:
    return {"dims": DIMS, "embed": DeterministicFakeEmbedding(size=DIMS), "fields": FIELDS, **extra}


def create_statement(admin: Connection[Any], table: str) -> str:
    row = admin.execute(f"SHOW CREATE TABLE {table}").fetchone()
    assert row is not None
    return row["create_statement"]


def vector_fields(admin: Connection[Any], store: Any) -> list[tuple[str, str, str]]:
    """The (prefix, key, field_name) of every vector row, sorted."""
    rows = admin.execute(
        f"SELECT prefix, key, field_name FROM {store.sql.vectors} ORDER BY 1, 2, 3"
    ).fetchall()
    return [(row["prefix"], row["key"], row["field_name"]) for row in rows]


async def avector_fields(aadmin: AsyncConnection[Any], store: Any) -> list[tuple[str, str, str]]:
    cur = await aadmin.execute(
        f"SELECT prefix, key, field_name FROM {store.sql.vectors} ORDER BY 1, 2, 3"
    )
    return [(row["prefix"], row["key"], row["field_name"]) for row in await cur.fetchall()]


def assert_index(statement: str, *, options: str | None) -> None:
    assert "VECTOR INDEX" in statement
    assert "(prefix, embedding vector_cosine_ops)" in statement
    if options is not None:
        assert options in statement


CSPANN_OPTIONS = "WITH (min_partition_size=16, max_partition_size=64)"


# Sync store


def test_setup_creates_vectors_table_and_index(store_factory, admin, table_name, mode) -> None:
    store = store_factory(mode, index=index_config())
    statement = create_statement(admin, store.sql.vectors)
    assert f"VECTOR({DIMS})" in statement
    assert_index(statement, options=None)
    # setup() runs the vector statements every time, and they are idempotent
    store.setup()
    assert create_statement(admin, store.sql.vectors) == statement

    sized, flat = f"{table_name}_sized", f"{table_name}_flat"
    try:
        store = store_factory(
            mode,
            table_name=sized,
            index=index_config(cspann={"min_partition_size": 16, "max_partition_size": 64}),
        )
        assert_index(create_statement(admin, store.sql.vectors), options=CSPANN_OPTIONS)
        store = store_factory(mode, table_name=flat, index=index_config(flat=True))
        statement = create_statement(admin, store.sql.vectors)
        assert f"VECTOR({DIMS})" in statement
        assert "VECTOR INDEX" not in statement
    finally:
        drop_store_tables(admin.info.dsn, sized)
        drop_store_tables(admin.info.dsn, flat)


def test_put_writes_one_vector_per_field_text(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(("docs", "a"), "k1", TWO_MESSAGES)
    assert vector_fields(admin, store) == [
        ("docs.a", "k1", "messages[*].content.0"),
        ("docs.a", "k1", "messages[*].content.1"),
        ("docs.a", "k1", "text"),
    ]
    # Each row holds the embedding of its own text
    rows = admin.execute(f"SELECT field_name, embedding FROM {store.sql.vectors}").fetchall()
    model = DeterministicFakeEmbedding(size=DIMS)
    texts = {"text": "hello", "messages[*].content.0": "first", "messages[*].content.1": "second"}
    for row in rows:
        expected = model.embed_documents([texts[row["field_name"]]])[0]
        stored = parse_vector(row["embedding"])
        assert all(
            math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
            for a, b in zip(stored, expected, strict=True)
        )
    # A put that names its own fields embeds those instead
    store.put(("docs", "a"), "k2", {"title": "t", "text": "x"}, index=["title"])
    assert ("docs.a", "k2", "title") in vector_fields(admin, store)
    assert ("docs.a", "k2", "text") not in vector_fields(admin, store)


def test_reput_with_fewer_fields_removes_stale_vectors(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(("docs",), "k1", TWO_MESSAGES)
    store.put(("docs",), "k2", TWO_MESSAGES)
    store.put(("docs",), "k1", {"text": "hello", "messages": [{"content": "only"}]})
    store.put(("docs",), "k2", {"text": "just text"})
    assert vector_fields(admin, store) == [
        ("docs", "k1", "messages[*].content"),
        ("docs", "k1", "text"),
        ("docs", "k2", "text"),
    ]


def test_put_index_false_removes_vectors(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(("docs",), "k1", TWO_MESSAGES)
    store.put(("docs",), "k2", {"text": "kept"})
    store.put(("docs",), "k1", {"text": "not indexed"}, index=False)
    assert vector_fields(admin, store) == [("docs", "k2", "text")]
    item = store.get(("docs",), "k1")
    assert item is not None and item.value == {"text": "not indexed"}


def test_delete_removes_vectors(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(("docs",), "k1", TWO_MESSAGES)
    store.put(("docs",), "k2", {"text": "kept"})
    store.put(("other",), "k1", {"text": "other namespace"})
    store.delete(("docs",), "k1")
    assert vector_fields(admin, store) == [("docs", "k2", "text"), ("other", "k1", "text")]
    # A put and a delete of the same key in one batch: the last op wins
    store.batch([PutOp(("docs",), "k2", {"text": "again"}), PutOp(("docs",), "k2", None)])
    assert vector_fields(admin, store) == [("other", "k1", "text")]


def assert_ttl_mirrored(rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> None:
    assert len(items) == 1 and items[0]["expires_at"] is not None
    assert len(rows) == 3
    for row in rows:
        assert float(row["ttl_minutes"]) == 5.0
        assert abs((row["expires_at"] - items[0]["expires_at"]).total_seconds()) < 1


def test_vectors_carry_the_items_ttl(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(("docs",), "k1", TWO_MESSAGES, ttl=5)
    items = admin.execute(f"SELECT expires_at FROM {store.sql.items}").fetchall()
    rows = admin.execute(f"SELECT expires_at, ttl_minutes FROM {store.sql.vectors}").fetchall()
    assert_ttl_mirrored(rows, items)


def stores_with_interruption(
    libpq_url: str, mode: str, table_name: str, cursor_factory: Any, *, sync: bool
) -> AbstractContextManager[CockroachDBStore] | AbstractAsyncContextManager[AsyncCockroachDBStore]:
    """A store whose connections use ``cursor_factory``, in the given mode."""
    kwargs: dict[str, Any] = {
        "index": index_config(),
        "retry_policy": FAST,
        "table_name": table_name,
    }
    if mode == "pool":
        kwargs["pool_config"] = {
            "min_size": 1,
            "max_size": 1,
            "kwargs": {"cursor_factory": cursor_factory},
        }
    else:
        kwargs["pipeline"] = mode == "pipeline"
        kwargs["connect_kwargs"] = {"cursor_factory": cursor_factory}
    cls = CockroachDBStore if sync else AsyncCockroachDBStore
    return cls.from_conn_string(libpq_url, **kwargs)


def batch_ops(text: str) -> list[PutOp]:
    return [
        PutOp(("docs",), "k1", {"text": text}),
        PutOp(("docs",), "k2", {"text": text}),
        PutOp(("docs",), "gone", None),
    ]


BEFORE = [
    ("docs", "gone", "text"),
    ("docs", "k1", "messages[*].content.0"),
    ("docs", "k1", "messages[*].content.1"),
    ("docs", "k1", "text"),
]
AFTER = [("docs", "k1", "text"), ("docs", "k2", "text")]


def test_put_batch_with_vectors_is_atomic_when_interrupted(
    libpq_url, admin, table_name, mode
) -> None:
    interruption = Interruption("insert into ")
    with stores_with_interruption(
        libpq_url, mode, table_name, interruption.cursor_class(), sync=True
    ) as store:
        assert isinstance(store, CockroachDBStore)
        store.setup()
        store.put(("docs",), "k1", TWO_MESSAGES)
        store.put(("docs",), "gone", {"text": "bye"})
        # Stop at the vector upsert, after the old vectors and the items were sent
        interruption.match = f"insert into {store.sql.vectors} ".lower()
        interruption.armed = True
        with pytest.raises(RuntimeError, match="interrupted"):
            store.batch(batch_ops("new"))
        assert vector_fields(admin, store) == BEFORE
        item = store.get(("docs",), "k1")
        assert item is not None and item.value == TWO_MESSAGES
        assert store.get(("docs",), "k2") is None
        store.batch(batch_ops("new"))
        assert vector_fields(admin, store) == AFTER


# Async store


async def test_async_setup_creates_vectors_table_and_index(
    astore_factory, aadmin, table_name, libpq_url, mode
) -> None:
    async def statement_of(store: AsyncCockroachDBStore) -> str:
        cur = await aadmin.execute(f"SHOW CREATE TABLE {store.sql.vectors}")
        row = await cur.fetchone()
        assert row is not None
        return row["create_statement"]

    store = await astore_factory(mode, index=index_config())
    statement = await statement_of(store)
    assert_index(statement, options=None)
    await store.setup()
    assert await statement_of(store) == statement

    sized, flat = f"{table_name}_sized", f"{table_name}_flat"
    try:
        store = await astore_factory(
            mode,
            table_name=sized,
            index=index_config(cspann={"min_partition_size": 16, "max_partition_size": 64}),
        )
        assert_index(await statement_of(store), options=CSPANN_OPTIONS)
        store = await astore_factory(mode, table_name=flat, index=index_config(flat=True))
        assert "VECTOR INDEX" not in await statement_of(store)
    finally:
        await asyncio.to_thread(drop_store_tables, libpq_url, sized)
        await asyncio.to_thread(drop_store_tables, libpq_url, flat)


async def test_async_put_writes_one_vector_per_field_text(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(("docs", "a"), "k1", TWO_MESSAGES)
    assert await avector_fields(aadmin, store) == [
        ("docs.a", "k1", "messages[*].content.0"),
        ("docs.a", "k1", "messages[*].content.1"),
        ("docs.a", "k1", "text"),
    ]


async def test_async_reput_with_fewer_fields_removes_stale_vectors(
    astore_factory, aadmin, mode
) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(("docs",), "k1", TWO_MESSAGES)
    await store.aput(("docs",), "k1", {"text": "hello", "messages": [{"content": "only"}]})
    assert await avector_fields(aadmin, store) == [
        ("docs", "k1", "messages[*].content"),
        ("docs", "k1", "text"),
    ]


async def test_async_put_index_false_removes_vectors(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(("docs",), "k1", TWO_MESSAGES)
    await store.aput(("docs",), "k1", {"text": "not indexed"}, index=False)
    assert await avector_fields(aadmin, store) == []
    item = await store.aget(("docs",), "k1")
    assert item is not None and item.value == {"text": "not indexed"}


async def test_async_delete_removes_vectors(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(("docs",), "k1", TWO_MESSAGES)
    await store.aput(("docs",), "k2", {"text": "kept"})
    await store.adelete(("docs",), "k1")
    assert await avector_fields(aadmin, store) == [("docs", "k2", "text")]


async def test_async_vectors_carry_the_items_ttl(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(("docs",), "k1", TWO_MESSAGES, ttl=5)
    items = await (await aadmin.execute(f"SELECT expires_at FROM {store.sql.items}")).fetchall()
    cur = await aadmin.execute(f"SELECT expires_at, ttl_minutes FROM {store.sql.vectors}")
    assert_ttl_mirrored(await cur.fetchall(), items)


async def test_put_batch_with_vectors_is_atomic_under_cancellation(
    libpq_url, aadmin, table_name, caplog, mode
) -> None:
    # The cancellation lands between BEGIN and COMMIT, with earlier statements sent
    interruption = Interruption("insert into ")
    async with stores_with_interruption(
        libpq_url, mode, table_name, interruption.async_cursor_class(), sync=False
    ) as store:
        assert isinstance(store, AsyncCockroachDBStore)
        await store.setup()
        await store.aput(("docs",), "k1", TWO_MESSAGES)
        await store.aput(("docs",), "gone", {"text": "bye"})
        interruption.match = f"insert into {store.sql.vectors} ".lower()
        interruption.reached = asyncio.Event()
        interruption.armed = True
        with caplog.at_level(logging.WARNING):
            task = asyncio.create_task(store.abatch(batch_ops("new")))
            await asyncio.wait_for(interruption.reached.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert await avector_fields(aadmin, store) == BEFORE
        item = await store.aget(("docs",), "k1")
        assert item is not None and item.value == TWO_MESSAGES
        assert await store.aget(("docs",), "k2") is None
        # The store rolled the stopped transaction back itself
        assert not any("rolling back returned connection" in r.getMessage() for r in caplog.records)
        await store.abatch(batch_ops("new"))
        assert await avector_fields(aadmin, store) == AFTER
