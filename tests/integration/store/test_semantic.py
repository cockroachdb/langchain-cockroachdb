"""Semantic search on the store.

The write path: indexed items get one vector row per text, written in the
same transaction as the item. Re-puts replace the vectors, ``index=False``
and deletes remove them, and vectors carry the item's TTL.

The query path: ``search(query=...)`` ranks items by vector similarity
through the C-SPANN index and returns scores, and ``setup()`` checks an
existing index against the store's configuration.
"""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import PutOp, SearchItem
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


def assert_embeddings_match(rows: list[dict[str, Any]]) -> None:
    """Each row of TWO_MESSAGES holds the embedding of its own text, within float32 precision."""
    model = DeterministicFakeEmbedding(size=DIMS)
    texts = {"text": "hello", "messages[*].content.0": "first", "messages[*].content.1": "second"}
    assert sorted(row["field_name"] for row in rows) == sorted(texts)
    for row in rows:
        expected = model.embed_documents([texts[row["field_name"]]])[0]
        stored = parse_vector(row["embedding"])
        assert all(
            math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
            for a, b in zip(stored, expected, strict=True)
        )


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
    assert_embeddings_match(rows)
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
        # Stop at the vector upsert. The earlier statements of the batch ran
        # or, in pipeline mode, may still sit in the client buffer. Either way
        # nothing of the batch may commit
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
    cur = await aadmin.execute(f"SELECT field_name, embedding FROM {store.sql.vectors}")
    assert_embeddings_match(await cur.fetchall())


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


# The query path

DOCS = {
    "apple": "apples are red and sweet",
    "sky": "the sky is blue at noon",
    "cat": "cats purr when they are happy",
}


def put_docs(store: CockroachDBStore, namespace: tuple[str, ...] = ("docs",)) -> None:
    for key, text in DOCS.items():
        store.put(namespace, key, {"text": text})


async def aput_docs(store: AsyncCockroachDBStore, namespace: tuple[str, ...] = ("docs",)) -> None:
    for key, text in DOCS.items():
        await store.aput(namespace, key, {"text": text})


def assert_ranked_by_cosine(results: list[SearchItem]) -> None:
    """The item whose text is the query comes first with similarity 1, the others lower."""
    assert [item.key for item in results][0] == "sky"
    assert sorted(item.key for item in results) == sorted(DOCS)
    scores = [item.score for item in results]
    assert all(score is not None for score in scores)
    # Vectors are stored as float32
    assert math.isclose(scores[0], 1.0, abs_tol=1e-6)  # type: ignore[arg-type]
    assert all(score < scores[0] - 1e-3 for score in scores[1:])  # type: ignore[operator]
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


def plain_order(results: list[SearchItem]) -> list[str]:
    return [item.key for item in results]


def test_query_ranks_by_similarity_and_returns_scores(store_factory, mode) -> None:
    store = store_factory(mode, index=index_config())
    put_docs(store)
    results = store.search(("docs",), query=DOCS["sky"])
    assert_ranked_by_cosine(results)
    assert results[0].value == {"text": DOCS["sky"]}
    assert results[0].namespace == ("docs",)
    # The same query twice in one batch, and a search of the whole store
    root = store.search((), query=DOCS["sky"], limit=2)
    assert [item.key for item in root] == [item.key for item in results[:2]]


def test_query_includes_descendant_namespaces_and_applies_filters(store_factory, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(("users", "a"), "top", {"text": "dark theme", "kind": "pref"})
    store.put(("users", "a", "notes"), "deep", {"text": "dark theme", "kind": "note"})
    store.put(("users", "a", "notes", "x"), "deeper", {"text": "light theme", "kind": "note"})
    # Siblings that share the text prefix but not the namespace
    store.put(("users", "ab"), "sibling", {"text": "dark theme", "kind": "pref"})
    store.put(("users",), "parent", {"text": "dark theme", "kind": "pref"})

    results = store.search(("users", "a"), query="dark theme")
    assert {(item.namespace, item.key) for item in results} == {
        (("users", "a"), "top"),
        (("users", "a", "notes"), "deep"),
        (("users", "a", "notes", "x"), "deeper"),
    }
    assert {item.key for item in results[:2]} == {"top", "deep"}
    notes = store.search(("users", "a"), query="dark theme", filter={"kind": "note"})
    assert [item.key for item in notes] == ["deep", "deeper"]
    assert store.search(("users", "zzz"), query="dark theme") == []
    everything = store.search((), query="dark theme", limit=10)
    assert len(everything) == 5


def test_multi_vector_items_are_deduplicated_keeping_best_score(store_factory, mode) -> None:
    store = store_factory(mode, index=index_config())
    store.put(
        ("docs",),
        "many",
        {"text": "alpha", "messages": [{"content": "beta"}, {"content": "gamma"}]},
    )
    store.put(("docs",), "one", {"text": "delta"})
    results = store.search(("docs",), query="gamma")
    assert [item.key for item in results] == ["many", "one"]
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-6)  # type: ignore[arg-type]


def test_semantic_paging_applies_after_dedup(store_factory, mode) -> None:
    store = store_factory(mode, index=index_config())
    for key in ("k1", "k2", "k3", "k4"):
        store.put(
            ("docs",),
            key,
            {"text": f"{key} text", "messages": [{"content": f"{key} a"}, {"content": f"{key} b"}]},
        )
    full = store.search(("docs",), query="k2 a", limit=10)
    assert len(full) == 4 and full[0].key == "k2"
    pages = [store.search(("docs",), query="k2 a", limit=1, offset=i) for i in range(4)]
    assert [page[0].key for page in pages] == [item.key for item in full]
    assert store.search(("docs",), query="k2 a", limit=2, offset=3) == full[3:]


def test_deep_semantic_page_has_candidates(store_factory, mode) -> None:
    # One vector per item, so over_fetch without the offset would be 2 * 1 * 2 + 1 = 5
    store = store_factory(mode, index=index_config(fields=["text"]))
    for i in range(12):
        store.put(("docs",), f"k{i:02}", {"text": f"item number {i}"})
    full = store.search(("docs",), query="item number 3", limit=12)
    assert len(full) == 12
    deep = store.search(("docs",), query="item number 3", limit=2, offset=10)
    assert plain_order(deep) == plain_order(full[10:])


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("distance", ["l2", "inner_product"])
def test_l2_and_inner_product_scores(store_factory, table_name, admin, mode, distance) -> None:
    store = store_factory(
        mode, table_name=f"{table_name}_{distance}", index=index_config(distance_type=distance)
    )
    try:
        put_docs(store)
        results = store.search(("docs",), query=DOCS["sky"])
        model = DeterministicFakeEmbedding(size=DIMS)
        query = model.embed_query(DOCS["sky"])
        expected = {}
        for key, text in DOCS.items():
            vector = model.embed_query(text)
            if distance == "l2":
                # The score is the negated L2 distance
                expected[key] = -math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(vector, query, strict=True))
                )
            else:
                # CockroachDB's <#> is the negated inner product; the score negates it again
                expected[key] = dot(vector, query)
        assert [item.key for item in results] == sorted(expected, key=expected.get, reverse=True)
        for item in results:
            assert math.isclose(item.score, expected[item.key], rel_tol=1e-4, abs_tol=1e-4)  # type: ignore[arg-type]
        if distance == "l2":
            assert results[0].key == "sky" and abs(results[0].score) < 1e-3  # type: ignore[arg-type]
    finally:
        drop_store_tables(admin.info.dsn, f"{table_name}_{distance}")


def test_query_without_index_warns_once_and_searches_plainly(store_factory, mode, caplog) -> None:
    store = store_factory(mode)
    store.put(("docs",), "old", {"text": "the sky is blue"})
    store.put(("docs",), "new", {"text": "cats"})
    with caplog.at_level(logging.WARNING):
        first = store.search(("docs",), query="the sky is blue")
        second = store.search(("docs",), query="cats")
    warnings = [r for r in caplog.records if "no index configured" in r.getMessage()]
    assert len(warnings) == 1
    # Ordered by updated_at, newest first, with no score
    assert plain_order(first) == plain_order(second) == ["new", "old"]
    assert all(item.score is None for item in first)


def test_setup_rejects_distance_change_and_dims_change(store_factory, mode) -> None:
    store_factory(mode, index=index_config())
    l2 = store_factory(mode, setup=False, index=index_config(distance_type="l2"))
    with pytest.raises(ValueError) as raised:
        l2.setup()
    name = l2.sql.vector_index_name
    assert str(raised.value) == (
        f"the vector index {name} uses vector_cosine_ops; drop it with drop_vector_index() "
        "and run setup() again to use vector_l2_ops"
    )
    wider = store_factory(mode, setup=False, index={**index_config(), "dims": 8})
    with pytest.raises(ValueError, match=rf"VECTOR\({DIMS}\).*dims=8"):
        wider.setup()


def test_setup_warns_on_partition_size_change(store_factory, mode, caplog) -> None:
    store_factory(
        mode, index=index_config(cspann={"min_partition_size": 16, "max_partition_size": 64})
    )
    with caplog.at_level(logging.WARNING):
        store_factory(
            mode, index=index_config(cspann={"min_partition_size": 32, "max_partition_size": 128})
        )
    messages = [r.getMessage() for r in caplog.records if "partition_size" in r.getMessage()]
    assert len(messages) == 1
    assert "min_partition_size = 32 (index has 16)" in messages[0]
    assert "max_partition_size = 128 (index has 64)" in messages[0]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        store_factory(mode, index=index_config(flat=True))
    assert any("flat" in r.getMessage() for r in caplog.records)


def test_drop_vector_index_then_setup_rebuilds(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=index_config())
    put_docs(store)
    l2 = store_factory(mode, setup=False, index=index_config(distance_type="l2"))
    l2.drop_vector_index()
    assert "VECTOR INDEX" not in create_statement(admin, l2.sql.vectors)
    # Dropping twice is fine
    l2.drop_vector_index()
    l2.setup()
    assert "embedding vector_l2_ops)" in create_statement(admin, l2.sql.vectors)
    results = l2.search(("docs",), query=DOCS["sky"])
    assert results[0].key == "sky"
    assert abs(results[0].score) < 1e-3  # type: ignore[arg-type]


def beam_probe(**_: Any) -> tuple[str, list]:
    """A search statement that returns the session's beam size as the score.

    Its window of one vector is not full, so the store keeps its one row.
    """
    return (
        "SELECT 'docs' AS prefix, 'probe' AS key, CAST('{}' AS JSONB) AS value, "
        "now() AS created_at, now() AS updated_at, "
        "CAST(current_setting('vector_search_beam_size') AS FLOAT8) AS score, "
        "CAST(1 AS INT8) AS window_size",
        [],
    )


def test_beam_size_is_applied(store_factory, mode, monkeypatch) -> None:
    store = store_factory(mode, index=index_config(beam_size=7))
    put_docs(store)
    # The real search works with the beam size set
    assert store.search(("docs",), query=DOCS["sky"])[0].key == "sky"
    # The setting holds inside the search group, which runs in one transaction
    monkeypatch.setattr(store.sql, "semantic_search_in", beam_probe)
    probe = store.search(("docs",), query="anything")
    assert [(item.key, item.score) for item in probe] == [("probe", 7.0)]
    if mode != "pool":
        # SET LOCAL ends with the transaction; the session keeps the default
        row = store.conn.execute("SHOW vector_search_beam_size").fetchone()
        assert row is not None and row["vector_search_beam_size"] != "7"


# The query path, async


async def test_async_query_ranks_by_similarity_and_returns_scores(astore_factory, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    await aput_docs(store)
    results = await store.asearch(("docs",), query=DOCS["sky"])
    assert_ranked_by_cosine(results)
    # Two searches in one tick run as one batch, with one embedding per text
    same, root = await asyncio.gather(
        store.asearch(("docs",), query=DOCS["sky"]), store.asearch((), query=DOCS["sky"])
    )
    assert plain_order(same) == plain_order(root) == plain_order(results)


async def test_async_query_includes_descendant_namespaces_and_applies_filters(
    astore_factory, mode
) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(("users", "a"), "top", {"text": "dark theme", "kind": "pref"})
    await store.aput(("users", "a", "notes"), "deep", {"text": "dark theme", "kind": "note"})
    await store.aput(("users", "ab"), "sibling", {"text": "dark theme", "kind": "pref"})
    results = await store.asearch(("users", "a"), query="dark theme")
    assert sorted(item.key for item in results) == ["deep", "top"]
    notes = await store.asearch(("users", "a"), query="dark theme", filter={"kind": "note"})
    assert [item.key for item in notes] == ["deep"]


async def test_async_multi_vector_items_are_deduplicated_keeping_best_score(
    astore_factory, mode
) -> None:
    store = await astore_factory(mode, index=index_config())
    await store.aput(
        ("docs",),
        "many",
        {"text": "alpha", "messages": [{"content": "beta"}, {"content": "gamma"}]},
    )
    await store.aput(("docs",), "one", {"text": "delta"})
    results = await store.asearch(("docs",), query="gamma")
    assert [item.key for item in results] == ["many", "one"]
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-6)  # type: ignore[arg-type]


async def test_async_semantic_paging_applies_after_dedup(astore_factory, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    for key in ("k1", "k2", "k3"):
        await store.aput(
            ("docs",), key, {"text": f"{key} text", "messages": [{"content": f"{key} a"}]}
        )
    full = await store.asearch(("docs",), query="k3 a", limit=10)
    assert len(full) == 3 and full[0].key == "k3"
    pages = [await store.asearch(("docs",), query="k3 a", limit=1, offset=i) for i in range(3)]
    assert [page[0].key for page in pages] == plain_order(full)


async def test_async_deep_semantic_page_has_candidates(astore_factory, mode) -> None:
    store = await astore_factory(mode, index=index_config(fields=["text"]))
    for i in range(12):
        await store.aput(("docs",), f"k{i:02}", {"text": f"item number {i}"})
    full = await store.asearch(("docs",), query="item number 3", limit=12)
    assert len(full) == 12
    deep = await store.asearch(("docs",), query="item number 3", limit=2, offset=10)
    assert plain_order(deep) == plain_order(full[10:])


async def test_async_l2_and_inner_product_scores(astore_factory, mode) -> None:
    store = await astore_factory(mode, index=index_config(distance_type="l2"))
    await aput_docs(store)
    results = await store.asearch(("docs",), query=DOCS["sky"])
    assert results[0].key == "sky" and abs(results[0].score) < 1e-3  # type: ignore[arg-type]
    scores = [item.score for item in results]
    assert all(score <= 0 for score in scores)  # type: ignore[operator]
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


async def test_async_query_without_index_warns_once_and_searches_plainly(
    astore_factory, mode, caplog
) -> None:
    store = await astore_factory(mode)
    await store.aput(("docs",), "old", {"text": "the sky is blue"})
    await store.aput(("docs",), "new", {"text": "cats"})
    with caplog.at_level(logging.WARNING):
        first = await store.asearch(("docs",), query="the sky is blue")
        second = await store.asearch(("docs",), query="cats")
    assert len([r for r in caplog.records if "no index configured" in r.getMessage()]) == 1
    assert plain_order(first) == plain_order(second) == ["new", "old"]


async def test_async_setup_rejects_distance_change_and_dims_change(astore_factory, mode) -> None:
    await astore_factory(mode, index=index_config())
    ip = await astore_factory(mode, setup=False, index=index_config(distance_type="inner_product"))
    with pytest.raises(ValueError, match="uses vector_cosine_ops; drop it with drop_vector_index"):
        await ip.setup()
    wider = await astore_factory(mode, setup=False, index={**index_config(), "dims": 8})
    with pytest.raises(ValueError, match=r"dims=8"):
        await wider.setup()


async def test_async_setup_warns_on_partition_size_change(astore_factory, mode, caplog) -> None:
    await astore_factory(
        mode, index=index_config(cspann={"min_partition_size": 16, "max_partition_size": 64})
    )
    with caplog.at_level(logging.WARNING):
        await astore_factory(
            mode, index=index_config(cspann={"min_partition_size": 16, "max_partition_size": 96})
        )
    messages = [r.getMessage() for r in caplog.records if "partition_size" in r.getMessage()]
    assert len(messages) == 1 and "max_partition_size = 96 (index has 64)" in messages[0]
    assert "min_partition_size" not in messages[0]


async def test_async_drop_vector_index_then_setup_rebuilds(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=index_config())
    await aput_docs(store)
    l2 = await astore_factory(mode, setup=False, index=index_config(distance_type="l2"))
    await l2.adrop_vector_index()
    await l2.adrop_vector_index()
    await l2.setup()
    cur = await aadmin.execute(f"SHOW CREATE TABLE {l2.sql.vectors}")
    row = await cur.fetchone()
    assert row is not None and "embedding vector_l2_ops)" in row["create_statement"]
    assert (await l2.asearch(("docs",), query=DOCS["sky"]))[0].key == "sky"


async def test_async_beam_size_is_applied(astore_factory, mode, monkeypatch) -> None:
    store = await astore_factory(mode, index=index_config(beam_size=5))
    await aput_docs(store)
    assert (await store.asearch(("docs",), query=DOCS["sky"]))[0].key == "sky"
    monkeypatch.setattr(store.sql, "semantic_search_in", beam_probe)
    probe = await store.asearch(("docs",), query="anything")
    assert [(item.key, item.score) for item in probe] == [("probe", 5.0)]
    if mode != "pool":
        # SET LOCAL ends with the transaction; the session keeps the default
        cur = await store.conn.execute("SHOW vector_search_beam_size")
        row = await cur.fetchone()
        assert row is not None and row["vector_search_beam_size"] != "5"
