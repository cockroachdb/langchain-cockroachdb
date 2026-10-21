"""Semantic search applies filters and expiry before it cuts the candidates.

A search with a query must return what InMemoryStore returns: the best
items that match the filter and have not expired. The whole-branch review
found the store returning nothing in two cases, repeated here in every
search form:

- 60 items in ("mem",), 3 of them {"type": "rare"}, a filter on "rare" and
  limit=3. InMemoryStore returns all three.
- 40 expired items (ttl=0.001 minutes, not yet deleted) and 4 live ones in
  ("u",), limit=4. The live items are the answer.

The forms are the indexed search (C-SPANN), whose short first pass from a
full window runs again exactly; a flat store; a search of the whole store,
with prefix (); and the fallback for a lookup that finds more than
MAX_SEARCH_NAMESPACES namespaces, forced by setting it to 1. The fallback
cases add one item in a second namespace, ("mem", "x") or ("u", "x"), so
that the lookup finds more than one. That item does not change the answer.

Each query is chosen so that the nearest vectors, as many as the indexed
window holds, include no match. The first pass then finds nothing, which
is how the review saw it, and the second pass must find everything.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import PutOp, SearchItem
from langgraph.store.memory import InMemoryStore

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore
from langchain_cockroachdb.store.sql import NAMESPACES, StoreSQL, bind

DIMS = 16
FORMS = ("indexed", "flat", "root", "fallback")
MODEL = DeterministicFakeEmbedding(size=DIMS)

Item = tuple[tuple[str, ...], str, dict[str, Any]]


def index_config(**extra: Any) -> dict[str, Any]:
    return {
        "dims": DIMS,
        "embed": DeterministicFakeEmbedding(size=DIMS),
        "fields": ["text"],
        **extra,
    }


def reference(items: list[Item]) -> InMemoryStore:
    """InMemoryStore holding ``items``, embedded the same way."""
    store = InMemoryStore(index=index_config())  # type: ignore[arg-type]
    for namespace, key, value in items:
        store.put(namespace, key, value)
    return store


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def query_missing(items: list[Item], matches: set[str], nearest: int) -> str:
    """A query whose ``nearest`` closest items (by exact cosine) are none of ``matches``."""
    vectors = {key: MODEL.embed_query(value["text"]) for _, key, value in items}
    for n in range(2000):
        query = f"question {n}"
        vector = MODEL.embed_query(query)
        closest = sorted(vectors, key=lambda key: -cosine(vector, vectors[key]))[:nearest]
        if not matches & set(closest):
            return query
    raise AssertionError("no query keeps the matches out of the nearest items")


def rare_items() -> list[Item]:
    """The review's first case: 60 items in ("mem",), k07, k27 and k47 rare."""
    return [
        (("mem",), f"k{i:02}", {"text": f"memory {i}", "type": "rare" if i % 20 == 7 else "common"})
        for i in range(60)
    ]


RARE = {"k07", "k27", "k47"}
EXTRA_MEM: Item = (("mem", "x"), "extra", {"text": "another memory", "type": "common"})
EXPIRED = [(("u",), f"old{i:02}", {"text": f"old memory {i}"}) for i in range(40)]
LIVE = [(("u",), f"live{i}", {"text": f"live memory {i}"}) for i in range(4)]
EXTRA_EXPIRED: Item = (("u", "x"), "old_extra", {"text": "old memory extra"})


def form_kwargs(form: str) -> dict[str, Any]:
    """Store arguments for a search form."""
    return {"index": index_config(flat=form == "flat")}


def form_prefix(form: str, prefix: tuple[str, ...]) -> tuple[str, ...]:
    return () if form == "root" else prefix


def force_fallback(form: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if form == "fallback":
        monkeypatch.setattr("langchain_cockroachdb.store.base.MAX_SEARCH_NAMESPACES", 1)
        monkeypatch.setattr("langchain_cockroachdb.store.sql.MAX_SEARCH_NAMESPACES", 1)


def record_reads(store: CockroachDBStore) -> list[tuple[str, list[str]]]:
    """Record the label and SQL of every read group the store runs from now on."""
    reads: list[tuple[str, list[str]]] = []
    original = store._read_all

    def recording(statements: list[tuple[str, list]], label: str, session: Any = None) -> Any:
        reads.append((label, [sql for sql, _ in statements]))
        return original(statements, label, session)

    store._read_all = recording  # type: ignore[method-assign]
    return reads


def arecord_reads(store: AsyncCockroachDBStore) -> list[tuple[str, list[str]]]:
    reads: list[tuple[str, list[str]]] = []
    original = store._read_all

    async def recording(statements: list[tuple[str, list]], label: str, session: Any = None) -> Any:
        reads.append((label, [sql for sql, _ in statements]))
        return await original(statements, label, session)

    store._read_all = recording  # type: ignore[method-assign]
    return reads


def assert_form_ran(form: str, reads: list[tuple[str, list[str]]]) -> None:
    """The read groups each form runs, for a first pass that found nothing."""
    labels = [label for label, _ in reads]
    if form == "indexed":
        assert labels == ["store search namespaces", "store search", "store search exact"]
        assert "window_size" in reads[1][1][0]
        assert "GROUP BY i.prefix, i.key" in reads[2][1][0]
    elif form == "fallback":
        assert labels == ["store search namespaces", "store search"]
        [statement] = reads[1][1]
        assert "ANY(" not in statement and "GROUP BY i.prefix, i.key" in statement
    else:
        assert labels == ["store search"]
        assert "GROUP BY i.prefix, i.key" in reads[0][1][0]


def assert_same_results(mine: list[SearchItem], theirs: list[SearchItem]) -> None:
    """Same items in the same order, with scores equal within float32 precision."""
    assert [(item.namespace, item.key) for item in mine] == [
        (item.namespace, item.key) for item in theirs
    ]
    for a, b in zip(mine, theirs, strict=True):
        assert a.value == b.value
        assert math.isclose(a.score, b.score, abs_tol=1e-4)  # type: ignore[arg-type]
    scores = [item.score for item in mine]
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


def expire(store_put: Callable[[list[PutOp]], Any], items: list[Item]) -> None:
    store_put([PutOp(ns, key, value, ttl=0.001) for ns, key, value in items])


# Sync store


@pytest.mark.parametrize("form", FORMS)
def test_selective_filter_matches_in_memory_store(store_factory, mode, form, monkeypatch) -> None:
    items = rare_items() + ([EXTRA_MEM] if form == "fallback" else [])
    store = store_factory(mode, **form_kwargs(form))
    store.batch([PutOp(ns, key, value) for ns, key, value in items])
    force_fallback(form, monkeypatch)
    # over_fetch for limit=3 and one vector per item: 3 * 1 * 2 + 1 = 7
    query = query_missing(items, RARE, 7)
    prefix = form_prefix(form, ("mem",))
    reads = record_reads(store)
    mine = store.search(prefix, query=query, filter={"type": "rare"}, limit=3)
    theirs = reference(items).search(prefix, query=query, filter={"type": "rare"}, limit=3)
    assert {item.key for item in theirs} == RARE
    assert_same_results(mine, theirs)
    assert_form_ran(form, reads)


@pytest.mark.parametrize("form", FORMS)
def test_expired_items_do_not_crowd_out_live_ones(store_factory, mode, form, monkeypatch) -> None:
    expired = EXPIRED + ([EXTRA_EXPIRED] if form == "fallback" else [])
    store = store_factory(mode, **form_kwargs(form))
    expire(store.batch, expired)
    store.batch([PutOp(ns, key, value) for ns, key, value in LIVE])
    time.sleep(0.2)
    force_fallback(form, monkeypatch)
    # over_fetch for limit=4: 4 * 1 * 2 + 1 = 9
    query = query_missing(expired + LIVE, {key for _, key, _ in LIVE}, 9)
    prefix = form_prefix(form, ("u",))
    plain = store.search(prefix, limit=4)
    assert {item.key for item in plain} == {key for _, key, _ in LIVE}
    reads = record_reads(store)
    mine = store.search(prefix, query=query, limit=4)
    # The expired items are gone as far as a reader can tell
    theirs = reference(LIVE).search(prefix, query=query, limit=4)
    assert len(theirs) == 4
    assert_same_results(mine, theirs)
    assert_form_ran(form, reads)


def alice_items() -> list[Item]:
    """240 memories of alice, 10 of them rare, and 60 of bob, some rare too."""
    items: list[Item] = [
        (
            ("user", "alice", "memories"),
            f"a{i:03}",
            {"text": f"alice memory {i}", "type": "rare" if i % 24 == 5 else "common"},
        )
        for i in range(240)
    ]
    items += [
        (
            ("user", "bob", "memories"),
            f"b{i:02}",
            {"text": f"bob memory {i}", "type": "rare" if i % 6 == 0 else "common"},
        )
        for i in range(60)
    ]
    return items


def test_full_window_with_selective_filter_is_exact_and_ordered(store_factory, mode) -> None:
    items = alice_items()
    store = store_factory(mode, index=index_config())
    store.batch([PutOp(ns, key, value) for ns, key, value in items])
    rare = {key for _, key, value in items if value["type"] == "rare" and key.startswith("a")}
    # A window of (5 + 2) * 1 * 2 + 1 = 15; keep the rare items out of twice that
    query = query_missing([item for item in items if item[1].startswith("a")], rare, 30)
    reads = record_reads(store)
    mine = store.search(("user", "alice"), query=query, filter={"type": "rare"}, limit=5, offset=2)
    theirs = reference(items).search(
        ("user", "alice"), query=query, filter={"type": "rare"}, limit=5, offset=2
    )
    assert len(theirs) == 5
    assert_same_results(mine, theirs)
    assert [label for label, _ in reads] == [
        "store search namespaces",
        "store search",
        "store search exact",
    ]
    # The whole page, from the first item, is exact too
    full = store.search(("user", "alice"), query=query, filter={"type": "rare"}, limit=10)
    assert_same_results(
        full,
        reference(items).search(("user", "alice"), query=query, filter={"type": "rare"}, limit=10),
    )
    assert [item.key for item in full][2:7] == [item.key for item in mine]


def test_partial_window_is_exact_without_a_second_pass(store_factory, mode) -> None:
    items: list[Item] = [
        (("mem",), f"k{i}", {"text": f"memory {i}", "type": "rare" if i == 2 else "common"})
        for i in range(4)
    ]
    store = store_factory(mode, index=index_config())
    store.batch([PutOp(ns, key, value) for ns, key, value in items])
    reads = record_reads(store)
    # The window of 7 holds all 4 vectors, so filtering it is exact
    mine = store.search(("mem",), query="memory", filter={"type": "rare"}, limit=3)
    assert_same_results(
        mine, reference(items).search(("mem",), query="memory", filter={"type": "rare"}, limit=3)
    )
    assert [item.key for item in mine] == ["k2"]
    assert [label for label, _ in reads] == ["store search namespaces", "store search"]


def test_second_pass_refreshes_what_it_returns(store_factory, admin, mode) -> None:
    items = rare_items()
    store = store_factory(mode, index=index_config(), ttl={"default_ttl": 10})
    store.batch([PutOp(ns, key, value, ttl=10) for ns, key, value in items])
    query = query_missing(items, RARE, 7)

    def expiries() -> dict[str, Any]:
        rows = admin.execute(f"SELECT key, expires_at FROM {store.sql.items}").fetchall()
        return {row["key"]: row["expires_at"] for row in rows}

    before = expiries()
    time.sleep(0.05)
    reads = record_reads(store)
    mine = store.search(("mem",), query=query, filter={"type": "rare"}, limit=3)
    assert {item.key for item in mine} == RARE
    assert [label for label, _ in reads][-1] == "store search exact"
    after = expiries()
    assert all(after[key] > before[key] for key in RARE)
    assert all(after[key] == before[key] for key in before if key not in RARE)


def explain(admin: Any, statement: tuple[str, list]) -> str:
    rows = admin.execute("EXPLAIN " + statement[0], statement[1]).fetchall()
    return "\n".join(row["info"] for row in rows)


def test_only_the_indexed_form_uses_the_vector_index(store_factory, admin) -> None:
    """EXPLAIN: the indexed form searches the vector index, the exact forms never do.

    The window is small next to the namespace, so the optimizer prefers the
    index for the indexed form. The exact forms have no LIMIT it could
    serve, so they scan, even though the index exists.
    """
    items: list[Item] = [
        (
            ("user", "big"),
            f"k{i:03}",
            {"text": f"memory {i}", "type": "rare" if i % 25 == 3 else "c"},
        )
        for i in range(500)
    ]
    store = store_factory("pool", index=index_config())
    store.batch([PutOp(ns, key, value) for ns, key, value in items])
    admin.execute(f"ANALYZE {store.sql.items}")
    admin.execute(f"ANALYZE {store.sql.vectors}")
    query = MODEL.embed_query("a memory")
    common = {"filter": {"type": "rare"}, "limit": 3, "offset": 0, "omit_expired": True}

    def bound(statement: tuple[str, list]) -> tuple[str, list]:
        return StoreSQL.bind_query_vector(bind(statement, NAMESPACES, ["user.big"]), query)

    indexed = explain(admin, bound(store.sql.semantic_search_in(over_fetch=7, **common)))
    assert "vector search" in indexed, indexed
    assert store.sql.vector_index_name in indexed, indexed
    assert "window" in indexed, indexed
    for statement in (
        store.sql.semantic_search(namespace_prefix=("user",), **common),
        store.sql.semantic_search(namespace_prefix=(), **common),
        store.sql.exact_semantic_search_in(**common),
        store.sql.exact_semantic_search_in(**{**common, "filter": None}),
    ):
        plan = explain(admin, bound(statement))
        assert "vector search" not in plan, plan


# Async store


@pytest.mark.parametrize("form", FORMS)
async def test_async_selective_filter_matches_in_memory_store(
    astore_factory, mode, form, monkeypatch
) -> None:
    items = rare_items() + ([EXTRA_MEM] if form == "fallback" else [])
    store = await astore_factory(mode, **form_kwargs(form))
    await store.abatch([PutOp(ns, key, value) for ns, key, value in items])
    force_fallback(form, monkeypatch)
    query = query_missing(items, RARE, 7)
    prefix = form_prefix(form, ("mem",))
    reads = arecord_reads(store)
    mine = await store.asearch(prefix, query=query, filter={"type": "rare"}, limit=3)
    theirs = await reference(items).asearch(prefix, query=query, filter={"type": "rare"}, limit=3)
    assert {item.key for item in theirs} == RARE
    assert_same_results(mine, theirs)
    assert_form_ran(form, reads)


@pytest.mark.parametrize("form", FORMS)
async def test_async_expired_items_do_not_crowd_out_live_ones(
    astore_factory, mode, form, monkeypatch
) -> None:
    expired = EXPIRED + ([EXTRA_EXPIRED] if form == "fallback" else [])
    store = await astore_factory(mode, **form_kwargs(form))
    await store.abatch([PutOp(ns, key, value, ttl=0.001) for ns, key, value in expired])
    await store.abatch([PutOp(ns, key, value) for ns, key, value in LIVE])
    await asyncio.sleep(0.2)
    force_fallback(form, monkeypatch)
    query = query_missing(expired + LIVE, {key for _, key, _ in LIVE}, 9)
    prefix = form_prefix(form, ("u",))
    plain = await store.asearch(prefix, limit=4)
    assert {item.key for item in plain} == {key for _, key, _ in LIVE}
    reads = arecord_reads(store)
    mine = await store.asearch(prefix, query=query, limit=4)
    theirs = await reference(LIVE).asearch(prefix, query=query, limit=4)
    assert len(theirs) == 4
    assert_same_results(mine, theirs)
    assert_form_ran(form, reads)


async def test_async_full_window_with_selective_filter_is_exact_and_ordered(
    astore_factory, mode
) -> None:
    items = alice_items()
    store = await astore_factory(mode, index=index_config())
    await store.abatch([PutOp(ns, key, value) for ns, key, value in items])
    rare = {key for _, key, value in items if value["type"] == "rare" and key.startswith("a")}
    query = query_missing([item for item in items if item[1].startswith("a")], rare, 30)
    reads = arecord_reads(store)
    mine = await store.asearch(
        ("user", "alice"), query=query, filter={"type": "rare"}, limit=5, offset=2
    )
    theirs = await reference(items).asearch(
        ("user", "alice"), query=query, filter={"type": "rare"}, limit=5, offset=2
    )
    assert len(theirs) == 5
    assert_same_results(mine, theirs)
    assert [label for label, _ in reads] == [
        "store search namespaces",
        "store search",
        "store search exact",
    ]
