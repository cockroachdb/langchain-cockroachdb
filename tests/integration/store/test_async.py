"""AsyncCockroachDBStore matches LangGraph's InMemoryStore, and batches like AsyncPostgresStore.

The parity tests are the async form of those in test_parity.py and use the
same op lists. Search results are compared as sets for the reason given
there: InMemoryStore returns insertion order, the store the most recently
updated first.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import pytest
from langgraph.store.base import BaseStore, GetOp, InvalidNamespaceError, PutOp
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb import AsyncCockroachDBStore
from tests.integration.store.test_parity import (
    FILTER_CASES,
    FRUIT,
    GET_OPS,
    LIST_CASES,
    MIXED_OPS,
    NAMESPACES,
    PREFIX_CASES,
    PREFIX_ITEMS,
    UNICODE_VALUE,
    WAIT,
    as_set,
    norm,
)


async def both(stores: list[BaseStore], ops: list) -> list[list]:
    """Run ``ops`` on every store with abatch; return the normalized results of each."""
    return [norm(await store.abatch(ops)) for store in stores]


async def test_async_parity_put_get_delete(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        await s.aput(("docs", "a"), "k1", {"text": "one"})
        await s.aput(("docs", "a"), "k2", {"text": "two"})
        await s.aput(("docs", "b"), "k1", {"text": "other"})
        await s.aput(("docs", "a"), "k1", {"text": "one again", "n": 1})
        await s.adelete(("docs", "a"), "k2")
    mine, theirs = await both([store, ref], GET_OPS)
    assert mine == theirs
    assert mine[0] == (("docs", "a"), "k1", {"text": "one again", "n": 1})
    assert mine[1] is None
    item = await store.aget(("docs", "a"), "k1")
    assert item is not None
    assert item.created_at <= item.updated_at


async def test_async_parity_search_prefix_and_descendants(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        for namespace, key, value in PREFIX_ITEMS:
            await s.aput(namespace, key, value)
    for prefix, expected_keys in PREFIX_CASES:
        mine, theirs = await store.asearch(prefix), await ref.asearch(prefix)
        assert as_set(mine) == as_set(theirs), prefix
        assert {item.key for item in mine} == expected_keys, prefix
        assert all(item.score is None for item in mine)


async def test_async_parity_filters(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        for key, value in FRUIT.items():
            await s.aput(("fruit",), key, value)
    for filter, expected in FILTER_CASES:
        mine = await store.asearch(("fruit",), filter=filter)
        theirs = await ref.asearch(("fruit",), filter=filter)
        assert as_set(mine) == as_set(theirs), filter
        assert {item.key for item in mine} == expected, filter


async def test_async_parity_ordering_and_paging(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        for i in range(4):
            await s.aput(("log",), f"k{i}", {"i": i})
        await s.abatch([PutOp(("log", "z"), "b", {"i": 9}), PutOp(("log", "y"), "a", {"i": 9})])
        await s.aput(("log",), "k1", {"i": 1, "again": True})

    full = await store.asearch(("log",), limit=10)
    assert [(item.namespace, item.key) for item in full] == [
        (("log",), "k1"),
        (("log", "y"), "a"),
        (("log", "z"), "b"),
        (("log",), "k3"),
        (("log",), "k2"),
        (("log",), "k0"),
    ]
    assert as_set(full) == as_set(await ref.asearch(("log",), limit=10))
    for limit, offset in [(2, 0), (2, 1), (3, 3), (1, 5), (10, 0)]:
        page = await store.asearch(("log",), limit=limit, offset=offset)
        assert norm(page) == norm(full[offset : offset + limit]), (limit, offset)
    assert await store.asearch(("log",), limit=0) == []


async def test_async_paging_edges(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        await s.aput(("p",), "k", {"v": 1})
    with pytest.raises(ValueError, match="limit"):
        await store.asearch(("p",), limit=-1)
    with pytest.raises(ValueError, match="offset"):
        await store.asearch(("p",), offset=-1)
    with pytest.raises(ValueError, match="limit"):
        await store.alist_namespaces(limit=-1)
    with pytest.raises(ValueError, match="offset"):
        await store.alist_namespaces(offset=-2)
    assert await store.asearch(("p",), offset=5) == await ref.asearch(("p",), offset=5) == []
    assert await store.alist_namespaces(offset=5) == await ref.alist_namespaces(offset=5) == []


async def test_async_parity_list_namespaces(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        for ns in NAMESPACES:
            await s.aput(ns, "k", {"v": 1})
    for case in LIST_CASES:
        assert await store.alist_namespaces(**case) == await ref.alist_namespaces(**case), case


async def test_async_values_round_trip_unicode_and_nesting(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        await s.aput(("ünï", "命名空间"), "ключ-🔑", UNICODE_VALUE)
    mine = await store.aget(("ünï", "命名空间"), "ключ-🔑")
    theirs = await ref.aget(("ünï", "命名空间"), "ключ-🔑")
    assert mine is not None and theirs is not None
    assert norm(mine) == norm(theirs)
    assert mine.value == UNICODE_VALUE
    assert as_set(await store.asearch(("ünï",))) == as_set(await ref.asearch(("ünï",)))
    assert await store.alist_namespaces() == [("ünï", "命名空间")]


async def test_async_put_rejects_unserializable_value(astore_factory, mode) -> None:
    store = await astore_factory(mode)
    with pytest.raises(ValueError, match=r"value for ns/bad is not JSON serializable"):
        await store.abatch(
            [
                PutOp(("ns",), "good", {"v": 1}),
                PutOp(("ns",), "bad", {"when": datetime.now(timezone.utc)}),
            ]
        )
    assert await store.aget(("ns",), "good") is None
    assert await store.asearch(("ns",)) == []


async def test_async_batch_mixed_ops_keep_result_order(astore_factory, reference, mode) -> None:
    store, ref = await astore_factory(mode), reference()
    for s in (store, ref):
        await s.aput(("m", "1"), "a", {"v": 1})
        await s.aput(("m", "2"), "b", {"v": 2})
    mine, theirs = await both([store, ref], MIXED_OPS)
    assert mine == theirs
    assert mine[0] is None and mine[5] is None
    # Reads run before the batch's puts, in both stores
    assert mine[6] is None
    assert await both([store, ref], [GetOp(("m", "1"), "a"), GetOp(("m", "3"), "c")]) == [
        [None, (("m", "3"), "c", {"v": 3})],
        [None, (("m", "3"), "c", {"v": 3})],
    ]


async def test_concurrent_awaits_are_coalesced_into_one_batch(astore_factory, mode) -> None:
    store = await astore_factory(mode)
    for i in range(20):
        await store.aput(("co",), f"k{i}", {"i": i})
    calls: list[int] = []
    original = store.abatch

    async def counting(ops: Any) -> Any:
        ops = list(ops)
        calls.append(len(ops))
        return await original(ops)

    store.abatch = counting  # type: ignore[method-assign]
    items = await asyncio.gather(*(store.aget(("co",), f"k{i}") for i in range(20)))
    assert [item.value["i"] for item in items if item is not None] == list(range(20))
    assert 0 < len(calls) < 20
    assert sum(calls) == 20


@pytest.mark.timeout(60)
async def test_pooled_async_store_runs_batches_concurrently(astore_factory) -> None:
    store = await astore_factory("pool")
    await store.aput(("c",), "k", {"v": 1})
    # Each batch waits inside its own connection until the other is inside
    # too. If the store ran batches one at a time, the wait would time out.
    inside = 0
    both_inside = asyncio.Event()
    original = store._cursor

    @contextlib.asynccontextmanager
    async def meeting_cursor(**kwargs: Any) -> AsyncIterator[Any]:
        nonlocal inside
        async with original(**kwargs) as cur:
            inside += 1
            if inside == 2:
                both_inside.set()
            await asyncio.wait_for(both_inside.wait(), WAIT)
            yield cur

    store._cursor = meeting_cursor  # type: ignore[method-assign,assignment]
    # The queue behind aget runs one abatch at a time, as in AsyncPostgresStore,
    # so call abatch directly, as two graph nodes on one store would through
    # their own batches
    results = await asyncio.gather(
        store.abatch([GetOp(("c",), "k")]), store.abatch([GetOp(("c",), "k")])
    )
    assert [norm(result) for result in results] == [[(("c",), "k", {"v": 1})]] * 2


async def test_sync_methods_from_another_thread(astore_factory) -> None:
    store = await astore_factory("pool")
    await asyncio.to_thread(store.put, ("s",), "k", {"v": 1})
    item = await asyncio.to_thread(store.get, ("s",), "k")
    assert item is not None and item.value == {"v": 1}
    with pytest.raises(asyncio.InvalidStateError):
        store.get(("s",), "k")


async def test_adelete_namespace(astore_factory, mode) -> None:
    store = await astore_factory(mode)
    await store.aput(("user", "a"), "k1", {"v": 1})
    await store.aput(("user", "a"), "k2", {"v": 2})
    await store.aput(("user", "a", "b"), "k3", {"v": 3})
    await store.aput(("user", "ab"), "k4", {"v": 4})
    await store.aput(("user_a",), "k5", {"v": 5})
    assert await store.adelete_namespace(("user", "a")) == 3
    assert {(item.namespace, item.key) for item in await store.asearch(())} == {
        (("user", "ab"), "k4"),
        (("user_a",), "k5"),
    }
    assert await store.adelete_namespace(("user", "a")) == 0
    with pytest.raises(InvalidNamespaceError, match="drop the store's tables"):
        await store.adelete_namespace(())


async def test_from_conn_string_modes(libpq_url, table_name) -> None:
    for kwargs in (
        {},
        {"pipeline": True},
        {"pool_config": {"min_size": 1, "max_size": 2, "kwargs": {"application_name": "t5"}}},
    ):
        async with AsyncCockroachDBStore.from_conn_string(
            libpq_url, table_name=table_name, **kwargs
        ) as store:
            await store.setup()
            await store.aput(("f",), str(kwargs), {"v": 1})
            assert await store.aget(("f",), str(kwargs)) is not None
    # pool_config overrides pipeline, as in AsyncPostgresStore
    async with AsyncCockroachDBStore.from_conn_string(
        libpq_url, table_name=table_name, pipeline=True, pool_config={"min_size": 1}
    ) as store:
        assert isinstance(store.conn, AsyncConnectionPool)
        assert store.pipe is None
        await store.aput(("f",), "both", {"v": 1})
        assert await store.aget(("f",), "both") is not None
    with pytest.raises(ValueError, match="autocommit"):
        async with AsyncCockroachDBStore.from_conn_string(
            libpq_url, pool_config={"kwargs": {"autocommit": False}}
        ):
            pass
