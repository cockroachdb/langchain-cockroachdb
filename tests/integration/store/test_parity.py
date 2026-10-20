"""CockroachDBStore matches LangGraph's InMemoryStore.

Each parity test runs the same ops on the store, in every connection mode,
and on InMemoryStore, and compares values, keys, namespaces and, where both
define it, order. Timestamps are not compared.

Search order is one place they differ by design: InMemoryStore returns items
in insertion order, while the store, like PostgresStore, returns the most
recently updated first. Those results are compared as sets, and the store's
order is checked on its own.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import CockroachDBStore, RetryPolicy

WAIT = 10


def norm(result: Any) -> Any:
    """A result without timestamps, comparable across stores."""
    if isinstance(result, Item):
        return (result.namespace, result.key, result.value)
    if isinstance(result, list):
        return [norm(r) for r in result]
    return result


def as_set(result: list) -> list:
    """Search results in a fixed order, for comparing without order."""
    return sorted(norm(result), key=repr)


def both(stores: list[BaseStore], ops: list) -> list[list]:
    """Run ``ops`` on every store; return the normalized results of each."""
    return [norm(store.batch(ops)) for store in stores]


def test_parity_put_get_delete(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    for s in (store, ref):
        s.put(("docs", "a"), "k1", {"text": "one"})
        s.put(("docs", "a"), "k2", {"text": "two"})
        s.put(("docs", "b"), "k1", {"text": "other"})
        s.put(("docs", "a"), "k1", {"text": "one again", "n": 1})
        s.delete(("docs", "a"), "k2")
    ops = [
        GetOp(("docs", "a"), "k1"),
        GetOp(("docs", "a"), "k2"),
        GetOp(("docs", "b"), "k1"),
        GetOp(("docs",), "k1"),
        GetOp(("docs", "a"), "k1"),
    ]
    mine, theirs = both([store, ref], ops)
    assert mine == theirs
    assert mine[0] == (("docs", "a"), "k1", {"text": "one again", "n": 1})
    assert mine[1] is None
    item = store.get(("docs", "a"), "k1")
    assert item is not None
    assert item.created_at <= item.updated_at


def test_parity_search_prefix_and_descendants(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    for s in (store, ref):
        s.put(("user", "a"), "k1", {"v": 1})
        s.put(("user", "a", "b"), "k2", {"v": 2})
        s.put(("user", "ab"), "k3", {"v": 3})
        s.put(("other",), "k4", {"v": 4})
    for prefix, expected_keys in [
        (("user",), {"k1", "k2", "k3"}),
        (("user", "a"), {"k1", "k2"}),
        (("user", "a", "b"), {"k2"}),
        (("user", "ab"), {"k3"}),
        (("use",), set()),
        ((), {"k1", "k2", "k3", "k4"}),
    ]:
        mine, theirs = store.search(prefix), ref.search(prefix)
        assert as_set(mine) == as_set(theirs), prefix
        assert {item.key for item in mine} == expected_keys, prefix
        assert all(item.score is None for item in mine)


def test_parity_filters(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    values = {
        "apple": {"color": "red", "size": 3, "meta": {"lang": "en"}, "tags": ["a", "b"]},
        "banana": {"color": "yellow", "size": 10, "meta": {"lang": "fr"}, "tags": ["b"]},
        "cherry": {"color": "red", "size": 7.5, "meta": {"lang": "en", "x": 1}},
        "date": {"size": 5, "tags": ["a", "b", "c"]},
    }
    for s in (store, ref):
        for key, value in values.items():
            s.put(("fruit",), key, value)
    filters: list[tuple[dict, set[str]]] = [
        ({"color": "red"}, {"apple", "cherry"}),
        ({"color": {"$eq": "yellow"}}, {"banana"}),
        ({"meta": {"lang": "en"}}, {"apple", "cherry"}),
        ({"size": {"$gt": 5}}, {"banana", "cherry"}),
        ({"size": {"$gte": 5, "$lt": 10}}, {"cherry", "date"}),
        ({"size": {"$lte": 3}}, {"apple"}),
        ({"color": {"$ne": "red"}}, {"banana", "date"}),
        ({"tags": ["a", "b"]}, {"apple"}),
        ({"color": "red", "size": {"$gt": 4}}, {"cherry"}),
        ({"missing": "x"}, set()),
    ]
    for filter, expected in filters:
        mine, theirs = (
            store.search(("fruit",), filter=filter),
            ref.search(("fruit",), filter=filter),
        )
        assert as_set(mine) == as_set(theirs), filter
        assert {item.key for item in mine} == expected, filter

    # Text comparisons: InMemoryStore converts to float and cannot run these.
    # Numeric comparisons skip values that are not JSON numbers.
    store.put(("fruit",), "elder", {"size": "7", "color": "black"})
    text = store.search(("fruit",), filter={"color": {"$gt": "red"}})
    assert {item.key for item in text} == {"banana"}
    numeric = store.search(("fruit",), filter={"size": {"$gt": 6}})
    assert {item.key for item in numeric} == {"banana", "cherry"}


def test_parity_ordering_and_paging(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    for s in (store, ref):
        for i in range(4):
            s.put(("log",), f"k{i}", {"i": i})
        # One batch: the same updated_at, so namespace and key break the tie
        s.batch([PutOp(("log", "z"), "b", {"i": 9}), PutOp(("log", "y"), "a", {"i": 9})])
        s.put(("log",), "k1", {"i": 1, "again": True})

    full = store.search(("log",), limit=10)
    assert [(item.namespace, item.key) for item in full] == [
        (("log",), "k1"),
        (("log", "y"), "a"),
        (("log", "z"), "b"),
        (("log",), "k3"),
        (("log",), "k2"),
        (("log",), "k0"),
    ]
    assert as_set(full) == as_set(ref.search(("log",), limit=10))
    for limit, offset in [(2, 0), (2, 1), (3, 3), (1, 5), (10, 0)]:
        page = store.search(("log",), limit=limit, offset=offset)
        assert norm(page) == norm(full[offset : offset + limit]), (limit, offset)
    assert store.search(("log",), limit=0) == ref.search(("log",), limit=0) == []


def test_paging_edges(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    for s in (store, ref):
        s.put(("p",), "k", {"v": 1})
    with pytest.raises(ValueError, match="limit"):
        store.search(("p",), limit=-1)
    with pytest.raises(ValueError, match="offset"):
        store.search(("p",), offset=-1)
    with pytest.raises(ValueError, match="limit"):
        store.list_namespaces(limit=-1)
    with pytest.raises(ValueError, match="offset"):
        store.list_namespaces(offset=-2)
    assert store.search(("p",), offset=5) == ref.search(("p",), offset=5) == []
    assert store.list_namespaces(offset=5) == ref.list_namespaces(offset=5) == []


def test_parity_list_namespaces(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    namespaces = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("x", "b"),
        ("x", "f"),
    ]
    for s in (store, ref):
        for ns in namespaces:
            s.put(ns, "k", {"v": 1})
    cases: list[dict[str, Any]] = [
        {},
        {"prefix": ("a", "b")},
        {"prefix": ("a", "b"), "max_depth": 3},
        {"suffix": ("f",)},
        {"prefix": ("a",), "suffix": ("f",)},
        {"prefix": ("a", "*", "f")},
        {"suffix": ("*", "f")},
        {"prefix": ("*", "b")},
        {"max_depth": 1},
        {"max_depth": 2},
        {"max_depth": 2, "limit": 2, "offset": 1},
        {"limit": 3},
        {"limit": 2, "offset": 4},
        {"prefix": ("nope",)},
    ]
    for case in cases:
        assert store.list_namespaces(**case) == ref.list_namespaces(**case), case
    assert store.list_namespaces(prefix=("a", "b"), max_depth=3) == [
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("a", "b", "f"),
    ]


def test_values_round_trip_unicode_and_nesting(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    value = {
        "text": "héllo wörld, 你好, emoji 🎉, quote ' and \" and \\ backslash",
        "nested": {"list": [1, "two", {"three": [3.5, None, True, False]}], "empty": {}},
        "big": 12345678901234567890,
        "negative": -42,
        "float": 0.125,
        "none": None,
        "ключ 🔑": "unicode key",
    }
    for s in (store, ref):
        s.put(("ünï", "命名空间"), "ключ-🔑", value)
    mine = store.get(("ünï", "命名空间"), "ключ-🔑")
    theirs = ref.get(("ünï", "命名空间"), "ключ-🔑")
    assert mine is not None and theirs is not None
    assert norm(mine) == norm(theirs)
    assert mine.value == value
    assert as_set(store.search(("ünï",))) == as_set(ref.search(("ünï",)))
    assert store.list_namespaces() == ref.list_namespaces() == [("ünï", "命名空间")]


def test_put_rejects_unserializable_value_before_embedding(store_factory, mode) -> None:
    store = store_factory(mode)
    with pytest.raises(ValueError, match=r"value for ns/bad is not JSON serializable"):
        store.batch(
            [
                PutOp(("ns",), "good", {"v": 1}),
                PutOp(("ns",), "bad", {"when": datetime.now(timezone.utc)}),
            ]
        )
    with pytest.raises(ValueError, match="not JSON serializable"):
        store.put(("ns",), "nan", {"v": float("nan")})
    # Nothing of the failed batch was written
    assert store.get(("ns",), "good") is None
    assert store.search(("ns",)) == []


def test_batch_mixed_ops_keep_result_order(store_factory, reference, mode) -> None:
    store, ref = store_factory(mode), reference()
    for s in (store, ref):
        s.put(("m", "1"), "a", {"v": 1})
        s.put(("m", "2"), "b", {"v": 2})
    ops = [
        PutOp(("m", "3"), "c", {"v": 3}),
        GetOp(("m", "1"), "a"),
        SearchOp(("m", "2")),
        ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("m",)),)),
        GetOp(("m", "2"), "b"),
        PutOp(("m", "1"), "a", None),
        GetOp(("m", "3"), "c"),
        SearchOp(("m",), limit=0),
    ]
    mine, theirs = both([store, ref], ops)
    assert mine == theirs
    assert mine[0] is None and mine[5] is None
    # Reads run before the batch's puts, in both stores
    assert mine[6] is None
    assert both([store, ref], [GetOp(("m", "1"), "a"), GetOp(("m", "3"), "c")]) == [
        [None, (("m", "3"), "c", {"v": 3})],
        [None, (("m", "3"), "c", {"v": 3})],
    ]


@pytest.mark.timeout(60)
def test_pooled_store_runs_batches_concurrently(store_factory) -> None:
    store = store_factory("pool")
    store.put(("c",), "k", {"v": 1})
    # Each batch waits inside its own connection until the other is inside
    # too. If the store ran batches one at a time, the barrier would time out.
    barrier = threading.Barrier(2, timeout=WAIT)
    original = store._cursor

    @contextlib.contextmanager
    def meeting_cursor(**kwargs: Any) -> Iterator[Any]:
        with original(**kwargs) as cur:
            barrier.wait()
            yield cur

    store._cursor = meeting_cursor  # type: ignore[method-assign]
    errors: list[BaseException] = []
    results: list[Any] = []

    def run() -> None:
        try:
            results.append(store.get(("c",), "k"))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT * 2)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert [item.value for item in results] == [{"v": 1}, {"v": 1}]


def run_setup(url: str, table_name: str, barrier: Any) -> None:
    """Run setup() in a child process once every process is ready."""
    with CockroachDBStore.from_conn_string(
        url,
        table_name=table_name,
        retry_policy=RetryPolicy(initial_backoff=0.05, max_attempts=10),
    ) as store:
        barrier.wait(60)
        store.setup()


@pytest.mark.timeout(240)
def test_setup_from_two_processes_at_once(libpq_url, table_name, admin) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [
        context.Process(target=run_setup, args=(libpq_url, table_name, barrier)) for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(180)
    assert [process.exitcode for process in processes] == [0, 0]
    versions = [
        row["v"] for row in admin.execute(f"SELECT v FROM {table_name}_migrations ORDER BY v")
    ]
    assert versions == [0, 1]
    with CockroachDBStore.from_conn_string(libpq_url, table_name=table_name) as store:
        store.setup()
        store.put(("s",), "k", {"v": 1})
        item = store.get(("s",), "k")
        assert item is not None and item.value == {"v": 1}


def test_delete_namespace_counts_and_scopes(store_factory, mode) -> None:
    store = store_factory(mode)
    store.put(("user", "a"), "k1", {"v": 1})
    store.put(("user", "a"), "k2", {"v": 2})
    store.put(("user", "a", "b"), "k3", {"v": 3})
    store.put(("user", "ab"), "k4", {"v": 4})
    store.put(("user_a",), "k5", {"v": 5})
    assert store.delete_namespace(("user", "a")) == 3
    assert {(item.namespace, item.key) for item in store.search(())} == {
        (("user", "ab"), "k4"),
        (("user_a",), "k5"),
    }
    assert store.delete_namespace(("user", "a")) == 0
    with pytest.raises(ValueError, match="drop the store's tables"):
        store.delete_namespace(())


def test_ttl_expiry_refresh_and_sweep(store_factory, admin, table_name, mode) -> None:
    # A TTL configuration also turns on row-level TTL in setup()
    store = store_factory(mode, ttl={"default_ttl": 1.0, "refresh_on_read": True})
    store.put(("t",), "short", {"v": 1}, ttl=0.01)
    store.put(("t",), "long", {"v": 2})
    store.put(("t",), "none", {"v": 3}, ttl=None)
    store.put(("t",), "one", {"v": 4}, ttl=1)

    def expires(key: str) -> Any:
        row = admin.execute(
            f"SELECT expires_at, ttl_minutes FROM {table_name} WHERE key = %s", (key,)
        ).fetchone()
        return row

    # No ttl argument: the default. ttl=None: no expiry. ttl=1: one minute
    assert expires("long")["ttl_minutes"] == 1
    assert expires("none")["ttl_minutes"] is None
    assert expires("none")["expires_at"] is None
    assert expires("one")["ttl_minutes"] == 1
    before = expires("long")["expires_at"]
    time.sleep(1.2)
    assert store.get(("t",), "short") is None
    assert {item.key for item in store.search(("t",))} == {"long", "none", "one"}
    after = expires("long")["expires_at"]
    assert after > before
    assert store.get(("t",), "long", refresh_ttl=False) is not None
    assert expires("long")["expires_at"] == after
    assert store.sweep_ttl() == 1
    assert store.sweep_ttl() == 0
    assert expires("short") is None
    assert "ttl_expiration_expression" in create_statement(admin, table_name)
    store.disable_ttl()
    assert "ttl_expiration_expression" not in create_statement(admin, table_name)


def create_statement(admin: Any, table_name: str) -> str:
    row = admin.execute(f"SHOW CREATE TABLE {table_name}").fetchone()
    return row["create_statement"]


@pytest.mark.timeout(60)
def test_ttl_sweeper_thread_deletes_expired_rows(store_factory, admin, table_name) -> None:
    store = store_factory("pool", ttl={"default_ttl": 1.0})
    store.put(("t",), "short", {"v": 1}, ttl=0.005)
    store.put(("t",), "long", {"v": 2})
    # About 0.6 seconds between sweeps
    future = store.start_ttl_sweeper(0.01)  # type: ignore[arg-type]
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        keys = [row["key"] for row in admin.execute(f"SELECT key FROM {table_name}")]
        if keys == ["long"]:
            break
        time.sleep(0.2)
    assert keys == ["long"]
    assert store.stop_ttl_sweeper(timeout=WAIT)
    assert future.result(timeout=WAIT) is None
    assert store.stop_ttl_sweeper() is True


def test_store_without_ttl_config_has_no_sweeper(store_factory) -> None:
    store = store_factory("connection")
    future = store.start_ttl_sweeper()
    assert future.done() and future.result() is None


def test_from_conn_string_modes(libpq_url, table_name) -> None:
    for kwargs in (
        {},
        {"pipeline": True},
        {"pool_config": {"min_size": 1, "max_size": 2, "kwargs": {"application_name": "t4"}}},
    ):
        with CockroachDBStore.from_conn_string(libpq_url, table_name=table_name, **kwargs) as store:
            store.setup()
            store.put(("f",), str(kwargs), {"v": 1})
            assert store.get(("f",), str(kwargs)) is not None
    # pool_config overrides pipeline, as in PostgresStore
    with CockroachDBStore.from_conn_string(
        libpq_url, table_name=table_name, pipeline=True, pool_config={"min_size": 1}
    ) as store:
        assert isinstance(store.conn, ConnectionPool)
        assert store.pipe is None
        store.put(("f",), "both", {"v": 1})
        assert store.get(("f",), "both") is not None
    with (
        pytest.raises(ValueError, match="autocommit"),
        CockroachDBStore.from_conn_string(libpq_url, pool_config={"kwargs": {"autocommit": False}}),
    ):
        pass
