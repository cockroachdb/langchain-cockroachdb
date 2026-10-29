"""Unit tests for the store's shared planning code. No database."""

import asyncio
import contextlib
import json
import logging
import math
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    NOT_PROVIDED,
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb import _psycopg_conn
from langchain_cockroachdb.retry import RetryPolicy
from langchain_cockroachdb.store import base as store_base
from langchain_cockroachdb.store.aio import AsyncCockroachDBStore
from langchain_cockroachdb.store.base import (
    MAX_SEARCH_NAMESPACES,
    BaseCockroachDBStore,
    ProposalPlan,
    PutPlan,
    SchemaCheck,
    StoreIdentity,
    _group_ops,
    default_valid_at,
    merge_candidates,
    needs_exact_candidates,
    needs_exact_search,
)
from langchain_cockroachdb.store.config import resolve_history_config, resolve_index_config
from langchain_cockroachdb.store.reconcile import Candidate
from langchain_cockroachdb.store.sql import MIGRATION_RECONCILE, StoreSQL
from langchain_cockroachdb.store.store import CockroachDBStore

FAST = RetryPolicy(initial_backoff=0.001, max_backoff=0.002)


class CountingEmbeddings(Embeddings):
    """Embeddings that count their calls and return one vector per text.

    ``calls`` counts sync calls and ``async_calls`` awaited ones. The
    vector of a text is its length, then zeros, so tests can tell texts apart.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.async_calls = 0
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        return [[float(len(text)), 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0, 0.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.async_calls += 1
        self.texts.extend(texts)
        return [[float(len(text)), 0.0, 0.0] for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self.async_calls += 1
        return [1.0, 0.0, 0.0]


def make_base(*, fields: list[str] | None = None, ttl=None, history=None) -> BaseCockroachDBStore:
    """BaseCockroachDBStore state built directly, without a connection."""
    store = BaseCockroachDBStore()
    index = None
    if fields is not None:
        index = resolve_index_config({"dims": 3, "embed": CountingEmbeddings(), "fields": fields})
    store._configure(
        index=index,
        ttl=ttl,
        table_name="store",
        schema="public",
        history=resolve_history_config(history),
    )
    return store


def test_group_ops_validates_labels_and_keeps_order() -> None:
    ops = [
        GetOp(("a",), "k1"),
        PutOp(("a",), "k2", {"x": 1}),
        SearchOp(("a",)),
        GetOp(("b",), "k3"),
        ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("a",)),)),
    ]
    grouped, total = _group_ops(ops)
    assert total == 5
    assert grouped[GetOp] == [(0, ops[0]), (3, ops[3])]
    assert grouped[PutOp] == [(1, ops[1])]
    assert grouped[SearchOp] == [(2, ops[2])]
    assert grouped[ListNamespacesOp] == [(4, ops[4])]

    bad_ops = [
        GetOp(("a.b",), "k"),
        PutOp(("",), "k", {}),
        SearchOp(("a", "b.c")),
        ListNamespacesOp(match_conditions=(MatchCondition("suffix", ("x.y",)),)),
    ]
    for op in bad_ops:
        with pytest.raises(InvalidNamespaceError):
            _group_ops([GetOp(("ok",), "k"), op])


def test_plan_puts_dedups_last_wins_and_splits_deletes() -> None:
    store = make_base(ttl={"default_ttl": 5.0})
    plan = store._plan_puts(
        [
            PutOp(("a",), "k1", {"v": 1}),
            PutOp(("a",), "k2", {"v": 2}),
            PutOp(("a",), "k1", {"v": 3}, ttl=10.0),
            PutOp(("a", "b"), "k2", None),
            PutOp(("a",), "k2", None),
            PutOp(("c",), "k9", {"v": 9}),
            PutOp(("c",), "k9", None),
        ]
    )
    assert isinstance(plan, PutPlan)
    assert plan.items == [("a", "k1", '{"v": 3}', 10.0)]
    assert plan.deletes == {"a.b": ["k2"], "a": ["k2"], "c": ["k9"]}
    assert plan.embed_requests == []
    assert plan.reindexed == {}

    # The op's TTL is used as given: BaseStore.put already applied the
    # default, so None means no expiry even with a default configured
    plan = store._plan_puts([PutOp(("a",), "k1", {"v": 1})])
    assert plan.items == [("a", "k1", '{"v": 1}', None)]
    plan = make_base()._plan_puts([PutOp(("a",), "k1", {"v": 1})])
    assert plan.items == [("a", "k1", '{"v": 1}', None)]


def test_plan_puts_rejects_unserializable_value() -> None:
    store = make_base(fields=["text"])
    assert store.index is not None
    embeddings = store.index.embeddings
    assert isinstance(embeddings, CountingEmbeddings)
    with pytest.raises(ValueError, match=r"value for a\.b/k2 is not JSON serializable"):
        store._plan_puts(
            [
                PutOp(("a", "b"), "k1", {"text": "fine"}),
                PutOp(("a", "b"), "k2", {"text": "x", "when": datetime.now(timezone.utc)}),
            ]
        )
    with pytest.raises(ValueError, match="not JSON serializable"):
        store._plan_puts([PutOp(("a",), "k", {"n": float("nan")})])
    assert embeddings.calls == 0


def test_plan_puts_extracts_texts_and_names_vectors() -> None:
    store = make_base(fields=["messages[*].content"])
    value = {"messages": [{"content": "hello"}, {"content": "world"}]}
    plan = store._plan_puts(
        [
            PutOp(("a",), "k1", value),
            PutOp(("a",), "k2", {"messages": [{"content": "solo"}]}),
            PutOp(("a",), "k3", {"title": "t", "body": "b"}, index=["title"]),
            PutOp(("a",), "k4", value, index=False),
        ]
    )
    assert plan.embed_requests == [
        ("a", "k1", "messages[*].content.0", "hello"),
        ("a", "k1", "messages[*].content.1", "world"),
        ("a", "k2", "messages[*].content", "solo"),
        ("a", "k3", "title", "t"),
    ]
    # Every put item loses its old vectors first, index=False included
    assert plan.reindexed == {"a": ["k1", "k2", "k3", "k4"]}
    assert [row[1] for row in plan.items] == ["k1", "k2", "k3", "k4"]


def test_search_plan_warns_once_without_index(caplog: pytest.LogCaptureFixture) -> None:
    store = make_base()
    ops = [
        (0, SearchOp(("a",), query="hello", limit=5)),
        (3, SearchOp(("b",), query="again", limit=5, offset=1)),
    ]
    with caplog.at_level(logging.WARNING):
        statements, to_embed, lookups = store._search_plan(ops)
        store._search_plan(ops)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no index" in warnings[0].getMessage()
    assert to_embed == [] and lookups == []
    assert len(statements) == 2
    sql, params = statements[1]
    assert "ORDER BY updated_at DESC" in sql
    assert params[-2:] == [5, 1]


def test_paging_zero_limit_short_circuits() -> None:
    # A connection the store must never touch: using it would raise
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    assert store.batch([SearchOp(("a",), limit=0)]) == [[]]
    assert store.search(("a",), limit=0) == []
    with pytest.raises(ValueError, match="limit"):
        store.batch([SearchOp(("a",), limit=-1)])
    with pytest.raises(ValueError, match="offset"):
        store.batch([SearchOp(("a",), limit=0, offset=-1)])
    with pytest.raises(ValueError, match="limit"):
        store.batch([ListNamespacesOp(limit=-1)])


def test_index_is_accepted() -> None:
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"]},
    )
    assert store.index is not None
    assert store.index.dims == 3
    assert len(store.sql.vector_statements()) == 2


class RecordingCursor:
    """A cursor that records executes and fetches in a shared log."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.connection = self

    def cursor(self, **kwargs: object) -> "RecordingCursor":
        return RecordingCursor(self.log)

    def execute(self, query: str, params: list) -> None:
        self.log.append("execute")

    def fetchall(self) -> list:
        self.log.append("fetch")
        return []

    def close(self) -> None:
        pass


def test_read_group_queues_every_statement_before_fetching() -> None:
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    log: list[str] = []

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[RecordingCursor]:
        log.append("cursor")
        yield RecordingCursor(log)

    store._cursor = fake_cursor  # type: ignore[method-assign]
    results = store.batch([GetOp(("a",), "k"), GetOp(("b",), "k"), GetOp(("c",), "k")])
    assert results == [None, None, None]
    # One cursor context, three statements, every fetch after every execute
    assert log == ["cursor"] + ["execute"] * 3 + ["fetch"] * 3

    log.clear()
    store.batch([SearchOp(("a",)), SearchOp(("b",), refresh_ttl=False)])
    assert log == ["cursor"] + ["execute"] * 2 + ["fetch"] * 2


class FlakyCursor:
    """A write cursor whose first attempt fails with a retryable 40001.

    Every attempt records its statements in ``attempts``.
    """

    def __init__(self) -> None:
        self.attempts: list[list[tuple[str, list]]] = []

    def enter(self) -> list[tuple[str, list]]:
        self.attempts.append([])
        if len(self.attempts) == 1:
            raise psycopg.errors.SerializationFailure("restart transaction")
        return self.attempts[-1]


class StatementLog:
    def __init__(self, statements: list[tuple[str, list]]) -> None:
        self.statements = statements
        self.connection = self

    def execute(self, query: str, params: list) -> None:
        self.statements.append((query, params))


class AsyncStatementLog(StatementLog):
    async def execute(self, query: str, params: list) -> None:  # type: ignore[override]
        self.statements.append((query, params))


INDEX = {"dims": 3, "fields": ["text", "messages[*].content"]}
INDEXED_OPS = [
    PutOp(("a",), "k1", {"text": "one", "messages": [{"content": "hi"}, {"content": "there"}]}),
    PutOp(("a",), "k2", {"text": "two"}, ttl=5.0),
    PutOp(("a",), "k3", {"text": "skip me"}, index=False),
    PutOp(("a",), "k4", None),
]


def statement_kinds(statements: list[tuple[str, list]]) -> list[str]:
    """Each statement as its verb and table, e.g. "DELETE public.store_vectors"."""
    return [f"{sql.split()[0]} {sql.split()[2]}" for sql, _ in statements]


def vector_upserts(statements: list[tuple[str, list]], store: BaseCockroachDBStore) -> list:
    """The (prefix, key, field_name, vector literal, ttl) rows of the vector upserts."""
    rows = []
    for sql, params in statements:
        if sql.startswith(f"INSERT INTO {store.sql.vectors} "):
            rows.extend(tuple(params[i : i + 5]) for i in range(0, len(params), 5))
    return rows


def assert_write_batch(statements: list[tuple[str, list]], store: BaseCockroachDBStore) -> None:
    vectors, items = store.sql.vectors, store.sql.items
    # Old vectors go first, then items, new vectors, and the deletes
    assert statement_kinds(statements) == [
        f"DELETE {vectors}",
        f"INSERT {items}",
        f"INSERT {vectors}",
        f"DELETE {items}",
        f"DELETE {vectors}",
    ]
    assert statements[0][1] == ["a", ["k1", "k2", "k3"]]
    assert vector_upserts(statements, store) == [
        ("a", "k1", "text", "[3.0,0.0,0.0]", None),
        ("a", "k1", "messages[*].content.0", "[2.0,0.0,0.0]", None),
        ("a", "k1", "messages[*].content.1", "[5.0,0.0,0.0]", None),
        ("a", "k2", "text", "[3.0,0.0,0.0]", 5.0),
    ]
    assert statements[-1][1] == ["a", ["k4"]]


def test_put_plan_embeds_once_outside_retry() -> None:
    embeddings = CountingEmbeddings()
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={**INDEX, "embed": embeddings},  # type: ignore[typeddict-item]
        retry_policy=FAST,
    )
    # The fake cursor cannot answer the schema check: say it found a table
    # that setup() has not given migration 2 yet
    store._cache_columns([{"ready": False}])
    flaky = FlakyCursor()

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[StatementLog]:
        assert kwargs["pipeline"] is True and kwargs["transaction"] is True
        yield StatementLog(flaky.enter())

    store._cursor = fake_cursor  # type: ignore[method-assign]
    store.batch(INDEXED_OPS)
    assert embeddings.calls == 1
    assert embeddings.async_calls == 0
    assert embeddings.texts == ["one", "hi", "there", "two"]
    # The retry reran the same statements with the same vectors
    assert len(flaky.attempts) == 2
    assert_write_batch(flaky.attempts[1], store)


async def test_async_put_plan_embeds_once_outside_retry() -> None:
    embeddings = CountingEmbeddings()
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={**INDEX, "embed": embeddings},  # type: ignore[typeddict-item]
        retry_policy=FAST,
    )
    # The fake cursor cannot answer the schema check: say it found a table
    # that setup() has not given migration 2 yet
    store._cache_columns([{"ready": False}])
    flaky = FlakyCursor()

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncStatementLog]:
        assert kwargs["pipeline"] is True and kwargs["transaction"] is True
        yield AsyncStatementLog(flaky.enter())

    store._cursor = fake_cursor  # type: ignore[method-assign]
    await store.abatch(INDEXED_OPS)
    assert embeddings.async_calls == 1
    assert embeddings.calls == 0
    assert embeddings.texts == ["one", "hi", "there", "two"]
    assert len(flaky.attempts) == 2
    assert_write_batch(flaky.attempts[1], store)


BAD_BATCHES = [
    [PutOp(("a",), "good", {"text": "fine"}), PutOp(("a",), "bad", {"text": float("nan")})],
    [PutOp(("a",), "good", {"text": "fine"}), PutOp(("a.b",), "k", {"text": "x"})],
    [PutOp(("a",), "good", {"text": "fine"}), SearchOp(("a",), limit=-1)],
]


def test_validation_fails_before_embedding() -> None:
    embeddings = CountingEmbeddings()
    # A connection the store must never touch: using it would raise
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
    )
    for ops in BAD_BATCHES:
        with pytest.raises((ValueError, InvalidNamespaceError)):
            store.batch(ops)
    assert embeddings.calls == 0
    assert embeddings.async_calls == 0


async def test_async_validation_fails_before_embedding() -> None:
    embeddings = CountingEmbeddings()
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
    )
    for ops in BAD_BATCHES:
        with pytest.raises((ValueError, InvalidNamespaceError)):
            await store.abatch(ops)
    assert embeddings.calls == 0
    assert embeddings.async_calls == 0


def test_vector_rows_dedup_and_mirror_ttl() -> None:
    store = make_base(fields=["text"])
    plan = store._plan_puts(
        [
            PutOp(("a",), "k1", {"text": "one"}, ttl=7.0),
            PutOp(("a",), "k2", {"text": "two", "title": "t"}, index=["text", "title", "text"]),
        ]
    )
    # A field listed twice is embedded once
    assert plan.embed_requests == [
        ("a", "k1", "text", "one"),
        ("a", "k2", "text", "two"),
        ("a", "k2", "title", "t"),
    ]
    rows = store._vector_rows(plan, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert rows == [
        ("a", "k1", "text", [1.0, 0.0, 0.0], 7.0),
        ("a", "k2", "text", [2.0, 0.0, 0.0], None),
        ("a", "k2", "title", [3.0, 0.0, 0.0], None),
    ]


def test_vector_rows_check_the_embedding_model_output() -> None:
    store = make_base(fields=["text"])
    plan = store._plan_puts([PutOp(("a",), "k1", {"text": "one"})])
    with pytest.raises(ValueError, match="returned 2 embeddings for 1 texts"):
        store._vector_rows(plan, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="2 dimensions, but the index has 3"):
        store._vector_rows(plan, [[1.0, 0.0]])
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match=r"NaN or infinite values for a/k1, field text"):
            store._vector_rows(plan, [[1.0, bad, 0.0]])


class NaNEmbeddings(CountingEmbeddings):
    """A broken model whose vectors hold NaN."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float("nan"), 0.0, 0.0] for _ in texts]


def test_nan_embedding_fails_before_any_statement() -> None:
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": NaNEmbeddings(), "fields": ["text"]},
    )
    # The schema check runs before embedding, and the fake cursor cannot
    # answer it: say it found migration 2
    store._cache_columns([{"ready": True}])
    opened: list[object] = []

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[StatementLog]:
        opened.append(kwargs)
        yield StatementLog([])

    store._cursor = fake_cursor  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=r"NaN or infinite values for a/k, field text"):
        store.put(("a",), "k", {"text": "x"})
    assert opened == []


def test_put_without_texts_skips_the_model_and_still_removes_vectors() -> None:
    embeddings = CountingEmbeddings()
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
        retry_policy=FAST,
    )
    # The fake cursor cannot answer the schema check: say it found a table
    # that setup() has not given migration 2 yet
    store._cache_columns([{"ready": False}])
    batches: list[list[tuple[str, list]]] = []

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[StatementLog]:
        batches.append([])
        yield StatementLog(batches[-1])

    store._cursor = fake_cursor  # type: ignore[method-assign]
    store.batch([PutOp(("a",), "k", {"text": "x"}, index=False), PutOp(("a",), "j", None)])
    assert embeddings.calls == 0
    vectors, items = store.sql.vectors, store.sql.items
    assert statement_kinds(batches[0]) == [
        f"DELETE {vectors}",
        f"INSERT {items}",
        f"DELETE {items}",
        f"DELETE {vectors}",
    ]


async def test_async_put_without_texts_skips_the_model_and_still_removes_vectors() -> None:
    embeddings = CountingEmbeddings()
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
        retry_policy=FAST,
    )
    # The fake cursor cannot answer the schema check: say it found a table
    # that setup() has not given migration 2 yet
    store._cache_columns([{"ready": False}])
    batches: list[list[tuple[str, list]]] = []

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncStatementLog]:
        batches.append([])
        yield AsyncStatementLog(batches[-1])

    store._cursor = fake_cursor  # type: ignore[method-assign]
    await store.abatch([PutOp(("a",), "k", {"text": "x"}, index=False), PutOp(("a",), "j", None)])
    assert embeddings.calls == 0
    assert embeddings.async_calls == 0
    vectors, items = store.sql.vectors, store.sql.items
    assert statement_kinds(batches[0]) == [
        f"DELETE {vectors}",
        f"INSERT {items}",
        f"DELETE {items}",
        f"DELETE {vectors}",
    ]


def assert_same_write_each_attempt(
    attempts: list[list[tuple[str, list]]], store: BaseCockroachDBStore, ready: bool
) -> None:
    """Both attempts sent the same statements; the item upsert matches the schema."""
    assert len(attempts) == 2
    # The write id, like the vectors, was chosen before the first attempt
    assert attempts[0] == attempts[1]
    assert_write_batch(attempts[1], store)
    [(upsert, params)] = [
        (sql, params)
        for sql, params in attempts[1]
        if sql.startswith(f"INSERT INTO {store.sql.items} ")
    ]
    if ready:
        uuid.UUID(params[0])
        assert f"version = {store.sql.items}.version + 1" in upsert
        rows = [params[i : i + 6] for i in range(1, len(params), 6)]
        # A plain put writes no provenance and no valid time
        assert [row[4:] for row in rows] == [[None, None]] * 3
    else:
        assert "write_id" not in upsert and "version" not in upsert
        rows = [params[i : i + 4] for i in range(0, len(params), 4)]
    assert [row[1] for row in rows] == ["k1", "k2", "k3"]


@pytest.mark.parametrize("ready", [True, False], ids=["migrated", "0.5-table"])
def test_put_batch_sends_one_write_id_in_every_attempt(ready: bool) -> None:
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={**INDEX, "embed": CountingEmbeddings()},  # type: ignore[typeddict-item]
        retry_policy=FAST,
    )
    store._cache_columns([{"ready": ready}])
    attempts: list[list[tuple[str, list]]] = []

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[StatementLog]:
        attempts.append([])
        yield StatementLog(attempts[-1])
        # The first attempt fails as it ends, after all its statements were sent
        if len(attempts) == 1:
            raise psycopg.errors.SerializationFailure("restart transaction")

    store._cursor = fake_cursor  # type: ignore[method-assign]
    store.batch(INDEXED_OPS)
    assert_same_write_each_attempt(attempts, store, ready)


@pytest.mark.parametrize("ready", [True, False], ids=["migrated", "0.5-table"])
async def test_async_put_batch_sends_one_write_id_in_every_attempt(ready: bool) -> None:
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={**INDEX, "embed": CountingEmbeddings()},  # type: ignore[typeddict-item]
        retry_policy=FAST,
    )
    store._cache_columns([{"ready": ready}])
    attempts: list[list[tuple[str, list]]] = []

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncStatementLog]:
        attempts.append([])
        yield AsyncStatementLog(attempts[-1])
        if len(attempts) == 1:
            raise psycopg.errors.SerializationFailure("restart transaction")

    store._cursor = fake_cursor  # type: ignore[method-assign]
    await store.abatch(INDEXED_OPS)
    assert_same_write_each_attempt(attempts, store, ready)


def test_batch_plan_keeps_search_queries_to_embed() -> None:
    store = make_base(fields=["text"])
    plan = store._plan_batch(
        [SearchOp(("a",), query="hello", limit=3), SearchOp(("b",)), SearchOp(("c",), query="x")]
    )
    assert plan.search_to_embed == [(0, "hello"), (2, "x")]
    assert make_base()._plan_batch([SearchOp(("a",), query="q")]).search_to_embed == []


def test_only_a_store_with_a_ttl_configuration_refreshes_expiry() -> None:
    # BaseStore resolves refresh_ttl to true without a TTL configuration
    ops = [GetOp(("a",), "k", refresh_ttl=True), SearchOp(("a",), refresh_ttl=True)]
    for store, refreshes in ((make_base(), False), (make_base(ttl={"default_ttl": 1}), True)):
        plan = store._plan_batch(ops)
        statements = [*plan.get_statements, *plan.search_statements]
        assert ["UPDATE" in sql for sql, _ in statements] == [refreshes, refreshes]


class SearchConnection:
    """A fake connection for read groups. Every statement and its params is logged.

    Namespace lookups return ``namespaces``, indexed searches (the ones with
    a window) ``indexed_rows``, exact searches (GROUP BY) ``exact_rows``, and
    other statements no rows. The first attempt of the "store search" group
    fails with a retryable 40001 as its block ends, after its statements
    were sent.
    """

    def __init__(self, namespaces: list[str]) -> None:
        self.namespaces = namespaces
        self.indexed_rows: list[dict[str, object]] = []
        self.exact_rows: list[dict[str, object]] = []
        self.attempts: dict[str, int] = {}
        self.executed: list[tuple[str, str, list]] = []
        self.session: list[str] = []
        self.kwargs: list[dict[str, object]] = []
        self.label = ""

    def enter(self, kwargs: dict[str, object]) -> "SearchConnection":
        self.kwargs.append(kwargs)
        self.label = str(kwargs["label"])
        self.attempts[self.label] = self.attempts.get(self.label, 0) + 1
        return self

    def exit(self) -> None:
        if self.label == "store search" and self.attempts[self.label] == 1:
            raise psycopg.errors.SerializationFailure("restart transaction")

    @property
    def connection(self) -> "SearchConnection":
        return self

    def cursor(self, **kwargs: object) -> "SearchCursor":
        return SearchCursor(self)

    def execute(self, query: str, params: list | None = None, **kwargs: object) -> None:
        self.session.append(query)


class SearchCursor:
    def __init__(self, conn: SearchConnection) -> None:
        self.conn = conn
        self.rows: list[dict[str, object]] = []

    def execute(self, query: str, params: list) -> None:
        self.conn.executed.append((self.conn.label, query, params))
        if query.startswith("WITH RECURSIVE"):
            self.rows = [{"prefix": ns} for ns in self.conn.namespaces]
        elif "window_size" in query:
            self.rows = self.conn.indexed_rows
        elif "GROUP BY" in query:
            self.rows = self.conn.exact_rows

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows

    def close(self) -> None:
        pass


class AsyncSearchCursor(SearchCursor):
    async def execute(self, query: str, params: list) -> None:  # type: ignore[override]
        SearchCursor.execute(self, query, params)

    async def fetchall(self) -> list[dict[str, object]]:  # type: ignore[override]
        return self.rows

    async def close(self) -> None:  # type: ignore[override]
        pass


class AsyncSearchConnection(SearchConnection):
    def cursor(self, **kwargs: object) -> AsyncSearchCursor:  # type: ignore[override]
        return AsyncSearchCursor(self)

    async def execute(self, query: str, params: list | None = None, **kwargs: object) -> None:  # type: ignore[override]
        self.session.append(query)


QUERY_OPS = [
    SearchOp(("a",), query="first", limit=2, offset=3),
    SearchOp(("b",), query="first"),
    SearchOp((), query="second"),
    SearchOp(("c",)),
]


def assert_query_statements(conn: SearchConnection) -> None:
    lookups = [params for label, _, params in conn.executed if label == "store search namespaces"]
    # One lookup per semantic search under a namespace, sent once: lookups are not retried here
    assert lookups == [
        ["a", "a.", "a/"] * 2 + [MAX_SEARCH_NAMESPACES],
        ["b", "b.", "b/"] * 2 + [MAX_SEARCH_NAMESPACES],
    ]
    searches = [(sql, params) for label, sql, params in conn.executed if label == "store search"]
    # Two attempts of four statements each; the retry reuses the same vectors
    assert len(searches) == 8
    assert searches[:4] == searches[4:]
    first, second, root, plain = searches[4:]
    assert "prefix = ANY(CAST(%s AS STRING[]))" in first[0]
    assert "count(*) OVER () AS window_size" in first[0]
    assert first[1][:3] == ["[1.0,0.0,0.0]", ["a", "a.x"], "[1.0,0.0,0.0]"]
    # over_fetch = (limit + offset) * estimated_vectors * 2 + 1
    assert first[1][3] == (2 + 3) * 1 * 2 + 1
    assert second[1][1] == ["a", "a.x"]
    # A search of the whole store has no lookup and ranks exactly at once
    assert "ANY(" not in root[0] and "GROUP BY i.prefix, i.key" in root[0]
    assert root[1][0] == "[1.0,0.0,0.0]"
    assert "embedding" not in plain[0]
    # No rows came back, so both indexed searches ran again, exactly
    exact = [(sql, params) for label, sql, params in conn.executed if label == "store search exact"]
    assert len(exact) == 2
    for sql, params in exact:
        assert "v.prefix = ANY(CAST(%s AS STRING[]))" in sql
        assert "LIMIT %s OFFSET %s" in sql and sql.count("LIMIT") == 1
        assert params[:2] == ["[1.0,0.0,0.0]", ["a", "a.x"]]
    assert [params[-2:] for _, params in exact] == [[2, 3], [10, 0]]
    for _, params in [*searches, *exact]:
        assert all(type(param) is not object for param in params)


def test_query_is_embedded_once_per_distinct_text_outside_retry() -> None:
    embeddings = CountingEmbeddings()
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
        retry_policy=FAST,
    )
    conn = SearchConnection(["a", "a.x"])

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[SearchConnection]:
        yield conn.enter(kwargs)
        conn.exit()

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    results = store.batch(QUERY_OPS)
    assert results == [[], [], [], []]
    # Two distinct texts, each embedded once with embed_query, before any attempt
    assert embeddings.calls == 2
    assert embeddings.async_calls == 0
    assert conn.attempts == {
        "store search namespaces": 1,
        "store search": 2,
        "store search exact": 1,
    }
    assert_query_statements(conn)
    # No beam size configured: no session statement and no transaction
    assert conn.session == []
    assert all(kwargs["transaction"] is False for kwargs in conn.kwargs)


async def test_async_query_is_embedded_once_per_distinct_text_outside_retry() -> None:
    embeddings = CountingEmbeddings()
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
        retry_policy=FAST,
    )
    conn = AsyncSearchConnection(["a", "a.x"])

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncSearchConnection]:
        yield conn.enter(kwargs)  # type: ignore[misc]
        conn.exit()

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    results = await store.abatch(QUERY_OPS)
    assert results == [[], [], [], []]
    assert embeddings.async_calls == 2
    assert embeddings.calls == 0
    assert conn.attempts == {
        "store search namespaces": 1,
        "store search": 2,
        "store search exact": 1,
    }
    assert_query_statements(conn)
    assert conn.session == []


class FailingEmbeddings(CountingEmbeddings):
    """aembed_query fails for "bad" and waits forever for any other text."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = 0

    async def aembed_query(self, text: str) -> list[float]:
        if text == "bad":
            await asyncio.sleep(0)
            raise RuntimeError("embedding service down")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return [1.0, 0.0, 0.0]


async def test_async_failed_query_embedding_cancels_the_others() -> None:
    embeddings = FailingEmbeddings()
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
    )
    with pytest.raises(RuntimeError, match="embedding service down"):
        await store.abatch(
            [SearchOp(("a",), query="slow"), SearchOp(("a",), query="bad"), SearchOp((), query="x")]
        )
    # Let the cancellations run
    await asyncio.sleep(0)
    assert embeddings.cancelled == 2


def test_beam_size_runs_set_local_in_the_search_transaction() -> None:
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"], "beam_size": 24},
        retry_policy=FAST,
    )
    conn = SearchConnection(["a"])

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[SearchConnection]:
        yield conn.enter(kwargs)
        conn.exit()

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    store.batch([SearchOp(("a",), query="q"), GetOp(("a",), "k")])
    # Once per attempt of the search group, and only there
    assert conn.session == ["SET LOCAL vector_search_beam_size = 24"] * 2
    transactions = {str(kwargs["label"]): kwargs["transaction"] for kwargs in conn.kwargs}
    # The exact search after the empty indexed one has no beam size to set
    assert transactions == {
        "store search namespaces": False,
        "store get": False,
        "store search": True,
        "store search exact": False,
    }
    # A batch without a query needs no beam size
    conn.session.clear()
    conn.attempts.clear()
    store.batch([SearchOp(("a",))])
    assert conn.session == []


def fake_search_store(
    namespaces: list[str], **index: object
) -> tuple[CockroachDBStore, SearchConnection]:
    """A store with an index whose reads go to a SearchConnection, with no injected failure."""
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"], **index},  # type: ignore[typeddict-item]
        retry_policy=FAST,
    )
    conn = SearchConnection(namespaces)
    conn.attempts["store search"] = 1  # no injected failure

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[SearchConnection]:
        yield conn.enter(kwargs)
        conn.exit()

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    return store, conn


def statements_with_namespaces(count: int) -> list[tuple[str, str, list]]:
    """The (label, SQL, params) sent when the lookup finds ``count`` namespaces."""
    store, conn = fake_search_store([f"a.n{i}" for i in range(count)])
    store.batch([SearchOp(("a",), query="q", filter={"x": 1})])
    return conn.executed


def test_too_many_namespaces_fall_back_to_one_statement() -> None:
    assert MAX_SEARCH_NAMESPACES == 1000
    executed = statements_with_namespaces(MAX_SEARCH_NAMESPACES)
    [capped] = [(sql, params) for label, sql, params in executed if label == "store search"]
    assert "prefix = ANY(CAST(%s AS STRING[]))" in capped[0]
    assert len(capped[1][1]) == MAX_SEARCH_NAMESPACES

    executed = statements_with_namespaces(MAX_SEARCH_NAMESPACES + 1)
    [fallback] = [(sql, params) for label, sql, params in executed if label == "store search"]
    # The exact form over the namespace range, no array and no window
    assert "ANY(" not in fallback[0] and "window_size" not in fallback[0]
    assert "(v.prefix = %s OR (v.prefix >= %s AND v.prefix < %s))" in fallback[0]
    vector = "[1.0,0.0,0.0]"
    assert fallback[1] == [vector, "a", "a.", "a/", "x", "1", 10, 0]
    # It is exact already, so it never runs again, though it found nothing
    assert [label for label, _, _ in executed] == ["store search namespaces", "store search"]


def search_row(
    key: str, score: float, window_size: int | None = None, held: int | None = None
) -> dict[str, object]:
    """A search row; an indexed one has a window size and namespace_vectors (default: equal)."""
    row: dict[str, object] = {
        "prefix": "a",
        "key": key,
        "value": {"text": key},
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "score": score,
    }
    if window_size is not None:
        row["window_size"] = window_size
        row["namespace_vectors"] = window_size if held is None else held
    return row


def test_needs_exact_search() -> None:
    over_fetch = 21
    # Short, and the window equals the count of the namespaces' vectors: the
    # index returned every vector, so the page is exact
    assert not needs_exact_search([search_row("k1", 0.9, 5, 5)], limit=10, over_fetch=over_fetch)
    # Short, and the namespaces hold more than the window: the index cut it
    # short, as a small beam_size or small partitions can
    assert needs_exact_search([search_row("k1", 0.9, 5, 6)], limit=10, over_fetch=over_fetch)
    assert needs_exact_search([search_row("k1", 0.9, 5, 21)], limit=10, over_fetch=over_fetch)
    # Nothing came back: the sizes are unknown
    assert needs_exact_search([], limit=10, over_fetch=over_fetch)
    # A full page is complete, whatever the window
    page = [search_row(f"k{i}", 0.9, 5, 21) for i in range(10)]
    assert not needs_exact_search(page, limit=10, over_fetch=over_fetch)
    assert not needs_exact_search(page[:3], limit=3, over_fetch=over_fetch)
    # Short from a full window: the count stops at over_fetch, so equal sizes
    # cannot show that the namespaces hold no more vectors
    assert needs_exact_search([search_row("k1", 0.9, 21, 21)], limit=10, over_fetch=over_fetch)


EXACT_ROWS = [search_row("best", 0.99), search_row("next", 0.5)]


@pytest.mark.parametrize(
    ("indexed_rows", "namespaces", "runs_again"),
    [
        ([search_row("k1", 0.9, 21)], ["a"], True),
        ([search_row("k1", 0.9, 20)], ["a"], False),
        ([search_row("k1", 0.9, 20, 21)], ["a"], True),
        ([search_row(f"k{i}", 0.9, 21) for i in range(10)], ["a"], False),
        ([], ["a"], True),
        ([], [], False),
    ],
    ids=[
        "short-full-window",
        "short-window-equals-count",
        "short-window-below-count",
        "full-page",
        "empty",
        "no-namespace",
    ],
)
def test_short_indexed_search_runs_again_exactly(
    indexed_rows: list[dict[str, object]], namespaces: list[str], runs_again: bool
) -> None:
    store, conn = fake_search_store(namespaces)
    conn.indexed_rows = indexed_rows
    conn.exact_rows = EXACT_ROWS
    # The default limit of 10, so over_fetch is 10 * 1 * 2 + 1 = 21
    [result] = store.batch([SearchOp(("a",), query="q", filter={"x": 1})])
    labels = [label for label, _, _ in conn.executed]
    assert labels[:2] == ["store search namespaces", "store search"]
    if runs_again:
        assert labels[2:] == ["store search exact"]
        _, sql, params = conn.executed[2]
        assert "GROUP BY i.prefix, i.key" in sql
        assert params == ["[1.0,0.0,0.0]", namespaces, "x", "1", 10, 0]
        assert [(item.key, item.score) for item in result] == [("best", 0.99), ("next", 0.5)]
    else:
        assert labels[2:] == []
        assert [item.key for item in result] == [row["key"] for row in indexed_rows]


def test_exact_search_keeps_each_result_in_its_slot() -> None:
    store, conn = fake_search_store(["a"])
    conn.exact_rows = EXACT_ROWS
    results = store.batch(
        [
            GetOp(("a",), "k"),
            SearchOp(("a",), query="q"),
            SearchOp(("a",)),
            SearchOp(("b",), query="q", limit=2),
        ]
    )
    assert results[0] is None and results[2] == []
    assert [item.key for item in results[1]] == ["best", "next"]
    assert [item.key for item in results[3]] == ["best", "next"]
    exact = [params for label, _, params in conn.executed if label == "store search exact"]
    assert [params[-2:] for params in exact] == [[10, 0], [2, 0]]


async def test_async_short_indexed_search_runs_again_exactly() -> None:
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"]},
        retry_policy=FAST,
    )
    conn = AsyncSearchConnection(["a"])
    conn.attempts["store search"] = 1
    conn.indexed_rows = [search_row("k1", 0.9, 21)]
    conn.exact_rows = EXACT_ROWS

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncSearchConnection]:
        yield conn.enter(kwargs)  # type: ignore[misc]
        conn.exit()

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    [result] = await store.abatch([SearchOp(("a",), query="q")])
    assert [label for label, _, _ in conn.executed] == [
        "store search namespaces",
        "store search",
        "store search exact",
    ]
    assert [item.key for item in result] == ["best", "next"]


@pytest.mark.parametrize("prefix", [("a",), ()])
def test_flat_store_ranks_exactly_without_a_lookup(prefix: tuple[str, ...]) -> None:
    store, conn = fake_search_store(["a"], flat=True)
    conn.exact_rows = EXACT_ROWS
    [result] = store.batch([SearchOp(prefix, query="q", filter={"x": 1}, limit=3)])
    # One round trip: no lookup, since there is no vector index for it to serve
    [(label, sql, params)] = conn.executed
    assert label == "store search"
    assert "GROUP BY i.prefix, i.key" in sql and "ANY(" not in sql
    namespace = ["a", "a.", "a/"] if prefix else []
    assert params == ["[1.0,0.0,0.0]", *namespace, "x", "1", 3, 0]
    assert [item.key for item in result] == ["best", "next"]


def test_query_vector_is_checked_before_any_statement() -> None:
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 4, "embed": CountingEmbeddings(), "fields": ["text"]},
    )
    with pytest.raises(ValueError, match="3 dimensions, but the index has 4"):
        store.batch([SearchOp(("a",), query="q")])


VECTORS_TABLE = """CREATE TABLE public.store_vectors (
\tprefix STRING NOT NULL,
\tkey STRING NOT NULL,
\tfield_name STRING NOT NULL,
\tembedding VECTOR({dims}) NOT NULL,
\tCONSTRAINT store_vectors_pkey PRIMARY KEY (prefix ASC, key ASC, field_name ASC),
{index}
)"""
COSINE_INDEX = "\tVECTOR INDEX store_vectors_embedding_idx (prefix, embedding vector_cosine_ops)"


def index_base(**extra: object) -> BaseCockroachDBStore:
    store = BaseCockroachDBStore()
    index = resolve_index_config(
        {"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"], **extra}  # type: ignore[typeddict-item]
    )
    store._configure(index=index, ttl=None, table_name="store", schema="public")
    return store


def test_index_check_passes_on_a_matching_index(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        index_base()._check_vector_index(VECTORS_TABLE.format(dims=3, index=COSINE_INDEX))
        # No index at all (setup creates it) is not an error either
        index_base()._check_vector_index(VECTORS_TABLE.format(dims=3, index=""))
    assert caplog.records == []


def test_index_check_rejects_dims_and_distance_changes() -> None:
    with pytest.raises(ValueError, match=r"VECTOR\(8\).*dims=3"):
        index_base()._check_vector_index(VECTORS_TABLE.format(dims=8, index=COSINE_INDEX))
    with pytest.raises(ValueError) as raised:
        index_base(distance_type="l2")._check_vector_index(
            VECTORS_TABLE.format(dims=3, index=COSINE_INDEX)
        )
    assert str(raised.value) == (
        "the vector index store_vectors_embedding_idx uses vector_cosine_ops; drop it with "
        "drop_vector_index() and run setup() again to use vector_l2_ops"
    )


def test_index_check_warns_on_partition_sizes_and_flat(caplog: pytest.LogCaptureFixture) -> None:
    sized = COSINE_INDEX + " WITH (min_partition_size=16, max_partition_size=64)"
    statement = VECTORS_TABLE.format(dims=3, index=sized)
    with caplog.at_level(logging.WARNING):
        index_base(
            cspann={"min_partition_size": 32, "max_partition_size": 128}
        )._check_vector_index(statement)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "min_partition_size = 32 (index has 16)" in message
    assert "max_partition_size = 128 (index has 64)" in message

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # No partition sizes configured: whatever the index has is accepted
        index_base()._check_vector_index(statement)
        index_base(cspann={"min_partition_size": 16, "max_partition_size": 64})._check_vector_index(
            statement
        )
    assert caplog.records == []

    with caplog.at_level(logging.WARNING):
        index_base(flat=True)._check_vector_index(statement)
    assert len(caplog.records) == 1
    assert "flat" in caplog.records[0].getMessage()
    assert "drop_vector_index()" in caplog.records[0].getMessage()


# CockroachDB v25.4 refuses a TTL change while another schema change on the
# table runs. RetryPolicy() alone does not retry that error, so these tests
# pin that setup() and disable_ttl run their TTL statements under
# schema_change_policy.
BUSY_MESSAGE = (
    "cannot modify TTL settings while another schema change on the table is being processed"
)


def busy_error() -> psycopg.Error:
    """The error v25.4 raises for a TTL change during another schema change."""
    error = psycopg.errors.lookup("0A000")(BUSY_MESSAGE)
    error.sqlstate = "0A000"
    return error


class BusyOnceCursor:
    """A cursor that fails the first TTL statement with the busy error.

    It records every statement, and reports ``latest`` as the migration
    version, so setup() goes straight to its TTL step.
    """

    def __init__(self, latest: int) -> None:
        self.latest = latest
        self.statements: list[str] = []
        self.failed = False

    def run(self, statement: str) -> None:
        self.statements.append(statement)
        if statement.startswith("ALTER TABLE") and not self.failed:
            self.failed = True
            raise busy_error()

    def execute(self, statement: str, *args: object) -> "BusyOnceCursor":
        self.run(statement)
        return self

    def fetchone(self) -> dict:
        return {"v": self.latest}


class AsyncBusyOnceCursor(BusyOnceCursor):
    async def execute(self, statement: str, *args: object) -> "AsyncBusyOnceCursor":  # type: ignore[override]
        self.run(statement)
        return self

    async def fetchone(self) -> dict:  # type: ignore[override]
        return {"v": self.latest}


def ttl_statements(cursor: BusyOnceCursor) -> list[str]:
    return [statement for statement in cursor.statements if statement.startswith("ALTER TABLE")]


@pytest.mark.parametrize("operation", ["setup", "disable_ttl"])
def test_ttl_changes_retry_a_busy_schema_change(operation: str) -> None:
    assert not FAST.is_retryable(busy_error())
    for policy, retried in ((FAST, True), (RetryPolicy(max_attempts=1), False)):
        store = CockroachDBStore(object(), ttl={"default_ttl": 5}, retry_policy=policy)  # type: ignore[arg-type]
        cursor = BusyOnceCursor(latest=len(store.sql.migrations()) - 1)

        @contextlib.contextmanager
        def fake_cursor(
            cursor: BusyOnceCursor = cursor, **kwargs: object
        ) -> Iterator[BusyOnceCursor]:
            yield cursor

        store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
        if retried:
            getattr(store, operation)()
            # The busy statement ran again, under the store's own backoff
            statements = ttl_statements(cursor)
            assert len(statements) == 2 and statements[0] == statements[1]
        else:
            # max_attempts=1 turns the retry off
            with pytest.raises(psycopg.errors.FeatureNotSupported):
                getattr(store, operation)()
            assert len(ttl_statements(cursor)) == 1


@pytest.mark.parametrize("operation", ["setup", "adisable_ttl"])
async def test_async_ttl_changes_retry_a_busy_schema_change(operation: str) -> None:
    for policy, retried in ((FAST, True), (RetryPolicy(max_attempts=1), False)):
        store = AsyncCockroachDBStore(
            AsyncConnectionPool("", open=False), ttl={"default_ttl": 5}, retry_policy=policy
        )
        assert not hasattr(store, "disable_ttl")
        cursor = AsyncBusyOnceCursor(latest=len(store.sql.migrations()) - 1)

        @contextlib.asynccontextmanager
        async def fake_cursor(
            cursor: AsyncBusyOnceCursor = cursor, **kwargs: object
        ) -> AsyncIterator[AsyncBusyOnceCursor]:
            yield cursor

        store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
        if retried:
            await getattr(store, operation)()
            statements = ttl_statements(cursor)
            assert len(statements) == 2 and statements[0] == statements[1]
        else:
            with pytest.raises(psycopg.errors.FeatureNotSupported):
                await getattr(store, operation)()
            assert len(ttl_statements(cursor)) == 1


def test_cache_columns_reads_whether_the_items_table_has_them() -> None:
    store = make_base()
    assert store._columns_ready is None
    assert store._cache_columns([]) is False
    assert store._columns_ready is False
    assert store._cache_columns([{"ready": False}]) is False
    assert store._cache_columns([{"ready": True}]) is True
    assert store._columns_ready is True


class SchemaReads:
    """A stand-in for a store's _read_all that answers the schema check.

    It returns ``columns`` as whether migration 2's columns exist and, when
    asked, ``history`` as whether the history table exists. It records every call.
    """

    def __init__(self, columns: bool, history: bool = False) -> None:
        self.columns = columns
        self.history = history
        self.calls: list[tuple[list[tuple[str, list]], str]] = []

    def answer(self, statements: list[tuple[str, list]], label: str) -> list[list[dict]]:
        self.calls.append((statements, label))
        found = [[{"ready": self.columns}]]
        if len(statements) > 1:
            found.append([{"ready": self.history}])
        return found


class FakeClock:
    """Stands in for the time module in store.base: monotonic() returns ``now``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def test_cached_columns_keep_true_for_good_and_false_for_a_while(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(store_base, "time", clock)
    store = make_base()
    assert store._cached_columns(refresh=False) is None
    store._cache_columns([{"ready": False}])
    assert store._cached_columns(refresh=False) is False
    # A method that needs migration 2 does not trust False at all
    assert store._cached_columns(refresh=True) is None
    # A plain write trusts it for _SCHEMA_RECHECK_SECONDS, then reads again
    assert store._SCHEMA_RECHECK_SECONDS == 60.0
    clock.now += 59.0
    assert store._cached_columns(refresh=False) is False
    clock.now += 1.0
    assert store._cached_columns(refresh=False) is None
    # The window starts again with each read that finds no migration 2
    store._cache_columns([{"ready": False}])
    clock.now += 59.0
    assert store._cached_columns(refresh=False) is False
    # Migrations are never undone, so True holds for good
    store._cache_columns([{"ready": True}])
    clock.now += 365 * 86400.0
    assert store._cached_columns(refresh=False) is True
    assert store._cached_columns(refresh=True) is True


def test_schema_check_caches_its_answer_and_setup_clears_it(monkeypatch) -> None:
    store = CockroachDBStore(object(), retry_policy=FAST)  # type: ignore[arg-type]
    reads = SchemaReads(columns=False)

    def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    assert store._reconcile_columns() is False
    assert store._reconcile_columns() is False
    assert reads.calls == [([store.sql.reconcile_migration_recorded()], "store schema check")]
    with pytest.raises(ValueError, match=r"public\.store .* run setup\(\) to add them"):
        store._require_columns(store._reconcile_columns())
    # A method that needs migration 2 reads again rather than trust the cached False
    assert store._reconcile_columns(refresh=True) is False
    assert len(reads.calls) == 2
    # So does a plain write, once the False is older than _SCHEMA_RECHECK_SECONDS
    monkeypatch.setattr(store, "_SCHEMA_RECHECK_SECONDS", 0.0)
    assert store._reconcile_columns() is False
    assert len(reads.calls) == 3

    # setup() finds migration 2 recorded, as when another process ran it,
    # and clears the cached answer, so the next check reads again
    cursor = BusyOnceCursor(latest=MIGRATION_RECONCILE)

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[BusyOnceCursor]:
        yield cursor

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    store.setup()
    assert store._columns_ready is None
    reads.columns = True
    assert store._reconcile_columns() is True
    # True is kept for good, even with refresh and a window of 0
    assert store._reconcile_columns() is True
    assert store._reconcile_columns(refresh=True) is True
    assert len(reads.calls) == 4
    store._require_columns(store._reconcile_columns(refresh=True))


async def test_async_schema_check_caches_its_answer_and_setup_clears_it(monkeypatch) -> None:
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False), retry_policy=FAST)
    reads = SchemaReads(columns=False)

    async def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    assert await store._areconcile_columns() is False
    assert await store._areconcile_columns() is False
    assert reads.calls == [([store.sql.reconcile_migration_recorded()], "store schema check")]
    with pytest.raises(ValueError, match=r"run setup\(\)"):
        store._require_columns(await store._areconcile_columns())
    assert await store._areconcile_columns(refresh=True) is False
    assert len(reads.calls) == 2
    monkeypatch.setattr(store, "_SCHEMA_RECHECK_SECONDS", 0.0)
    assert await store._areconcile_columns() is False
    assert len(reads.calls) == 3

    cursor = AsyncBusyOnceCursor(latest=MIGRATION_RECONCILE)

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncBusyOnceCursor]:
        yield cursor

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    await store.setup()
    assert store._columns_ready is None
    reads.columns = True
    assert await store._areconcile_columns() is True
    assert await store._areconcile_columns() is True
    assert await store._areconcile_columns(refresh=True) is True
    assert len(reads.calls) == 4
    store._require_columns(await store._areconcile_columns(refresh=True))


def test_cache_columns_keeps_true_once_seen() -> None:
    store = make_base()
    assert store._cache_columns([{"ready": True}]) is True
    # A read that started before another process ran setup() can finish
    # after one that saw migration 2. It must not take the answer back
    assert store._cache_columns([{"ready": False}]) is True
    assert store._cache_columns([]) is True
    assert store._columns_ready is True


def test_require_columns_acts_on_the_check_it_is_given() -> None:
    store = make_base()
    # setup() in another thread can clear the cache between the check and this call
    assert store._columns_ready is None
    store._require_columns(True)
    store._cache_columns([{"ready": True}])
    with pytest.raises(ValueError, match=r"public\.store .* run setup\(\) to add them"):
        store._require_columns(False)


def test_reconcile_uses_the_schema_check_it_ran(monkeypatch) -> None:
    store = CockroachDBStore(object(), retry_policy=FAST)  # type: ignore[arg-type]

    def check(*, refresh: bool = False) -> bool:
        # As if setup() in another thread cleared the cache right after the read
        store._columns_ready = None
        return True

    monkeypatch.setattr(store, "_reconcile_columns", check)
    with pytest.raises(NotImplementedError):
        store.reconcile(("a",), "k", {"text": "x"}, policy=lambda proposal, candidates: None)


class DeniedSchemaReads(SchemaReads):
    """SchemaReads for a role without a grant on the migrations table.

    The migrations read raises ``error``, by default the InsufficientPrivilege
    that CockroachDB raises. The column read and the history check answer as
    in SchemaReads. Every call is recorded, the failed ones too.
    """

    def __init__(
        self, columns: bool, history: bool = False, error: Exception | None = None
    ) -> None:
        super().__init__(columns, history)
        self.error = error or psycopg.errors.InsufficientPrivilege(
            "user app does not have SELECT privilege on relation store_migrations"
        )

    def answer(self, statements: list[tuple[str, list]], label: str) -> list[list[dict]]:
        if "_migrations" in statements[0][0]:
            self.calls.append((statements, label))
            raise self.error
        return super().answer(statements, label)


def store_reading(
    make: str, reads: SchemaReads, **settings: object
) -> CockroachDBStore | AsyncCockroachDBStore:
    """A sync or async store whose reads go to ``reads``."""
    if make == "sync":

        def read_all(
            statements: list[tuple[str, list]], label: str, session: object = None
        ) -> list[list[dict]]:
            return reads.answer(statements, label)

        store = CockroachDBStore(object(), retry_policy=FAST, **settings)  # type: ignore[arg-type]
        store._read_all = read_all  # type: ignore[method-assign,assignment]
        return store

    async def aread_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    astore = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        retry_policy=FAST,
        **settings,  # type: ignore[arg-type]
    )
    astore._read_all = aread_all  # type: ignore[method-assign,assignment]
    return astore


async def check_columns(
    store: CockroachDBStore | AsyncCockroachDBStore, *, refresh: bool = False
) -> bool:
    """The store's check for migration 2's columns, sync or async."""
    if isinstance(store, AsyncCockroachDBStore):
        return await store._areconcile_columns(refresh=refresh)
    return store._reconcile_columns(refresh=refresh)


def sent(reads: SchemaReads) -> list[list[tuple[str, list]]]:
    """The statements of every schema check round trip, in order."""
    assert all(label == "store schema check" for _, label in reads.calls)
    return [statements for statements, _ in reads.calls]


@pytest.mark.parametrize("history", [False, True], ids=["plain", "history"])
@pytest.mark.parametrize("make", ["sync", "async"])
async def test_a_denied_migrations_read_falls_back_to_the_columns_for_good(
    make: str, history: bool, monkeypatch
) -> None:
    reads = DeniedSchemaReads(columns=False, history=True)
    store = store_reading(make, reads, history=history)
    migrations = store.sql.reconcile_migration_recorded()
    columns = store.sql.reconcile_columns_exist()
    tail = [store.sql.history_exists()] if history else []

    assert await check_columns(store) is False
    # The denial ended its round trip, so the column read and the history
    # check go out together in a second one
    assert sent(reads) == [[migrations, *tail], [columns, *tail]]
    # From then on the check reads the columns only, cached as on the
    # migrations path: False holds for _SCHEMA_RECHECK_SECONDS, and refresh
    # reads again
    assert await check_columns(store) is False
    assert len(reads.calls) == 2
    assert await check_columns(store, refresh=True) is False
    assert sent(reads)[2:] == [[columns, *tail]]
    monkeypatch.setattr(store, "_SCHEMA_RECHECK_SECONDS", 0.0)
    reads.columns = True
    assert await check_columns(store) is True
    assert sent(reads)[3:] == [[columns, *tail]]
    # True holds for good
    assert await check_columns(store, refresh=True) is True
    assert len(reads.calls) == 4


@pytest.mark.parametrize("make", ["sync", "async"])
async def test_other_errors_of_the_migrations_read_propagate(make: str) -> None:
    missing = psycopg.errors.UndefinedTable('relation "public.store_migrations" does not exist')
    reads = DeniedSchemaReads(columns=True, error=missing)
    store = store_reading(make, reads)
    for _ in range(2):
        with pytest.raises(psycopg.errors.UndefinedTable):
            await check_columns(store)
    # No column read: each check tried the migrations table, and nothing was cached
    assert sent(reads) == [[store.sql.reconcile_migration_recorded()]] * 2
    assert store._columns_ready is None


@pytest.mark.parametrize("make", ["sync", "async"])
async def test_a_denied_column_read_raises_without_trying_the_migrations_again(
    make: str,
) -> None:
    reads = DeniedSchemaReads(columns=True)
    store = store_reading(make, reads)

    def deny(statements: list[tuple[str, list]], label: str) -> list[list[dict]]:
        reads.calls.append((statements, label))
        raise psycopg.errors.InsufficientPrivilege("permission denied")

    reads.answer = deny  # type: ignore[method-assign]
    for _ in range(2):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await check_columns(store)
    migrations = store.sql.reconcile_migration_recorded()
    columns = store.sql.reconcile_columns_exist()
    assert sent(reads) == [[migrations], [columns], [columns]]


@pytest.mark.parametrize("make", ["sync", "async"])
async def test_a_migrations_read_denied_in_a_callers_transaction_raises_once(make: str) -> None:
    denied = psycopg.errors.InsufficientPrivilege(
        "user app does not have SELECT privilege on relation store_migrations"
    )
    _psycopg_conn.mark_in_caller_transaction(denied)
    reads = DeniedSchemaReads(columns=True, error=denied)
    store = store_reading(make, reads)
    # The denial aborted the caller's transaction, so no read can run in it:
    # the error goes to the caller
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        await check_columns(store)
    assert sent(reads) == [[store.sql.reconcile_migration_recorded()]]
    # When the caller tries again, the check goes straight to the columns
    assert await check_columns(store) is True
    assert sent(reads)[1:] == [[store.sql.reconcile_columns_exist()]]


@pytest.mark.parametrize("make", ["sync", "async"])
async def test_a_denied_write_is_not_taken_for_a_denied_migrations_read(make: str) -> None:
    reads = SchemaReads(columns=True)
    store = store_reading(make, reads)
    denied = psycopg.errors.InsufficientPrivilege(
        "user app does not have INSERT privilege on relation store"
    )
    if isinstance(store, CockroachDBStore):

        @contextlib.contextmanager
        def fake_cursor(**kwargs: object) -> Iterator[StatementLog]:
            raise denied
            yield

        store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            store.put(("a",), "k", {"x": 1})
    else:

        @contextlib.asynccontextmanager
        async def afake_cursor(**kwargs: object) -> AsyncIterator[AsyncStatementLog]:
            raise denied
            yield

        store._cursor = afake_cursor  # type: ignore[method-assign,assignment]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await store.aput(("a",), "k", {"x": 1})
    # The schema check read the migrations table once and was not denied
    assert sent(reads) == [[store.sql.reconcile_migration_recorded()]]
    assert store._migrations_denied is False


def test_history_check_keeps_true_for_good_and_false_for_a_while(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(store_base, "time", clock)
    store = make_base(history=True)
    assert store._cached_history(refresh=False) is None
    assert store._cache_history([{"ready": False}]) is False
    assert store._cached_history(refresh=False) is False
    # An operation about to refuse for a missing table does not trust False
    assert store._cached_history(refresh=True) is None
    clock.now += store._SCHEMA_RECHECK_SECONDS
    assert store._cached_history(refresh=False) is None
    assert store._cache_history([{"ready": True}]) is True
    # setup() never drops the table, so True holds for good, even against a late read
    assert store._cache_history([{"ready": False}]) is True
    clock.now += 365 * 86400.0
    assert store._cached_history(refresh=True) is True
    # A store without history never looks for the table
    assert make_base()._cached_history(refresh=True) is False


def test_history_check_reads_once_with_the_schema_before_refusing() -> None:
    store = CockroachDBStore(object(), history=True, retry_policy=FAST)  # type: ignore[arg-type]
    reads = SchemaReads(columns=True, history=False)

    def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    missing_table = r"The history table public\.store_history does not exist yet; run setup\(\)"
    with pytest.raises(ValueError, match=missing_table):
        store._history_schema()
    # One round trip reads both, once
    statements = [store.sql.reconcile_migration_recorded(), store.sql.history_exists()]
    assert reads.calls == [(statements, "store schema check")]
    # A plain read of the columns trusts the answers for a while
    assert store._reconcile_columns() is True
    assert len(reads.calls) == 1
    # An operation that needs the table does not trust its absence: it reads once
    with pytest.raises(ValueError, match=missing_table):
        store._history_schema()
    assert len(reads.calls) == 2

    # Another process runs setup(): the next operation that needs history sees it
    reads.history = True
    assert store._history_schema() == SchemaCheck(columns=True, history=True)
    assert len(reads.calls) == 3
    # and trusts it for good
    store._history_schema()
    assert len(reads.calls) == 3

    # Without migration 2 there is no history either, and the message says
    # what is missing
    store = CockroachDBStore(object(), history=True, retry_policy=FAST)  # type: ignore[arg-type]
    reads = SchemaReads(columns=False, history=True)
    store._read_all = read_all  # type: ignore[method-assign,assignment]
    with pytest.raises(
        ValueError,
        match=r"The table public\.store does not have the columns that history needs yet; "
        r"run setup\(\) to add them and to create the history table public\.store_history\.",
    ):
        store._history_schema()
    assert len(reads.calls) == 1


async def test_async_history_check_reads_once_with_the_schema_before_refusing() -> None:
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False), history=True, retry_policy=FAST
    )
    reads = SchemaReads(columns=True, history=False)

    async def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    with pytest.raises(
        ValueError, match=r"public\.store_history does not exist yet; run setup\(\)"
    ):
        await store._ahistory_schema()
    statements = [store.sql.reconcile_migration_recorded(), store.sql.history_exists()]
    assert reads.calls == [(statements, "store schema check")]
    reads.history = True
    assert await store._ahistory_schema() == SchemaCheck(columns=True, history=True)
    await store._ahistory_schema()
    assert len(reads.calls) == 2


def test_a_store_without_history_refuses_history_without_any_io() -> None:
    # A connection the store must never touch: using it would raise
    store = CockroachDBStore(object(), retry_policy=FAST)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="history=True"):
        store.history(("a",), "k")
    with pytest.raises(ValueError, match="history=True"):
        store.purge(("a",), "k")


def test_history_and_purge_validate_before_any_io() -> None:
    store = CockroachDBStore(object(), history=True, retry_policy=FAST)  # type: ignore[arg-type]
    for call in (
        lambda: store.history((), "k"),
        lambda: store.history(("a.b",), "k"),
        lambda: store.purge((), "k"),
        lambda: store.purge(("a", ""), "k"),
    ):
        with pytest.raises(InvalidNamespaceError):
            call()
    with pytest.raises(ValueError, match="limit must be non-negative"):
        store.history(("a",), "k", limit=-1)


@pytest.mark.parametrize("make", ["sync", "async"])
async def test_setup_creates_the_history_table_and_clears_its_check(make: str) -> None:
    if make == "sync":
        store: CockroachDBStore | AsyncCockroachDBStore = CockroachDBStore(
            object(),  # type: ignore[arg-type]
            history={"ttl_days": 30},
            retry_policy=FAST,
        )
        cursor: BusyOnceCursor = BusyOnceCursor(latest=MIGRATION_RECONCILE)

        @contextlib.contextmanager
        def fake_cursor(**kwargs: object) -> Iterator[BusyOnceCursor]:
            yield cursor

        store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    else:
        store = AsyncCockroachDBStore(
            AsyncConnectionPool("", open=False), history={"ttl_days": 30}, retry_policy=FAST
        )
        cursor = AsyncBusyOnceCursor(latest=MIGRATION_RECONCILE)

        @contextlib.asynccontextmanager
        async def afake_cursor(**kwargs: object) -> AsyncIterator[BusyOnceCursor]:
            yield cursor

        store._cursor = afake_cursor  # type: ignore[method-assign,assignment]
    store._cache_history([{"ready": False}])

    result = store.setup()
    if make == "async":
        await result  # type: ignore[misc]
    create, ttl = store.sql.history_statements(30.0)
    # The TTL change met another schema change once and ran again, as the
    # items' TTL does
    assert [s for s in cursor.statements if store.sql.history in s] == [create, ttl, ttl]
    assert store._history_ready is None


def test_put_batch_with_history_archives_what_it_ends_first() -> None:
    store = make_base(fields=["text"], history={"ttl_days": 7})
    plan = store._plan_puts(
        [
            PutOp(("a",), "k1", {"text": "one"}),
            PutOp(("b",), "k2", {"text": "two"}),
            PutOp(("a",), "k3", None),
        ]
    )
    plan.write_id = "w-1"
    rows = store._vector_rows(plan, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    statements = store._write_statements(plan, rows)
    history, items, vectors = store.sql.history, store.sql.items, store.sql.vectors
    assert statement_kinds(statements) == [
        f"INSERT {history}",
        f"INSERT {history}",
        f"INSERT {history}",
        f"DELETE {vectors}",
        f"DELETE {vectors}",
        f"INSERT {items}",
        f"INSERT {vectors}",
        f"DELETE {items}",
        f"DELETE {vectors}",
    ]
    # (invalid_at, reason, superseded_by, write_id, ttl_days, prefix, keys, write_id)
    archives = [params for _, params in statements[:3]]
    assert [(p[1], p[5], p[6]) for p in archives] == [
        ("overwritten", "a", ["k1"]),
        ("overwritten", "b", ["k2"]),
        ("deleted", "a", ["k3"]),
    ]
    assert all(p[0] is None and p[2] is None and p[4] == 7.0 for p in archives)
    assert all(p[3] == p[7] == "w-1" for p in archives)
    # The upsert numbers a new row after its versions in history
    assert f"FROM {history} AS h" in statements[5][0]

    # Without a write id the table has no migration 2 yet: nothing to archive
    plan.write_id = None
    assert history not in " ".join(sql for sql, _ in store._write_statements(plan, rows))
    # And without history the batch is the one it was
    plain = make_base(fields=["text"])
    plan.write_id = "w-1"
    assert history not in " ".join(sql for sql, _ in plain._write_statements(plan, rows))


# What delete_namespace(("user", "a")) sent in 0.5.1 on a store without an index
ZERO_FIVE_DELETE_NAMESPACE = (
    "DELETE FROM public.store\nWHERE (prefix = %s OR (prefix >= %s AND prefix < %s))\n"
    "RETURNING key",
    ["user.a", "user.a.", "user.a/"],
)


class HistoryTableReads:
    """A stand-in for a store's _read_all that says whether the history table exists."""

    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.calls: list[tuple[list[tuple[str, list]], str]] = []

    def answer(self, statements: list[tuple[str, list]], label: str) -> list[list[dict]]:
        self.calls.append((statements, label))
        return [[{"ready": self.exists}]]


class DeleteLog:
    """A cursor for delete_namespace that records its statement and deletes nothing."""

    def __init__(self, sent: list[tuple[str, list]]) -> None:
        self.sent = sent

    def execute(self, query: str, params: list) -> None:
        self.sent.append((query, params))

    def fetchall(self) -> list:
        return []


class AsyncDeleteLog(DeleteLog):
    async def execute(self, query: str, params: list) -> None:  # type: ignore[override]
        self.sent.append((query, params))

    async def fetchall(self) -> list:  # type: ignore[override]
        return []


def deleting_store_without_history(
    exists: bool,
) -> tuple[CockroachDBStore, HistoryTableReads, list[tuple[str, list]]]:
    """A store without history whose reads and delete statements are faked."""
    store = CockroachDBStore(object(), retry_policy=FAST)  # type: ignore[arg-type]
    reads = HistoryTableReads(exists)
    sent: list[tuple[str, list]] = []

    def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[DeleteLog]:
        yield DeleteLog(sent)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    return store, reads, sent


def test_delete_namespace_without_history_looks_for_a_history_table_each_time() -> None:
    store, reads, sent = deleting_store_without_history(exists=False)
    assert store.delete_namespace(("user", "a")) == 0
    assert store.delete_namespace(("user", "a")) == 0
    # One read per call while there is no history table, and the 0.5.x statement
    assert reads.calls == [([store.sql.history_exists()], "store history check")] * 2
    assert sent == [ZERO_FIVE_DELETE_NAMESPACE] * 2
    # The read leaves the schema check alone: its clock did not move, and
    # plain writes never look for the table
    assert store._schema_read_at == 0.0
    assert store._schema_statements(read_migrations=True) == [
        store.sql.reconcile_migration_recorded()
    ]


def test_delete_namespace_without_history_stops_looking_once_it_found_the_table() -> None:
    store, reads, sent = deleting_store_without_history(exists=True)
    store.delete_namespace(("user", "a"))
    reads.exists = False
    store.delete_namespace(("user", "a"))
    # The table was seen, so the second call trusts it and deletes history too
    assert len(reads.calls) == 1
    assert sent == [store.sql.delete_namespace(("user", "a"), history=True)] * 2
    # Plain writes still never look for it
    assert store._cached_history(refresh=True) is False


async def test_async_delete_namespace_without_history_looks_for_a_history_table() -> None:
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False), retry_policy=FAST)
    reads = HistoryTableReads(exists=False)
    sent: list[tuple[str, list]] = []

    async def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[AsyncDeleteLog]:
        yield AsyncDeleteLog(sent)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    assert await store.adelete_namespace(("user", "a")) == 0
    assert sent == [ZERO_FIVE_DELETE_NAMESPACE]
    # Once it finds the table it deletes history too, and stops looking
    reads.exists = True
    await store.adelete_namespace(("user", "a"))
    await store.adelete_namespace(("user", "a"))
    assert reads.calls == [([store.sql.history_exists()], "store history check")] * 2
    assert sent[1:] == [store.sql.delete_namespace(("user", "a"), history=True)] * 2
    assert store._schema_read_at == 0.0


def test_a_put_refused_for_a_missing_history_table_costs_no_embedding_call() -> None:
    embeddings = CountingEmbeddings()
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
        history=True,
        retry_policy=FAST,
    )
    reads = SchemaReads(columns=True, history=False)

    def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    with pytest.raises(
        ValueError, match=r"public\.store_history does not exist yet; run setup\(\)"
    ):
        store.put(("a",), "k", {"text": "x"})
    assert embeddings.calls == 0
    assert len(reads.calls) == 1


async def test_async_put_refused_for_a_missing_history_table_costs_no_embedding_call() -> None:
    embeddings = CountingEmbeddings()
    store = AsyncCockroachDBStore(
        AsyncConnectionPool("", open=False),
        index={"dims": 3, "embed": embeddings, "fields": ["text"]},
        history=True,
        retry_policy=FAST,
    )
    reads = SchemaReads(columns=True, history=False)

    async def read_all(
        statements: list[tuple[str, list]], label: str, session: object = None
    ) -> list[list[dict]]:
        return reads.answer(statements, label)

    store._read_all = read_all  # type: ignore[method-assign,assignment]
    with pytest.raises(
        ValueError, match=r"public\.store_history does not exist yet; run setup\(\)"
    ):
        await store.aput(("a",), "k", {"text": "x"})
    assert embeddings.async_calls == 0
    assert embeddings.calls == 0
    assert len(reads.calls) == 1


def test_cancelling_the_sweeper_future_stops_it_without_a_traceback(monkeypatch) -> None:
    store = CockroachDBStore(object(), ttl={"default_ttl": 1})  # type: ignore[arg-type]
    swept: list[int] = []
    monkeypatch.setattr(store, "sweep_ttl", lambda: swept.append(1) or 0)
    # What the sweeper thread would print as "Exception in thread ttl-sweeper"
    uncaught: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", uncaught.append)
    future = store.start_ttl_sweeper(sweep_interval_minutes=1)
    thread = store._ttl_sweeper_thread
    assert thread is not None and thread.is_alive()
    assert future.cancel()
    thread.join(5)
    assert not thread.is_alive()
    assert uncaught == []
    assert future.cancelled() and swept == []
    assert store.stop_ttl_sweeper(timeout=1)


def test_listing_bounds_prefix_conditions_to_their_literal_labels() -> None:
    store = make_base()
    op = ListNamespacesOp(
        match_conditions=(
            MatchCondition("prefix", ("user", "*", "memories")),
            MatchCondition("suffix", ("notes",)),
        ),
        max_depth=3,
        limit=5,
        offset=1,
    )
    sql, params = store._listing_statement(op)
    # The range for ("user",), then both regexes; a suffix adds no range
    assert sql.count("(prefix = %s OR (prefix >= %s AND prefix < %s))") == 1
    assert params == [
        "user",
        "user.",
        "user/",
        r"^user\.[^.]+\.memories(\.|\z)",
        r"(^|\.)notes\z",
        5,
        1,
    ]
    # A path that starts with a wildcard has no literal labels to bound
    _, params = store._listing_statement(
        ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("*", "b")),))
    )
    assert params == [r"^[^.]+\.b(\.|\z)", 100, 0]


# Candidates, find_conflicts and record

NS = ("user", "alice")
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
TEXT = {"text": "x"}


def proposal_store(distance: str | None = "cosine", *, ttl=None) -> BaseCockroachDBStore:
    """Shared store state with a 3-dimension index of the given distance, or no index."""
    store = BaseCockroachDBStore()
    index = None
    if distance is not None:
        index = resolve_index_config(
            {
                "dims": 3,
                "embed": CountingEmbeddings(),
                "fields": ["text"],
                "distance_type": distance,  # type: ignore[typeddict-item]
            }
        )
    store._configure(index=index, ttl=ttl, table_name="store", schema="public")
    return store


def plan_proposal(store: BaseCockroachDBStore, value: object = TEXT, **kwargs) -> ProposalPlan:
    """_plan_proposal with the find_conflicts defaults, under NS and key "k" unless given."""
    arguments: dict = {
        "slot": None,
        "threshold": None,
        "limit": 10,
        "filter": None,
        "index": None,
        "ttl": NOT_PROVIDED,
        "provenance": None,
        "valid_at": None,
        "method": "find_conflicts",
    }
    arguments.update(kwargs)
    namespace = arguments.pop("namespace", NS)
    key = arguments.pop("key", "k")
    return store._plan_proposal(namespace, key, value, **arguments)  # type: ignore[arg-type]


def test_plan_proposal_threshold_rules() -> None:
    cosine = proposal_store("cosine")
    # A cosine score is 1 minus the distance: the default is 0.8, and any
    # threshold lies between -1 and 1
    assert plan_proposal(cosine).threshold == 0.8
    for threshold in (-1, -0.5, 0, 0.95, 1):
        assert plan_proposal(cosine, threshold=threshold).threshold == float(threshold)
    for bad in (1.5, -1.01, math.nan, math.inf, True, "0.9"):
        with pytest.raises(ValueError, match="threshold"):
            plan_proposal(cosine, threshold=bad)

    # l2 and inner product scores are negated distances: no default fits
    for distance in ("l2", "inner_product"):
        store = proposal_store(distance)
        with pytest.raises(ValueError, match="threshold"):
            plan_proposal(store)
        assert plan_proposal(store, threshold=-0.6).threshold == -0.6
        assert plan_proposal(store, threshold=3).threshold == 3.0
        with pytest.raises(ValueError, match="threshold"):
            plan_proposal(store, threshold=math.nan)

    # Without an index, or with index=False, no similar candidates are
    # searched, so the threshold is ignored
    for store, extra in ((proposal_store(None), {}), (proposal_store("l2"), {"index": False})):
        for threshold in (None, 1.5, "0.9"):
            assert plan_proposal(store, threshold=threshold, **extra).threshold is None


def test_plan_proposal_slot_rules() -> None:
    store = proposal_store("cosine")
    for slot_value, as_json in (
        ("x", '"x"'),
        ("it's", '"it\'s"'),
        (42, "42"),
        (1.5, "1.5"),
        (True, "true"),
        (False, "false"),
    ):
        plan = plan_proposal(store, {"seat": slot_value, "text": "x"}, slot="seat")
        assert (plan.slot, plan.slot_value, plan.slot_value_json) == ("seat", slot_value, as_json)
    plan = plan_proposal(store)
    assert (plan.slot, plan.slot_value, plan.slot_value_json) == (None, None, None)

    # The slot must name a top-level field of the value, holding a string,
    # number or boolean
    with pytest.raises(ValueError, match="seat"):
        plan_proposal(store, {"text": "x"}, slot="seat")
    with pytest.raises(ValueError, match="seat"):
        plan_proposal(store, {"text": "x", "nested": {"seat": "x"}}, slot="seat")
    for bad in ({"a": 1}, [1, 2], None):
        with pytest.raises(ValueError, match="seat"):
            plan_proposal(store, {"seat": bad, "text": "x"}, slot="seat")
    with pytest.raises(ValueError, match="slot"):
        plan_proposal(store, {"1": "x"}, slot=1)
    # A value that is not JSON fails first, so a NaN or infinite slot value does too
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="JSON"):
            plan_proposal(store, {"seat": bad, "text": "x"}, slot="seat")


def test_plan_proposal_valid_at_must_be_aware() -> None:
    store = proposal_store("cosine")
    aware = datetime(2026, 3, 1, 12, tzinfo=timezone(timedelta(hours=-5)))
    assert plan_proposal(store, valid_at=aware).valid_at == aware
    # Without one, the database's clock gives the valid time once the
    # candidates are read (default_valid_at)
    assert plan_proposal(store).valid_at is None
    for bad in (
        datetime(2026, 3, 1, 12),
        "2026-03-01T12:00:00+00:00",
        date(2026, 3, 1),
        1767225600,
    ):
        with pytest.raises(ValueError, match="valid_at"):
            plan_proposal(store, valid_at=bad)


def test_plan_proposal_plans_one_put_with_a_write_id() -> None:
    store = proposal_store("cosine", ttl={"default_ttl": 30.0})
    plan = plan_proposal(
        store,
        {"text": "x", "pair": (1, 2)},
        key=None,
        provenance={"source": "chat", "turns": (3,)},
    )
    uuid.UUID(plan.key)  # a uuid4 when no key is given
    assert (plan.namespace, plan.prefix) == (NS, "user.alice")
    # The value and the provenance as their JSON reads back
    assert plan.value == {"text": "x", "pair": [1, 2]}
    assert json.loads(plan.value_json) == plan.value
    assert plan.provenance == {"source": "chat", "turns": [3]}
    assert plan.provenance_json is not None
    assert json.loads(plan.provenance_json) == plan.provenance
    # One put with the store's default TTL, the write id and the texts to embed
    assert plan.ttl == 30.0
    assert plan.put.items == [("user.alice", plan.key, plan.value_json, 30.0)]
    assert plan.put.write_id is not None
    uuid.UUID(plan.put.write_id)
    assert plan.put.write_id != plan_proposal(store).put.write_id
    assert plan.embed_requests == [("user.alice", plan.key, "text", "x")]
    assert plan.put.reindexed == {"user.alice": [plan.key]}
    assert (plan.limit, plan.filter, plan.index) == (10, None, None)

    # Keys become strings, as put makes them
    assert plan_proposal(store, key=7).key == "7"
    assert plan_proposal(store, ttl=None).ttl is None
    assert plan_proposal(store, ttl=5).put.items[0][3] == 5
    assert plan_proposal(store, provenance=None).provenance_json is None
    assert plan_proposal(store, limit=0).limit == 0
    assert plan_proposal(store, filter={"source": "chat"}).filter == {"source": "chat"}
    assert plan_proposal(store, index=False).embed_requests == []
    other = plan_proposal(store, {"text": "x", "other": "y"}, index=["other"])
    assert other.embed_requests == [("user.alice", "k", "other", "y")]
    assert plan_proposal(proposal_store(None)).embed_requests == []

    for kwargs, match in (
        ({"ttl": 0}, "ttl"),
        ({"ttl": -1}, "ttl"),
        ({"ttl": math.inf}, "ttl"),
        ({"ttl": "5"}, "ttl"),
        ({"ttl": True}, "ttl"),
        ({"provenance": ["chat"]}, "provenance"),
        ({"provenance": {"x": object()}}, "provenance"),
        ({"provenance": {"x": math.nan}}, "provenance"),
        ({"limit": -1}, "limit must be non-negative"),
        ({"limit": 1.5}, "limit"),
        ({"limit": True}, "limit"),
        ({"filter": {"x": {"$bad": 1}}}, "Unknown filter operator"),
        ({"filter": ["x"]}, "filter"),
    ):
        with pytest.raises(ValueError, match=match):
            plan_proposal(store, **kwargs)
    for bad_value in (None, ["x"], "x"):
        with pytest.raises(ValueError, match="value"):
            plan_proposal(store, bad_value)
    with pytest.raises(ValueError, match="JSON"):
        plan_proposal(store, {"x": object()})
    with pytest.raises(InvalidNamespaceError, match="find_conflicts"):
        plan_proposal(store, namespace=())
    with pytest.raises(InvalidNamespaceError):
        plan_proposal(store, namespace=("a.b",))


def candidate_row(
    key: str,
    *,
    score: float | None = None,
    version: int = 1,
    updated_at: datetime = T0,
    valid_at: datetime | None = None,
) -> dict[str, object]:
    """A row of a candidate statement; similar rows carry a score."""
    row: dict[str, object] = {
        "key": key,
        "value": {"text": key},
        "created_at": T0,
        "updated_at": updated_at,
        "version": version,
        "valid_at": updated_at if valid_at is None else valid_at,
        "provenance": {"source": key},
        "confirmed_at": None,
        "confirmations": version - 1,
        "write_id": None,
    }
    if score is not None:
        row["score"] = score
    return row


def test_merge_candidates_orders_and_caps() -> None:
    key_slot = [candidate_row("s2"), candidate_row("k", version=3), candidate_row("s1")]
    # Scores of the key and slot candidates, one list per proposal vector
    scores = [
        [{"key": "k", "score": 0.2}, {"key": "s1", "score": 0.5}],
        [{"key": "s1", "score": 0.7}],
    ]
    similar = [
        [
            candidate_row("a", score=0.6),
            candidate_row("c", score=0.85),
            candidate_row("d", score=0.8),
        ],
        [
            candidate_row("a", score=0.95),
            candidate_row("b", score=0.85),
            candidate_row("s1", score=0.75),
        ],
    ]
    found = merge_candidates(NS, "k", key_slot, scores, similar, limit=2)
    # Key, then slot, then similar; best score first; then by key. A slot
    # match without a vector has no score and comes after the scored ones
    assert [(c.key, c.reason, c.score) for c in found] == [
        ("k", "key", 0.2),
        ("s1", "slot", 0.75),
        ("s2", "slot", None),
        ("a", "similar", 0.95),
        ("b", "similar", 0.85),
    ]
    assert all(isinstance(c, Candidate) and c.item.namespace == NS for c in found)
    target = found[0]
    assert (target.version, target.provenance, target.confirmations) == (3, {"source": "k"}, 2)
    assert (target.valid_at, target.confirmed_at, target.item.value) == (T0, None, {"text": "k"})

    # The limit caps the similar candidates only: key and slot matches stay
    assert [c.key for c in merge_candidates(NS, "k", key_slot, scores, similar, limit=0)] == [
        "k",
        "s1",
        "s2",
    ]
    assert [c.key for c in merge_candidates(NS, "k", key_slot, scores, similar, limit=9)] == [
        "k",
        "s1",
        "s2",
        "a",
        "b",
        "c",
        "d",
    ]
    # The target comes by its key alone: a similar row under the proposal's
    # key that the key read did not find is left out
    late = [[candidate_row("k", score=0.99), candidate_row("a", score=0.9)]]
    assert [c.key for c in merge_candidates(NS, "k", [], [], late, limit=5)] == ["a"]
    # Negated distances (l2, inner product) order the same way
    far_near = [[candidate_row("far", score=-0.9), candidate_row("near", score=-0.1)]]
    assert [c.key for c in merge_candidates(NS, "new", [], [], far_near, limit=5)] == [
        "near",
        "far",
    ]
    assert merge_candidates(NS, "new", [], [], [], limit=5) == ()


def test_default_valid_at_is_raised_to_a_later_candidate_update() -> None:
    rows = [
        candidate_row("old", updated_at=NOW - timedelta(days=1)),
        candidate_row("ahead", updated_at=NOW + timedelta(seconds=5)),
    ]
    found = merge_candidates(NS, "new", rows, [], [], limit=10)
    # A node whose clock runs ahead wrote "ahead" after the database's now():
    # the proposal's valid time is not older than that write
    assert default_valid_at(NOW, found) == NOW + timedelta(seconds=5)
    assert default_valid_at(NOW, [c for c in found if c.key == "old"]) == NOW
    assert default_valid_at(NOW, ()) == NOW


def test_default_valid_at_never_takes_a_candidate_valid_at() -> None:
    # A caller gave this memory a valid time a year ahead. It still counts as
    # newer than the proposal, so the default is not raised to it
    row = candidate_row(
        "future", updated_at=NOW - timedelta(hours=1), valid_at=NOW + timedelta(days=365)
    )
    found = merge_candidates(NS, "new", [row], [], [], limit=10)
    assert default_valid_at(NOW, found) == NOW
    assert found[0].valid_at > default_valid_at(NOW, found)


# A row of the index form of StoreSQL.similar_candidates when no candidate passes
NO_HIT: dict[str, object] = dict.fromkeys(
    [
        "key",
        "value",
        "created_at",
        "updated_at",
        "version",
        "valid_at",
        "provenance",
        "confirmed_at",
        "confirmations",
        "write_id",
        "score",
    ]
)


def window_rows(
    hits: list[dict[str, object]],
    *,
    size: int = 21,
    lowest: float | None = 0.0,
    held: int | None = None,
) -> list[dict[str, object]]:
    """The rows of an index scan: its hits, or one row without a candidate, with the window.

    ``held`` is the namespace's vectors counted up to over_fetch (default: the window's size).
    """
    window = {
        "window_size": size,
        "lowest_score": lowest,
        "namespace_vectors": size if held is None else held,
    }
    return [{**row, **window} for row in hits] or [{**NO_HIT, **window}]


class CandidateReads:
    """Answers the read groups of find_conflicts with canned rows, and logs them.

    The first group gets the database's now(), ``rows`` for the key and slot
    statement and ``vectors`` for the count when sent. In the second, score
    statements get ``scores`` and similar statements ``similar``. An index
    scan's rows carry a window of 21 vectors whose lowest score is
    ``lowest``. An exact scan run again gets ``exact``.
    """

    def __init__(
        self,
        *,
        rows: list[dict[str, object]],
        vectors: int = 0,
        scores: list[dict[str, object]] | None = None,
        similar: list[dict[str, object]] | None = None,
        lowest: float | None = 0.0,
        exact: list[dict[str, object]] | None = None,
    ) -> None:
        self.rows = rows
        self.vectors = vectors
        self.scores = scores or []
        self.similar = similar or []
        self.lowest = lowest
        self.exact = exact or []
        self.calls: list[tuple[str, list[tuple[str, list]], list[str] | None]] = []

    def __call__(
        self, statements: list[tuple[str, list]], label: str, session: list[str] | None = None
    ) -> list[list[dict[str, object]]]:
        self.calls.append((label, statements, session))
        if label == "store find conflicts":
            found = [[{"now": NOW}], self.rows]
            return [*found, [{"n": self.vectors}]] if len(statements) == 3 else found
        if label == "store find conflicts exact":
            return [self.exact for _ in statements]
        assert label == "store find conflicts similar"
        answers = []
        for sql, _ in statements:
            if sql.startswith("SELECT key,"):
                answers.append(self.scores)
            elif "window_stats" in sql:
                answers.append(window_rows(self.similar, lowest=self.lowest))
            else:
                answers.append(self.similar)
        return answers


def finding_store(
    reads: CandidateReads, *, index: dict | None = None, **settings: object
) -> tuple[CockroachDBStore, CountingEmbeddings]:
    """A store with an index whose reads go to ``reads``, after migration 2."""
    embeddings = CountingEmbeddings()
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": embeddings, "fields": ["text"], **(index or {})},  # type: ignore[typeddict-item]
        retry_policy=FAST,
        **settings,  # type: ignore[arg-type]
    )
    store._columns_ready = True  # migration 2 seen: no schema read
    store._read_all = reads  # type: ignore[method-assign,assignment]
    return store, embeddings


@pytest.mark.parametrize(("vectors", "scan"), [(5, "exact"), (6, "index")])
def test_find_conflicts_reads_twice_and_scans_by_the_count(vectors: int, scan: str) -> None:
    reads = CandidateReads(
        rows=[candidate_row("k", version=3), candidate_row("s")],
        vectors=vectors,
        scores=[{"key": "k", "score": 0.2}, {"key": "s", "score": 0.9}],
        similar=[candidate_row("a", score=0.95), candidate_row("b", score=0.85)],
    )
    store, embeddings = finding_store(reads, index={"beam_size": 16}, reconcile_scan_limit=5)
    conflicts = store.find_conflicts(
        NS, "k", {"seat": "x", "text": "aisle"}, slot="seat", filter={"source": "chat"}
    )
    # One embedding call, of the document form a write stores
    assert embeddings.calls == 1
    assert embeddings.texts == ["aisle"]
    [(first_label, first, first_session), (second_label, second, second_session)] = reads.calls
    # First the clock, the target and slot matches, and the namespace's
    # vectors counted up to one past the scan limit
    assert (first_label, first_session) == ("store find conflicts", None)
    assert first == [
        store.sql.now(),
        store.sql.key_slot_candidates(
            "user.alice", "k", slot_field="seat", slot_value_json='"x"', filter={"source": "chat"}
        ),
        store.sql.namespace_vector_count("user.alice", 6),
    ]
    # Then, with the scan the count chose, the scores of the key and slot
    # candidates and the similar ones, which leave those keys out
    vector = [5.0, 0.0, 0.0]  # CountingEmbeddings: the length of "aisle"
    assert second_label == "store find conflicts similar"
    assert second == [
        StoreSQL.bind_query_vector(store.sql.candidate_scores("user.alice", ["k", "s"]), vector),
        StoreSQL.bind_query_vector(
            store.sql.similar_candidates(
                "user.alice",
                threshold=0.8,
                limit=10,
                filter={"source": "chat"},
                scan=scan,  # type: ignore[arg-type]
                over_fetch=10 * 1 * 2 + 1,
                exclude=["k", "s"],
            ),
            vector,
        ),
    ]
    # Only the index scan sets the beam size, in its read's transaction
    assert second_session == (["SET LOCAL vector_search_beam_size = 16"] if scan == "index" else [])

    assert [(c.key, c.reason, c.score) for c in conflicts.candidates] == [
        ("k", "key", 0.2),
        ("s", "slot", 0.9),
        ("a", "similar", 0.95),
        ("b", "similar", 0.85),
    ]
    assert (conflicts.target_version, conflicts.target_updated_at) == (3, T0)
    proposal = conflicts.proposal
    assert (proposal.namespace, proposal.key, proposal.slot, proposal.slot_value) == (
        NS,
        "k",
        "seat",
        "x",
    )
    assert proposal.value == {"seat": "x", "text": "aisle"} and proposal.provenance is None
    # No valid_at was given: the database's now(), as no candidate is newer
    assert proposal.valid_at == NOW
    state = conflicts._state
    assert state is not None
    assert state.plan.valid_at is None and state.plan.key == "k"
    assert state.vector_rows == (("user.alice", "k", "text", vector, None),)
    assert state.target == (3, T0)
    assert state.tokens == {"k": (3, T0), "s": (1, T0), "a": (1, T0), "b": (1, T0)}
    assert state.store == StoreIdentity(
        items="public.store", dims=3, distance="cosine", fields=("text",)
    )


def test_find_conflicts_skips_the_reads_it_does_not_need() -> None:
    # No text to embed: no embedding call, no count and no second read
    reads = CandidateReads(rows=[candidate_row("s")])
    store, embeddings = finding_store(reads)
    conflicts = store.find_conflicts(NS, None, {"seat": "x"}, slot="seat")
    assert embeddings.calls == 0
    [(_, first, _)] = reads.calls
    assert first[0] == store.sql.now() and len(first) == 2
    assert [(c.key, c.score) for c in conflicts.candidates] == [("s", None)]
    assert conflicts.target_version is None and conflicts.target_updated_at is None
    assert conflicts._state is not None and conflicts._state.target is None

    # index=False on the call: nothing to embed either
    reads = CandidateReads(rows=[])
    store, embeddings = finding_store(reads)
    store.find_conflicts(NS, "k", {"text": "t"}, index=False)
    assert embeddings.calls == 0 and len(reads.calls) == 1

    # limit=0 searches no similar candidates, so it counts nothing, but it
    # still scores the key and slot candidates
    reads = CandidateReads(rows=[candidate_row("s")], scores=[{"key": "s", "score": 0.5}])
    store, _ = finding_store(reads)
    conflicts = store.find_conflicts(NS, None, {"seat": "x", "text": "t"}, slot="seat", limit=0)
    [(_, first, _), (_, second, _)] = reads.calls
    assert len(first) == 2
    assert [sql.split("\n")[0] for sql, _ in second] == [
        "SELECT key, 1 - min(embedding <=> CAST(%s AS VECTOR)) AS score"
    ]
    assert [(c.key, c.score) for c in conflicts.candidates] == [("s", 0.5)]

    # Nothing found and limit=0: one read
    reads = CandidateReads(rows=[])
    store, _ = finding_store(reads)
    assert store.find_conflicts(NS, "k", {"text": "t"}, limit=0).candidates == ()
    assert len(reads.calls) == 1

    # A flat store has no index to choose: no count, and the exact scan
    reads = CandidateReads(rows=[], similar=[candidate_row("a", score=0.9)])
    store, _ = finding_store(reads, index={"flat": True})
    conflicts = store.find_conflicts(NS, "k", {"text": "t"})
    [(_, first, _), (_, [similar], _)] = reads.calls
    assert len(first) == 2
    assert "GROUP BY i.prefix, i.key" in similar[0]
    assert [c.key for c in conflicts.candidates] == ["a"]


def test_find_conflicts_takes_an_explicit_valid_at_as_given() -> None:
    newer = candidate_row("s", updated_at=NOW + timedelta(minutes=1))
    reads = CandidateReads(rows=[newer])
    store, _ = finding_store(reads)
    old = datetime(2025, 6, 1, tzinfo=timezone.utc)
    conflicts = store.find_conflicts(NS, None, {"seat": "x"}, slot="seat", valid_at=old)
    assert conflicts.proposal.valid_at == old
    assert conflicts._state is not None and conflicts._state.plan.valid_at == old
    # Without one, the valid time is raised to the newer candidate's update
    reads = CandidateReads(rows=[newer])
    store, _ = finding_store(reads)
    conflicts = store.find_conflicts(NS, None, {"seat": "x"}, slot="seat")
    assert conflicts.proposal.valid_at == NOW + timedelta(minutes=1)


def test_find_conflicts_and_record_check_everything_before_any_io() -> None:
    embeddings = CountingEmbeddings()
    # A connection the store must never touch: using it would raise
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": embeddings, "fields": ["text"], "distance_type": "l2"},
        retry_policy=FAST,
    )
    for call in (
        lambda: store.find_conflicts((), "k", TEXT, threshold=-1.0),
        lambda: store.find_conflicts(NS, "k", TEXT),
        lambda: store.find_conflicts(NS, "k", TEXT, threshold=-1.0, slot="seat"),
        lambda: store.find_conflicts(NS, "k", TEXT, threshold=-1.0, limit=-1),
        lambda: store.find_conflicts(NS, "k", TEXT, threshold=-1.0, valid_at=datetime(2026, 1, 1)),
        lambda: store.find_conflicts(NS, "k", TEXT, threshold=-1.0, provenance="chat"),  # type: ignore[arg-type]
        lambda: store.record((), "k"),
        lambda: store.record(("a.b",), "k"),
    ):
        with pytest.raises(ValueError):
            call()
    assert embeddings.calls == 0

    # Without migration 2 the schema check refuses, before anything is embedded
    def no_columns(*, refresh: bool = False) -> bool:
        assert refresh, "a method that needs migration 2 reads the schema again"
        return False

    store._reconcile_columns = no_columns  # type: ignore[method-assign,assignment]
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.find_conflicts(NS, "k", TEXT, threshold=-1.0)
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.record(NS, "k")
    assert embeddings.calls == 0


async def test_reconcile_scan_limit_is_a_positive_int_on_both_stores() -> None:
    assert CockroachDBStore(object()).reconcile_scan_limit == 2000  # type: ignore[arg-type]
    assert CockroachDBStore(object(), reconcile_scan_limit=1).reconcile_scan_limit == 1  # type: ignore[arg-type]
    pool = AsyncConnectionPool("", open=False)
    assert AsyncCockroachDBStore(pool, reconcile_scan_limit=7).reconcile_scan_limit == 7
    for bad in (0, -1, 1.5, True, "10", None):
        with pytest.raises(ValueError, match="reconcile_scan_limit"):
            CockroachDBStore(object(), reconcile_scan_limit=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reconcile_scan_limit"):
            AsyncCockroachDBStore(pool, reconcile_scan_limit=bad)  # type: ignore[arg-type]


def test_needs_exact_candidates() -> None:
    hit = candidate_row("a", score=0.9)
    other = candidate_row("b", score=0.88)

    def needs(
        rows: list[dict[str, object]],
        *,
        threshold: float = 0.8,
        limit: int = 10,
        over_fetch: int = 21,
    ) -> bool:
        return needs_exact_candidates(rows, threshold=threshold, limit=limit, over_fetch=over_fetch)

    # The window's lowest score is below the threshold: every vector beyond it
    # scores lower still, up to the index's approximation, so nothing is missing
    assert not needs(window_rows([], lowest=0.5))
    assert not needs(window_rows([hit], lowest=0.79))
    # At or above the threshold, matches may lie beyond the window: the filter,
    # the expiry check or the excluded keys may have taken every place in it
    assert needs(window_rows([], lowest=0.8))
    assert needs(window_rows([], lowest=0.95))
    assert needs(window_rows([hit], lowest=0.85))
    # Unless the window already filled the limit: a vector beyond it scores no
    # higher than its lowest, so it cannot outrank any of those hits
    assert not needs(window_rows([hit, other], lowest=0.85), limit=2)
    assert not needs(window_rows([hit], lowest=0.95), limit=1)
    # Fewer hits than the limit with the lowest score at or above the
    # threshold: the exact scan runs
    assert needs(window_rows([hit], lowest=0.85), limit=2)
    assert needs(window_rows([hit, other], lowest=0.8), limit=3)
    # A window the index cut short, with fewer rows than over_fetch while the
    # namespace holds more: vectors it never reached may score higher than its
    # lowest, so it runs again unless it filled the limit
    assert needs(window_rows([], size=15, lowest=0.5, held=21))
    assert needs(window_rows([hit], size=15, lowest=0.5, held=21), limit=2)
    assert not needs(window_rows([hit], size=15, lowest=0.5, held=21), limit=1)
    # A window smaller than over_fetch because the namespace holds no more is
    # whole: the lowest score decides
    assert not needs(window_rows([hit], size=5, lowest=0.5, held=5))
    assert needs(window_rows([hit], size=5, lowest=0.85, held=5))
    # An empty window tells nothing about the namespace
    assert needs(window_rows([], size=0, lowest=None))
    assert needs([])
    # Negated distances (l2, inner product) compare the same way
    assert not needs(window_rows([], lowest=-0.9), threshold=-0.6)
    assert needs(window_rows([], lowest=-0.5), threshold=-0.6)


def test_find_conflicts_runs_the_exact_scan_where_the_window_may_hide_matches() -> None:
    # Two vectors: "aaaaa" embeds to [5, 0, 0] and "bbbbbbb" to [7, 0, 0]. The
    # first vector's window ends above the threshold with no hit, so it may
    # hide matches; the second's ends below it
    def windows(statements: list[tuple[str, list]]) -> list[list[dict[str, object]]]:
        by_vector = {
            "[5.0,0.0,0.0]": window_rows([], lowest=0.9),
            "[7.0,0.0,0.0]": window_rows([candidate_row("b", score=0.85)], lowest=0.3),
        }
        return [by_vector[params[0]] for _, params in statements]

    calls: list[tuple[str, list[tuple[str, list]], list[str] | None]] = []

    def read_all(
        statements: list[tuple[str, list]], label: str, session: list[str] | None = None
    ) -> list[list[dict[str, object]]]:
        calls.append((label, statements, session))
        if label == "store find conflicts":
            return [[{"now": NOW}], [], [{"n": 6}]]
        if label == "store find conflicts similar":
            return windows(statements)
        assert label == "store find conflicts exact"
        return [[candidate_row("a", score=0.82)] for _ in statements]

    store, _ = finding_store(
        read_all,  # type: ignore[arg-type]
        index={"fields": ["text", "note"]},
        reconcile_scan_limit=5,
    )
    conflicts = store.find_conflicts(NS, "k", {"text": "aaaaa", "note": "bbbbbbb"})

    assert [label for label, _, _ in calls] == [
        "store find conflicts",
        "store find conflicts similar",
        "store find conflicts exact",
    ]
    # One more round trip, with the exact form of the first vector's statement only
    exact = store.sql.similar_candidates(
        "user.alice",
        threshold=0.8,
        limit=10,
        filter=None,
        scan="exact",
        over_fetch=10 * 2 * 2 + 1,
        exclude=["k"],
    )
    _, sent, session = calls[2]
    assert sent == [StoreSQL.bind_query_vector(exact, [5.0, 0.0, 0.0])]
    assert session is None
    # The exact rows replace that window's empty result; the other window's hit stays
    assert [(c.key, c.reason, c.score) for c in conflicts.candidates] == [
        ("b", "similar", 0.85),
        ("a", "similar", 0.82),
    ]

    # When every window ends below the threshold, the index result stands
    reads = CandidateReads(rows=[], vectors=6, similar=[candidate_row("b", score=0.85)])
    store, _ = finding_store(reads, reconcile_scan_limit=5)
    conflicts = store.find_conflicts(NS, "k", {"text": "aisle"})
    assert [label for label, _, _ in reads.calls] == [
        "store find conflicts",
        "store find conflicts similar",
    ]
    assert [c.key for c in conflicts.candidates] == ["b"]

    # A window that ends above the threshold but already filled the limit:
    # nothing beyond it can outrank its hits, so the index result stands too
    hits = [candidate_row("a", score=0.95), candidate_row("b", score=0.9)]
    reads = CandidateReads(rows=[], vectors=6, similar=hits, lowest=0.85)
    store, _ = finding_store(reads, reconcile_scan_limit=5)
    conflicts = store.find_conflicts(NS, "k", {"text": "aisle"}, limit=2)
    assert [label for label, _, _ in reads.calls] == [
        "store find conflicts",
        "store find conflicts similar",
    ]
    assert [c.key for c in conflicts.candidates] == ["a", "b"]
    # One hit short of the limit, the exact scan runs
    reads = CandidateReads(rows=[], vectors=6, similar=hits, lowest=0.85)
    store, _ = finding_store(reads, reconcile_scan_limit=5)
    store.find_conflicts(NS, "k", {"text": "aisle"}, limit=3)
    assert [label for label, _, _ in reads.calls][-1] == "store find conflicts exact"

    # A window that ends above the threshold with nothing beyond it: the exact
    # scan finds nothing either, and the empty row of the window is no candidate
    reads = CandidateReads(rows=[], vectors=6, lowest=0.9)
    store, _ = finding_store(reads, reconcile_scan_limit=5)
    assert store.find_conflicts(NS, "k", {"text": "aisle"}).candidates == ()
    assert [label for label, _, _ in reads.calls][-1] == "store find conflicts exact"


def test_conflicts_record_which_store_made_them() -> None:
    def identity(**settings: object) -> StoreIdentity:
        store = CockroachDBStore(object(), retry_policy=FAST, **settings)  # type: ignore[arg-type]
        return store._identity()

    index = {"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"]}
    base = identity(index=index, table_name="memories")
    assert base == StoreIdentity(
        items="public.memories", dims=3, distance="cosine", fields=("text",)
    )
    # The same table and index make the same store, whatever else differs
    assert identity(index={**index, "embed": CountingEmbeddings()}, table_name="memories") == base
    assert identity(index=index, table_name="memories", history=True) == base
    # Another table, schema, dims, distance or fields make another store
    others = [
        identity(index=index, table_name="other"),
        identity(index=index, table_name="memories", schema="app"),
        identity(index={**index, "dims": 4}, table_name="memories"),
        identity(index={**index, "distance_type": "l2"}, table_name="memories"),
        identity(index={**index, "fields": ["text", "note"]}, table_name="memories"),
        identity(table_name="memories"),
    ]
    assert all(other != base for other in others)
    assert len(set(others)) == len(others)
    assert others[-1] == StoreIdentity(
        items="public.memories", dims=None, distance=None, fields=None
    )


def test_conflicts_survive_a_checkpoint_round_trip() -> None:
    # An interrupt while a person decides keeps the Conflicts in a LangGraph
    # checkpoint. The serializer gives tuples back as lists; the types restore them
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    reads = CandidateReads(
        rows=[candidate_row("k", version=3), candidate_row("s")],
        vectors=6,
        scores=[{"key": "k", "score": 0.2}, {"key": "s", "score": 0.9}],
        similar=[candidate_row("a", score=0.95)],
    )
    store, _ = finding_store(reads, reconcile_scan_limit=5, ttl={"default_ttl": 30.0})
    conflicts = store.find_conflicts(
        NS,
        "k",
        {"seat": "x", "text": "aisle"},
        slot="seat",
        index=["text"],
        provenance={"source": "chat"},
        valid_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    serde = JsonPlusSerializer()
    back = serde.loads_typed(serde.dumps_typed(conflicts))

    assert back == conflicts
    assert back._state == conflicts._state
    state, plan = back._state, back._state.plan
    assert isinstance(back.candidates, tuple) and isinstance(back.proposal.namespace, tuple)
    assert state.store == store._identity() and isinstance(state.store.fields, tuple)
    assert state.target == (3, T0) and state.tokens["s"] == (1, T0)
    assert state.vector_rows == (("user.alice", "k", "text", [5.0, 0.0, 0.0], 30.0),)
    assert plan.namespace == NS and plan.index == ["text"]
    assert plan.put.items == [("user.alice", "k", plan.value_json, 30.0)]
    assert plan.put.embed_requests == [("user.alice", "k", "text", "aisle")]
    # Nested values keep their types: aware datetimes, the write id and the TTL
    assert plan.valid_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert back.candidates[0].valid_at.tzinfo is not None
    assert back.target_updated_at == T0
    assert (
        isinstance(plan.put.write_id, str)
        and plan.put.write_id == conflicts._state.plan.put.write_id
    )
    assert isinstance(plan.ttl, float) and plan.ttl == 30.0
    assert back.candidates[0].item.score == 0.2


@pytest.mark.parametrize(
    "bad",
    [
        {"reconcile_scan_limit": 0},
        {"history": {"ttl_days": -1}},
        {"index": {"dims": 0, "embed": CountingEmbeddings()}},
    ],
    ids=["scan-limit", "history", "index"],
)
@pytest.mark.parametrize("pool_config", [None, {"min_size": 1}], ids=["connection", "pool"])
def test_from_conn_string_checks_settings_before_connecting(
    bad: dict, pool_config: dict | None, monkeypatch
) -> None:
    from langchain_cockroachdb.store import store as store_module

    def connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("from_conn_string connected before checking its settings")

    monkeypatch.setattr(store_module.Connection, "connect", connect)
    monkeypatch.setattr(store_module, "ConnectionPool", connect)
    with (
        pytest.raises(ValueError),
        CockroachDBStore.from_conn_string(
            "postgresql://root@localhost:1/db", pool_config=pool_config, **bad
        ),
    ):
        pass


@pytest.mark.parametrize(
    "bad",
    [
        {"reconcile_scan_limit": 0},
        {"history": {"ttl_days": -1}},
        {"index": {"dims": 0, "embed": CountingEmbeddings()}},
    ],
    ids=["scan-limit", "history", "index"],
)
@pytest.mark.parametrize("pool_config", [None, {"min_size": 1}], ids=["connection", "pool"])
async def test_async_from_conn_string_checks_settings_before_connecting(
    bad: dict, pool_config: dict | None, monkeypatch
) -> None:
    from langchain_cockroachdb.store import aio as aio_module

    async def connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("from_conn_string connected before checking its settings")

    def pool(*args: object, **kwargs: object) -> None:
        raise AssertionError("from_conn_string opened a pool before checking its settings")

    monkeypatch.setattr(aio_module.AsyncConnection, "connect", connect)
    monkeypatch.setattr(aio_module, "AsyncConnectionPool", pool)
    with pytest.raises(ValueError):
        async with AsyncCockroachDBStore.from_conn_string(
            "postgresql://root@localhost:1/db", pool_config=pool_config, **bad
        ):
            pass


def test_from_conn_string_builds_the_embedding_model_once(monkeypatch) -> None:
    # The settings are checked before connecting, and the store reuses the
    # model that check built, so a model named by a string is built once
    from langchain_cockroachdb.store import config as config_module
    from langchain_cockroachdb.store import store as store_module

    built: list[object] = []

    def ensure_embeddings(embed: object) -> Embeddings:
        if isinstance(embed, Embeddings):
            return embed
        built.append(embed)
        return CountingEmbeddings()

    class FakeConnection:
        closed = False

        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(config_module, "ensure_embeddings", ensure_embeddings)
    monkeypatch.setattr(
        store_module.Connection, "connect", lambda *args, **kwargs: FakeConnection()
    )
    with CockroachDBStore.from_conn_string(
        "postgresql://root@localhost:1/db",
        index={"dims": 3, "embed": "fake:model"},  # type: ignore[typeddict-item]
    ) as store:
        assert isinstance(store.index.embeddings, CountingEmbeddings)  # type: ignore[union-attr]
    assert built == ["fake:model"]
