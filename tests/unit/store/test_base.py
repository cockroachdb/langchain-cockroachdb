"""Unit tests for the store's shared planning code. No database."""

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone

import psycopg
import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb.retry import RetryPolicy
from langchain_cockroachdb.store.aio import AsyncCockroachDBStore
from langchain_cockroachdb.store.base import (
    MAX_SEARCH_NAMESPACES,
    BaseCockroachDBStore,
    PutPlan,
    _group_ops,
    needs_exact_search,
)
from langchain_cockroachdb.store.config import resolve_index_config
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


def make_base(*, fields: list[str] | None = None, ttl=None) -> BaseCockroachDBStore:
    """BaseCockroachDBStore state built directly, without a connection."""
    store = BaseCockroachDBStore()
    index = None
    if fields is not None:
        index = resolve_index_config({"dims": 3, "embed": CountingEmbeddings(), "fields": fields})
    store._configure(index=index, ttl=ttl, table_name="store", schema="public")
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


def search_row(key: str, score: float, window_size: int | None = None) -> dict[str, object]:
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
    return row


def test_needs_exact_search() -> None:
    full, partial = 21, 5
    short = [search_row("k1", 0.9, full)]
    # Short from a full window: candidates may have failed the filter or expired
    assert needs_exact_search(short, limit=10, over_fetch=full)
    # Short from a window that was not full: it held every vector, so it is exact
    assert not needs_exact_search([search_row("k1", 0.9, partial)], limit=10, over_fetch=full)
    # Nothing came back: the window size is unknown
    assert needs_exact_search([], limit=10, over_fetch=full)
    # A full page is complete, whatever the window
    page = [search_row(f"k{i}", 0.9, full) for i in range(10)]
    assert not needs_exact_search(page, limit=10, over_fetch=full)
    assert not needs_exact_search(page[:3], limit=3, over_fetch=full)


EXACT_ROWS = [search_row("best", 0.99), search_row("next", 0.5)]


@pytest.mark.parametrize(
    ("indexed_rows", "namespaces", "runs_again"),
    [
        ([search_row("k1", 0.9, 21)], ["a"], True),
        ([search_row("k1", 0.9, 20)], ["a"], False),
        ([search_row(f"k{i}", 0.9, 21) for i in range(10)], ["a"], False),
        ([], ["a"], True),
        ([], [], False),
    ],
    ids=["short-full-window", "short-partial-window", "full-page", "empty", "no-namespace"],
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
