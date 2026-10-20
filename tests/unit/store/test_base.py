"""Unit tests for the store's shared planning code. No database."""

import asyncio
import contextlib
import logging
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


class SearchConnection:
    """A fake connection for read groups. Every statement and its params is logged.

    Namespace lookups return ``namespaces``; other statements return no rows.
    The first attempt of the "store search" group fails with a retryable 40001
    as its block ends, after its statements were sent.
    """

    def __init__(self, namespaces: list[str]) -> None:
        self.namespaces = namespaces
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
    assert first[1][:3] == ["[1.0,0.0,0.0]", ["a", "a.x"], "[1.0,0.0,0.0]"]
    # over_fetch = (limit + offset) * estimated_vectors * 2 + 1
    assert first[1][3] == (2 + 3) * 1 * 2 + 1
    assert second[1][1] == ["a", "a.x"]
    # A search of the whole store has no namespace condition and no lookup
    assert "ANY(" not in root[0] and root[1][0] == "[1.0,0.0,0.0]"
    assert "embedding" not in plain[0]
    for _, params in searches:
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
    assert conn.attempts == {"store search namespaces": 1, "store search": 2}
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
    assert conn.attempts == {"store search namespaces": 1, "store search": 2}
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
    assert transactions == {
        "store search namespaces": False,
        "store get": False,
        "store search": True,
    }
    # A batch without a query needs no beam size
    conn.session.clear()
    conn.attempts.clear()
    store.batch([SearchOp(("a",))])
    assert conn.session == []


def searches_with_namespaces(count: int) -> list[tuple[str, list]]:
    """The search statements sent when the lookup finds ``count`` namespaces."""
    store = CockroachDBStore(
        object(),  # type: ignore[arg-type]
        index={"dims": 3, "embed": CountingEmbeddings(), "fields": ["text"]},
        retry_policy=FAST,
    )
    conn = SearchConnection([f"a.n{i}" for i in range(count)])
    conn.attempts["store search"] = 1  # no injected failure

    @contextlib.contextmanager
    def fake_cursor(**kwargs: object) -> Iterator[SearchConnection]:
        yield conn.enter(kwargs)
        conn.exit()

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    store.batch([SearchOp(("a",), query="q", filter={"x": 1})])
    return [(sql, params) for label, sql, params in conn.executed if label == "store search"]


def test_too_many_namespaces_fall_back_to_one_statement() -> None:
    assert MAX_SEARCH_NAMESPACES == 1000
    [capped] = searches_with_namespaces(MAX_SEARCH_NAMESPACES)
    assert "prefix = ANY(CAST(%s AS STRING[]))" in capped[0]
    assert len(capped[1][1]) == MAX_SEARCH_NAMESPACES

    [fallback] = searches_with_namespaces(MAX_SEARCH_NAMESPACES + 1)
    # The scan form: the namespace range in the vectors CTE, no array
    assert "ANY(" not in fallback[0]
    assert "(prefix = %s OR (prefix >= %s AND prefix < %s))" in fallback[0]
    vector = "[1.0,0.0,0.0]"
    # over_fetch for the default limit of 10: 10 * 1 * 2 + 1
    assert fallback[1] == [vector, "a", "a.", "a/", vector, 21, "x", "1", 10, 0]


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
