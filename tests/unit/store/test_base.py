"""Unit tests for the store's shared planning code. No database."""

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
from langchain_cockroachdb.store.base import BaseCockroachDBStore, PutPlan, _group_ops
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
        statements, to_embed = store._search_plan(ops)
        store._search_plan(ops)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no index" in warnings[0].getMessage()
    assert to_embed == []
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


def test_batch_plan_keeps_search_queries_to_embed() -> None:
    store = make_base(fields=["text"])
    plan = store._plan_batch(
        [SearchOp(("a",), query="hello", limit=3), SearchOp(("b",)), SearchOp(("c",), query="x")]
    )
    assert plan.search_to_embed == [(0, "hello"), (2, "x")]
    assert make_base()._plan_batch([SearchOp(("a",), query="q")]).search_to_embed == []
