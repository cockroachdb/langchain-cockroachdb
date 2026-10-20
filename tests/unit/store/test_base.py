"""Unit tests for the store's shared planning code. No database."""

import contextlib
import logging
from collections.abc import Iterator
from datetime import datetime, timezone

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

from langchain_cockroachdb.store.base import BaseCockroachDBStore, PutPlan, _group_ops
from langchain_cockroachdb.store.config import resolve_index_config
from langchain_cockroachdb.store.store import CockroachDBStore


class CountingEmbeddings(Embeddings):
    """Embeddings that count their calls and return fixed vectors."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
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


def test_index_is_not_supported_yet() -> None:
    with pytest.raises(NotImplementedError, match="semantic search arrives in a later task"):
        CockroachDBStore(
            object(),  # type: ignore[arg-type]
            index={"dims": 3, "embed": CountingEmbeddings()},
        )


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
