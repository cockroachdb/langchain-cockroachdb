"""find_conflicts, record and reconcile: the memories a new one may conflict with.

Candidates are live memories in the new memory's own namespace: the one
under its key, those in the same slot and, on a store with an index, those
whose vectors score at or above the threshold. Expired memories are never
candidates, and neither find_conflicts nor record pushes back an expiry.
Without a valid_at from the caller, the new memory is valid from the
database's clock.

reconcile writes a memory and settles its candidates in one serializable
transaction: it reads them, runs the policy, and supersedes, deletes,
confirms or keeps each one, keeping what it ends in history on a store with
history. A retry after a commit whose result was lost finds its own write
and returns the result rebuilt from the rows.

resolve applies a decision about what find_conflicts found, in one
transaction, once it has checked that the target and every candidate the
decision names still have the version and updated_at find_conflicts read.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import PutOp
from psycopg import Connection

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore, RetryPolicy
from langchain_cockroachdb._sql import parse_vector
from langchain_cockroachdb.store import (
    Action,
    Conflicts,
    ConflictsChanged,
    Decision,
    MemoryRecord,
    Policy,
    ReconcileResult,
    keep_all,
    skip_duplicates,
    supersede_same_slot,
    supersede_similar,
)
from langchain_cockroachdb.store import base as store_base
from langchain_cockroachdb.store import reconcile as store_reconcile
from langchain_cockroachdb.store import store as store_module
from langchain_cockroachdb.store.sql import StoreSQL
from tests.integration.faults import FailAfterCommit, PsycopgFaults
from tests.integration.store.conftest import SEATS, FixedEmbeddings, seat_index
from tests.integration.store.test_schema_upgrade import schema_0_5
from tests.integration.store.test_semantic_filters import STATS_WAIT, explain

NS = ("user", "alice")
PREFIX = "user.alice"
WINDOW = "Prefers window seats"
LONG_FLIGHTS = "Prefers window seats on long flights"  # 0.94 against AISLE
AISLE = "Prefers aisle seats"
VEGETARIAN = "Vegetarian"  # 0.0 against AISLE
PEANUTS = "Allergic to peanuts"  # 0.0 against AISLE


def text(value: str, **fields: Any) -> dict[str, Any]:
    return {"text": value, **fields}


def reasons(conflicts: Conflicts) -> list[tuple[str, str]]:
    return [(c.key, c.reason) for c in conflicts.candidates]


def item_rows(admin: Connection[Any], store: Any) -> list[dict[str, Any]]:
    return admin.execute(
        f"SELECT prefix, key, value, version, updated_at, expires_at FROM {store.sql.items} "
        "ORDER BY prefix, key"
    ).fetchall()


def expiries(admin: Connection[Any], store: Any) -> list[tuple[str, Any]]:
    """The expiry of every item and vector under PREFIX."""
    found = []
    for table in (store.sql.items, store.sql.vectors):
        rows = admin.execute(
            f"SELECT key, expires_at FROM {table} WHERE prefix = %s ORDER BY key", (PREFIX,)
        ).fetchall()
        found.extend((f"{table}/{row['key']}", row["expires_at"]) for row in rows)
    return found


def toward(base: list[float], cosine: float) -> list[float]:
    """A unit vector whose cosine similarity with ``base`` is ``cosine``.

    ``base`` lies in the first two dimensions; the third takes the rest.
    """
    norm = math.sqrt(sum(x * x for x in base))
    return [cosine * base[0] / norm, cosine * base[1] / norm, math.sqrt(1 - cosine**2), 0.0]


def dense_store(store_factory: Any, mode: str) -> CockroachDBStore:
    """A store whose index knows 29 texts nearer to AISLE than "match", which scores 0.85.

    "near 0" to "near 28" score 0.99 down to 0.962, all above the default
    threshold, so an index window of 21 holds none of "match". The scan
    limit is 10, so 30 memories take the index path.
    """
    vectors = {AISLE: SEATS[AISLE], "match": toward(SEATS[AISLE], 0.85)}
    for i in range(29):
        vectors[f"near {i}"] = toward(SEATS[AISLE], 0.99 - 0.001 * i)
    return store_factory(
        mode,
        index={"dims": 4, "embed": FixedEmbeddings(vectors), "fields": ["text"]},
        reconcile_scan_limit=10,
    )


def spy_on_reads(store: CockroachDBStore, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the label, SQL and session statements of every read group the store sends."""
    sent: list[Any] = []
    real = store._read_all

    def read_all(
        statements: list[tuple[str, list]], label: str, session: list[str] | None = None
    ) -> Any:
        sent.append((label, [sql for sql, _ in statements], session))
        return real(statements, label, session)

    monkeypatch.setattr(store, "_read_all", read_all)
    return sent


def test_record_returns_version_provenance_and_confirmations(store_factory, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", text(WINDOW))

    record = store.record(NS, "k")
    item = store.get(NS, "k")
    assert isinstance(record, MemoryRecord) and item is not None
    assert record.item == item
    assert (record.version, record.provenance, record.confirmations) == (1, None, 0)
    assert record.confirmed_at is None
    # A plain put stores no valid time: the memory is valid from its write
    assert record.valid_at == item.updated_at

    store.put(NS, "k", text(AISLE))
    record = store.record(NS, "k")
    assert record is not None
    assert (record.version, record.item.value) == (2, text(AISLE))
    assert store.record(NS, "missing") is None
    assert store.record(("user", "bob"), "k") is None


def test_find_conflicts_returns_key_slot_and_similar_in_that_order(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", text(PEANUTS))
    store.put(NS, "s", text(VEGETARIAN, seat="x"))
    store.put(NS, "s2", text(PEANUTS, seat="x"), index=False)  # same slot, no vector
    store.put(NS, "w", text(WINDOW))
    store.put(NS, "other", text(VEGETARIAN, seat="y"))
    # A descendant namespace and another user hold matches too
    store.put(("user", "alice", "trips"), "child", text(WINDOW, seat="x"))
    store.put(("user", "bob"), "bob", text(WINDOW, seat="x"))
    before = item_rows(admin, store)

    conflicts = store.find_conflicts(NS, "k", text(AISLE, seat="x"), slot="seat")

    assert reasons(conflicts) == [("k", "key"), ("s", "slot"), ("s2", "slot"), ("w", "similar")]
    k, s, s2, w = conflicts.candidates
    assert w.score == pytest.approx(0.85, abs=1e-3)
    # Key and slot matches carry their score when they have vectors
    assert k.score == pytest.approx(0.0, abs=1e-6)
    assert s.score == pytest.approx(0.0, abs=1e-6)
    assert s2.score is None
    assert all(c.item.namespace == NS for c in conflicts.candidates)
    item = store.get(NS, "k")
    assert item is not None
    assert (k.version, k.valid_at, k.provenance, k.confirmations) == (1, item.updated_at, None, 0)
    assert (k.item.value, k.item.updated_at, k.confirmed_at) == (
        text(PEANUTS),
        item.updated_at,
        None,
    )

    # The version and updated_at of the target and of every candidate, for resolve
    assert (conflicts.target_version, conflicts.target_updated_at) == (1, item.updated_at)
    assert conflicts._state is not None
    assert conflicts._state.target == (1, item.updated_at)
    assert conflicts._state.tokens == {
        c.key: (c.version, c.item.updated_at) for c in conflicts.candidates
    }
    proposal = conflicts.proposal
    assert (proposal.namespace, proposal.key, proposal.value) == (NS, "k", text(AISLE, seat="x"))
    assert (proposal.slot, proposal.slot_value, proposal.provenance) == ("seat", "x", None)
    # Nothing was written
    assert item_rows(admin, store) == before


def test_find_conflicts_ignores_expired_memories(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", text(WINDOW), ttl=60)
    store.put(NS, "s", text(VEGETARIAN, seat="x"), ttl=60)
    store.put(NS, "w", text(WINDOW), ttl=60)
    store.put(NS, "live", text(LONG_FLIGHTS))
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() - INTERVAL '1 minute' "
        "WHERE prefix = %s AND key <> 'live'",
        (PREFIX,),
    )

    conflicts = store.find_conflicts(NS, "k", text(AISLE, seat="x"), slot="seat")
    assert reasons(conflicts) == [("live", "similar")]
    # The expired target counts as absent, the same as no row
    assert conflicts.target_version is None and conflicts.target_updated_at is None
    assert conflicts._state is not None and conflicts._state.target is None
    assert store.record(NS, "k") is None
    assert store.record(NS, "live") is not None


def test_filter_narrows_slot_and_similar_but_not_the_target_key(store_factory, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", text(PEANUTS, source="import"))
    store.put(NS, "s1", text(VEGETARIAN, seat="x", source="chat"))
    store.put(NS, "s2", text(PEANUTS, seat="x", source="import"))
    store.put(NS, "w1", text(WINDOW, source="chat"))
    store.put(NS, "w2", text(LONG_FLIGHTS, source="import"))
    proposal = text(AISLE, seat="x", source="chat")

    filtered = store.find_conflicts(NS, "k", proposal, slot="seat", filter={"source": "chat"})
    assert reasons(filtered) == [("k", "key"), ("s1", "slot"), ("w1", "similar")]

    unfiltered = store.find_conflicts(NS, "k", proposal, slot="seat")
    assert reasons(unfiltered) == [
        ("k", "key"),
        ("s1", "slot"),
        ("s2", "slot"),
        ("w2", "similar"),
        ("w1", "similar"),
    ]


def test_limit_caps_similar_candidates_but_keeps_key_and_slot(store_factory, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", text(WINDOW))  # the target, similar at 0.85
    store.put(NS, "s", text(LONG_FLIGHTS, seat="x"))  # a slot match, similar at 0.94
    store.put(NS, "a", text(LONG_FLIGHTS))
    store.put(NS, "b", text(WINDOW))
    store.put(NS, "c", text(WINDOW))
    proposal = text(AISLE, seat="x")

    def found(limit: int) -> Conflicts:
        return store.find_conflicts(NS, "k", proposal, slot="seat", limit=limit)

    assert reasons(found(1)) == [("k", "key"), ("s", "slot"), ("a", "similar")]
    assert reasons(found(2)) == [("k", "key"), ("s", "slot"), ("a", "similar"), ("b", "similar")]
    assert reasons(found(10)) == [
        ("k", "key"),
        ("s", "slot"),
        ("a", "similar"),
        ("b", "similar"),
        ("c", "similar"),
    ]
    only = found(0)
    assert reasons(only) == [("k", "key"), ("s", "slot")]
    # They keep their scores, though no similar candidate is searched
    assert [c.score for c in only.candidates] == pytest.approx([0.85, 0.938], abs=1e-3)
    with pytest.raises(ValueError, match="limit"):
        store.find_conflicts(NS, "k", proposal, limit=-1)


def test_find_conflicts_uses_the_index_above_the_scan_limit(
    store_factory, mode, monkeypatch
) -> None:
    store = store_factory(mode, index=seat_index(beam_size=8), reconcile_scan_limit=2)
    for key, value in (
        ("w", WINDOW),
        ("lf", LONG_FLIGHTS),
        ("v", VEGETARIAN),
        ("p", PEANUTS),
        ("v2", VEGETARIAN),
    ):
        store.put(NS, key, text(value))
    sent = spy_on_reads(store, monkeypatch)

    conflicts = store.find_conflicts(NS, None, text(AISLE))
    assert reasons(conflicts) == [("lf", "similar"), ("w", "similar")]
    assert [c.score for c in conflicts.candidates] == pytest.approx([0.938, 0.85], abs=1e-3)
    # Five vectors, above the limit of 2: the C-SPANN index gives the window,
    # with the beam size set in the read's transaction
    [(_, [similar], session)] = [s for s in sent if s[0] == "store find conflicts similar"]
    assert "ORDER BY embedding <=> CAST(%s AS VECTOR)\n    LIMIT %s" in similar
    assert session == ["SET LOCAL vector_search_beam_size = 8"]
    # The window holds all five, down to scores below the threshold, so
    # nothing lies beyond it and the result stands
    assert [label for label, _, _ in sent] == [
        "store find conflicts",
        "store find conflicts similar",
    ]

    # At the limit every vector of the namespace is ranked, with the same result
    exact = store_factory(mode, index=seat_index(), reconcile_scan_limit=5)
    sent = spy_on_reads(exact, monkeypatch)
    assert reasons(exact.find_conflicts(NS, None, text(AISLE))) == [
        ("lf", "similar"),
        ("w", "similar"),
    ]
    [(_, [similar], session)] = [s for s in sent if s[0] == "store find conflicts similar"]
    assert "GROUP BY i.prefix, i.key" in similar and session == []


def test_find_conflicts_looks_past_a_window_the_filter_emptied(
    store_factory, mode, monkeypatch
) -> None:
    # Twenty-nine memories of another kind lie nearer to the new memory than
    # the only match, all above the threshold. The index's window of 21 holds
    # none of the match's kind and still ends above the threshold, so matches
    # may lie beyond it, and the store ranks the namespace exactly
    store = dense_store(store_factory, mode)
    store.batch(
        [PutOp(NS, f"o{i}", {"text": f"near {i}", "kind": "other"}) for i in range(29)]
        + [PutOp(NS, "m", {"text": "match", "kind": "seat"})]
    )
    sent = spy_on_reads(store, monkeypatch)

    conflicts = store.find_conflicts(NS, None, text(AISLE, kind="seat"), filter={"kind": "seat"})

    assert reasons(conflicts) == [("m", "similar")]
    assert conflicts.candidates[0].score == pytest.approx(0.85, abs=1e-3)
    assert [label for label, _, _ in sent] == [
        "store find conflicts",
        "store find conflicts similar",
        "store find conflicts exact",
    ]
    [(_, [indexed], _), (_, [exact], session)] = sent[1:]
    assert "window_stats" in indexed
    assert "GROUP BY i.prefix, i.key" in exact and session is None


def test_find_conflicts_looks_past_a_window_of_excluded_slot_matches(
    store_factory, mode, monkeypatch
) -> None:
    # Twenty-nine memories in the new memory's slot lie nearer to it than a
    # similar memory outside the slot. The similar statement leaves the slot
    # matches out, since they are candidates already, but they still fill the
    # index's window
    store = dense_store(store_factory, mode)
    store.batch(
        [PutOp(NS, f"s{i:02}", {"text": f"near {i}", "seat": "x"}) for i in range(29)]
        + [PutOp(NS, "m", {"text": "match"})]
    )
    sent = spy_on_reads(store, monkeypatch)

    conflicts = store.find_conflicts(NS, None, text(AISLE, seat="x"), slot="seat")

    found = reasons(conflicts)
    assert found[:29] == [(f"s{i:02}", "slot") for i in range(29)]
    assert found[29:] == [("m", "similar")]
    assert [label for label, _, _ in sent][-1] == "store find conflicts exact"


def test_find_conflicts_looks_past_a_window_of_expired_memories(
    store_factory, admin, mode, monkeypatch
) -> None:
    # Twenty-nine expired memories lie nearer than a live match. Their vectors
    # still fill the index's window until the TTL job deletes them
    store = dense_store(store_factory, mode)
    store.batch(
        [PutOp(NS, f"e{i:02}", {"text": f"near {i}"}, ttl=60) for i in range(29)]
        + [PutOp(NS, "m", {"text": "match"})]
    )
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() - INTERVAL '1 minute' "
        "WHERE prefix = %s AND key <> 'm'",
        (PREFIX,),
    )
    sent = spy_on_reads(store, monkeypatch)

    conflicts = store.find_conflicts(NS, None, text(AISLE))

    assert reasons(conflicts) == [("m", "similar")]
    assert [label for label, _, _ in sent][-1] == "store find conflicts exact"


def test_only_the_index_form_uses_the_vector_index(store_factory, admin) -> None:
    """EXPLAIN: the index form takes its window from a vector search, the exact form never does.

    nearest has two readers, the window's figures and the hits, so
    CockroachDB computes it once into a buffer; a vector search on the
    C-SPANN index fills that buffer. The namespace's vectors are counted from
    the primary key, up to over_fetch. As in test_semantic_filters, the index
    wins on cost with a few thousand vectors in the namespace.
    """
    model = DeterministicFakeEmbedding(size=4)
    store = store_factory("pool", index={"dims": 4, "embed": model, "fields": ["text"]})
    store.batch(
        [
            PutOp(NS, f"k{i:04}", {"text": f"memory {i}", "kind": "rare" if i % 25 == 3 else "c"})
            for i in range(3000)
        ]
    )
    admin.execute(f"ANALYZE {store.sql.items}")
    admin.execute(f"ANALYZE {store.sql.vectors}")
    query = model.embed_query("a memory")
    common: dict[str, Any] = {
        "threshold": 0.8,
        "limit": 3,
        "filter": {"kind": "rare"},
        "over_fetch": 7,
        "exclude": ["k0003"],
    }

    def bound(scan: str) -> tuple[str, list]:
        statement = store.sql.similar_candidates(PREFIX, scan=scan, **common)  # type: ignore[arg-type]
        return StoreSQL.bind_query_vector(statement, query)

    # Each node picks up new statistics a few seconds after ANALYZE, so plan
    # again until the optimizer sees them, as test_semantic_filters does
    statement = bound("index")
    deadline = time.monotonic() + STATS_WAIT
    while "vector search" not in (indexed := explain(admin, statement)):
        if time.monotonic() > deadline:
            break
        time.sleep(1)
    assert "vector search" in indexed, (
        f"no vector search after {STATS_WAIT} s of planning again:\n{indexed}"
    )
    assert store.sql.vector_index_name in indexed, indexed
    assert "label: buffer 1 (nearest)" in indexed, indexed
    count = indexed[indexed.index("original sql: (SELECT count(*)") :]
    vectors = store.sql.vectors.split(".", 1)[1]
    assert f"table: {vectors}@{vectors}_pkey" in count, indexed
    assert f"spans: [/'{PREFIX}' - /'{PREFIX}']" in count, indexed
    assert "limit: 7" in count, indexed
    exact = explain(admin, bound("exact"))
    assert "vector search" not in exact, exact


def test_slot_matches_come_best_score_first(store_factory, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "s0", text(VEGETARIAN, seat="x"))
    store.put(NS, "s1", text(LONG_FLIGHTS, seat="x"))
    store.put(NS, "s2", text(WINDOW, seat="x"))
    store.put(NS, "s3", text(PEANUTS, seat="x"), index=False)

    conflicts = store.find_conflicts(NS, None, text(AISLE, seat="x"), slot="seat")

    # Key order alone would give s0, s1, s2, s3; a match without a vector comes last
    assert reasons(conflicts) == [("s1", "slot"), ("s2", "slot"), ("s0", "slot"), ("s3", "slot")]
    scores = [c.score for c in conflicts.candidates]
    assert scores[:3] == pytest.approx([0.938, 0.85, 0.0], abs=1e-3)
    assert scores[3] is None


def test_inner_product_store_finds_candidates_with_a_threshold(store_factory, mode) -> None:
    store = store_factory(mode, index=seat_index(distance_type="inner_product"))
    store.put(NS, "w", text(WINDOW))
    store.put(NS, "lf", text(LONG_FLIGHTS))
    store.put(NS, "v", text(VEGETARIAN))
    with pytest.raises(ValueError, match="threshold"):
        store.find_conflicts(NS, None, text(AISLE))

    # The score is the negated distance, which is the inner product here
    conflicts = store.find_conflicts(NS, None, text(AISLE), threshold=0.5)
    assert reasons(conflicts) == [("lf", "similar"), ("w", "similar")]
    assert [c.score for c in conflicts.candidates] == pytest.approx([0.938, 0.85], abs=1e-3)
    searched = {item.key: item.score for item in store.search(NS, query=AISLE)}
    assert [c.score for c in conflicts.candidates] == pytest.approx([searched["lf"], searched["w"]])


def test_cosine_candidate_scores_equal_search_scores_for_the_same_vectors(
    store_factory, mode
) -> None:
    # FixedEmbeddings gives queries and documents the same vectors, so the
    # store's score function must give find_conflicts and search the same scores:
    # for a slot match, scored by its vectors alone, and for similar memories
    store = store_factory(mode, index=seat_index())
    store.put(NS, "s", text(WINDOW, seat="x"))
    store.put(NS, "lf", text(LONG_FLIGHTS))
    store.put(NS, "v", text(VEGETARIAN))

    conflicts = store.find_conflicts(NS, None, text(AISLE, seat="x"), slot="seat")

    assert reasons(conflicts) == [("s", "slot"), ("lf", "similar")]
    searched = {item.key: item.score for item in store.search(NS, query=AISLE)}
    assert {c.key: c.score for c in conflicts.candidates} == pytest.approx(
        {"s": searched["s"], "lf": searched["lf"]}
    )


def test_store_without_index_finds_key_and_slot_candidates(store_factory, mode) -> None:
    store = store_factory(mode)
    store.put(NS, "k", text(PEANUTS))
    store.put(NS, "s", text(VEGETARIAN, seat="x"))
    store.put(NS, "w", text(WINDOW))
    # No threshold is needed, and one given is ignored
    for threshold in (None, 5.0):
        conflicts = store.find_conflicts(
            NS, "k", text(AISLE, seat="x"), slot="seat", threshold=threshold
        )
        assert reasons(conflicts) == [("k", "key"), ("s", "slot")]
        assert [c.score for c in conflicts.candidates] == [None, None]
        assert conflicts._state is not None and conflicts._state.vector_rows == ()


def test_l2_store_requires_a_threshold(store_factory, mode) -> None:
    index = seat_index(distance_type="l2")
    store = store_factory(mode, index=index)
    store.put(NS, "w", text(WINDOW))
    store.put(NS, "lf", text(LONG_FLIGHTS))
    store.put(NS, "v", text(VEGETARIAN))
    index["embed"].calls = 0

    with pytest.raises(ValueError, match="threshold"):
        store.find_conflicts(NS, None, text(AISLE))
    assert index["embed"].calls == 0

    # Scores are negated l2 distances, the ones search returns
    conflicts = store.find_conflicts(NS, None, text(AISLE), threshold=-0.6)
    assert reasons(conflicts) == [("lf", "similar"), ("w", "similar")]
    assert [c.score for c in conflicts.candidates] == pytest.approx([-0.3526, -0.5477], abs=1e-3)
    searched = {item.key: item.score for item in store.search(NS, query=AISLE)}
    assert [c.score for c in conflicts.candidates] == pytest.approx([searched["lf"], searched["w"]])


def test_slot_values_are_bound_not_formatted(store_factory, mode) -> None:
    store = store_factory(mode, index=seat_index())
    values: list[Any] = ["it's", "back\\slash", "ünïcödé", 42, True]
    # What the same values would look like formatted or loosely compared
    decoys: list[Any] = ["it''s", "back\\\\slash", "unicode", "42", 1, "true"]
    for i, kind in enumerate(values + decoys):
        store.put(NS, f"m{i}", text(VEGETARIAN, kind=kind))

    for i, kind in enumerate(values):
        conflicts = store.find_conflicts(NS, None, text(VEGETARIAN, kind=kind), slot="kind")
        assert [c.key for c in conflicts.candidates if c.reason == "slot"] == [f"m{i}"]


def test_find_conflicts_embeds_once(store_factory, mode) -> None:
    index = seat_index()
    store = store_factory(mode, index=index)
    store.put(NS, "w", text(WINDOW))
    store.put(NS, "s", text(VEGETARIAN, seat="x"))
    index["embed"].calls = 0

    conflicts = store.find_conflicts(NS, "s", text(AISLE, seat="x"), slot="seat")
    assert index["embed"].calls == 1
    # The embedding is kept for the write
    assert conflicts._state is not None
    [row] = conflicts._state.vector_rows
    assert row[:3] == (PREFIX, "s", "text")
    assert row[3] == pytest.approx(SEATS[AISLE])


def test_find_conflicts_needs_setup(store_factory, admin, mode) -> None:
    index = seat_index()
    store = store_factory(mode, setup=False, index=index)
    for statement in schema_0_5(store):
        admin.execute(statement)

    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.find_conflicts(NS, "k", text(AISLE))
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.record(NS, "k")
    assert index["embed"].calls == 0

    store.setup()
    store.put(NS, "k", text(WINDOW))
    assert reasons(store.find_conflicts(NS, "k", text(AISLE))) == [("k", "key")]


class LaggingClock(datetime):
    """A client clock an hour behind the database's."""

    @classmethod
    def now(cls, tz: Any = None) -> Any:  # type: ignore[override]
        return datetime.now(tz) - timedelta(hours=1)

    @classmethod
    def utcnow(cls) -> Any:  # type: ignore[override]
        return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)


def test_default_valid_time_comes_from_the_database_clock(
    store_factory, admin, mode, monkeypatch
) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    target = store.get(NS, "k")
    assert target is not None
    for module in (store_base, store_module, store_reconcile):
        monkeypatch.setattr(module, "datetime", LaggingClock, raising=False)
    row = admin.execute("SELECT now() AS now").fetchone()
    assert row is not None

    conflicts = store.find_conflicts(NS, "k", text(AISLE))
    # From a client clock that lags the database's, a correction written just
    # after the fact it corrects would look older than that fact
    assert conflicts.proposal.valid_at >= target.updated_at
    assert conflicts.proposal.valid_at >= row["now"]
    # No valid time was given, so the write stores none
    assert conflicts._state is not None and conflicts._state.plan.valid_at is None
    monkeypatch.undo()

    # A node whose clock runs an hour ahead writes the target: the default
    # is raised to that write's updated_at
    admin.execute(
        f"UPDATE {store.sql.items} SET updated_at = now() + INTERVAL '1 hour' "
        "WHERE prefix = %s AND key = 'k'",
        (PREFIX,),
    )
    ahead = store.get(NS, "k")
    assert ahead is not None
    assert store.find_conflicts(NS, "k", text(AISLE)).proposal.valid_at == ahead.updated_at

    # A valid time the caller gives is kept as given, even an older one
    old = datetime(2025, 6, 1, tzinfo=timezone.utc)
    conflicts = store.find_conflicts(NS, "k", text(AISLE), valid_at=old)
    assert conflicts.proposal.valid_at == old
    assert conflicts._state is not None and conflicts._state.plan.valid_at == old


def test_record_and_find_conflicts_never_push_back_an_expiry(store_factory, admin, mode) -> None:
    store = store_factory(
        mode, index=seat_index(), ttl={"default_ttl": 10.0, "refresh_on_read": True}
    )
    store.put(NS, "k", text(WINDOW))
    store.put(NS, "s", text(VEGETARIAN, seat="x"))
    for table in (store.sql.items, store.sql.vectors):
        admin.execute(
            f"UPDATE {table} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
            (PREFIX,),
        )
    before = expiries(admin, store)

    assert store.record(NS, "k") is not None
    conflicts = store.find_conflicts(NS, "k", text(AISLE, seat="x"), slot="seat")
    assert reasons(conflicts) == [("k", "key"), ("s", "slot")]
    assert expiries(admin, store) == before
    # A get on this store does push it back, so the expiry could have moved
    store.get(NS, "k")
    assert expiries(admin, store) != before


def test_from_conn_string_takes_reconcile_scan_limit(libpq_url, table_name) -> None:
    with CockroachDBStore.from_conn_string(
        libpq_url, table_name=table_name, index=seat_index(), reconcile_scan_limit=1
    ) as store:
        store.setup()
        assert store.reconcile_scan_limit == 1
        store.put(NS, "w", text(WINDOW))
        store.put(NS, "lf", text(LONG_FLIGHTS))
        conflicts = store.find_conflicts(NS, None, text(AISLE))
        assert reasons(conflicts) == [("lf", "similar"), ("w", "similar")]


async def test_async_from_conn_string_takes_reconcile_scan_limit(libpq_url, table_name) -> None:
    async with AsyncCockroachDBStore.from_conn_string(
        libpq_url, table_name=table_name, reconcile_scan_limit=3
    ) as store:
        assert store.reconcile_scan_limit == 3


# reconcile

RETRY_LOGGER = "langchain_cockroachdb.retry"
# Of everything reconcile sends, only the key and slot candidate read contains this
CANDIDATE_READ = "coalesce(valid_at, updated_at) as valid_at"
OLD = datetime(2025, 1, 1, tzinfo=timezone.utc)
OLDER = datetime(2024, 1, 1, tzinfo=timezone.utc)


def retries(caplog: pytest.LogCaptureFixture, label: str) -> list[str]:
    """The retry warnings logged for ``label``."""
    prefix = f"Retrying {label} after "
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def history_rows(admin: Connection[Any], store: Any) -> list[dict[str, Any]]:
    """Every history row under PREFIX."""
    return admin.execute(
        f"SELECT key, version, reason, superseded_by, value, valid_at, invalid_at, write_id "
        f"FROM {store.sql.history} WHERE prefix = %s ORDER BY key, version",
        (PREFIX,),
    ).fetchall()


def vector_keys(admin: Connection[Any], store: Any) -> list[str]:
    rows = admin.execute(
        f"SELECT key FROM {store.sql.vectors} WHERE prefix = %s ORDER BY key", (PREFIX,)
    ).fetchall()
    return [row["key"] for row in rows]


def stored_valid_at(admin: Connection[Any], store: Any, key: str) -> Any:
    """The valid_at column of an item, None when the write stored no valid time."""
    row = admin.execute(
        f"SELECT valid_at FROM {store.sql.items} WHERE prefix = %s AND key = %s", (PREFIX, key)
    ).fetchone()
    assert row is not None
    return row["valid_at"]


def in_slot(value: str) -> dict[str, Any]:
    return text(value, seat="x")


def slot_policy() -> Policy:
    return supersede_same_slot()


def test_reconcile_supersedes_an_older_fact_in_the_same_slot(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    ns = ("user", "alice")
    first = store.reconcile(
        ns,
        None,
        {"seat": "x", "text": "Prefers window seats"},
        policy=supersede_same_slot(),
        slot="seat",
    )
    second = store.reconcile(
        ns,
        None,
        {"seat": "x", "text": "Prefers aisle seats"},
        policy=supersede_same_slot(),
        slot="seat",
    )
    assert second.new == "written" and second.superseded == (first.key,)
    assert [i.value["text"] for i in store.search(ns)] == ["Prefers aisle seats"]
    [old] = store.history(ns, first.key)
    assert old.reason == "superseded" and old.superseded_by == second.key

    assert first == ReconcileResult(
        key=first.key, new="written", version=1, superseded=(), deleted=(), confirmed=(), kept=()
    )
    assert second == ReconcileResult(
        key=second.key,
        new="written",
        version=1,
        superseded=(first.key,),
        deleted=(),
        confirmed=(),
        kept=(),
    )
    # No valid time was given: the new memory stores none, so it is valid from
    # its write, and the fact it superseded ended at that write
    new = store.record(ns, second.key)
    assert new is not None and new.valid_at == new.item.updated_at
    assert stored_valid_at(admin, store, second.key) is None
    assert old.invalid_at == new.item.updated_at
    assert (old.version, old.value) == (1, in_slot(WINDOW))
    # The superseded memory's vectors went with it
    assert vector_keys(admin, store) == [second.key]


def test_reconcile_with_a_valid_time_ends_older_facts_at_it(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    first = store.reconcile(
        NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat", valid_at=OLDER
    )
    second = store.reconcile(
        NS,
        None,
        in_slot(AISLE),
        policy=slot_policy(),
        slot="seat",
        valid_at=OLD,
        provenance={"source": "chat"},
    )
    assert second.superseded == (first.key,)
    [old] = store.history(NS, first.key)
    assert (old.valid_at, old.invalid_at) == (OLDER, OLD)
    # The valid time and the provenance are stored as given
    assert stored_valid_at(admin, store, second.key) == OLD
    record = store.record(NS, second.key)
    assert record is not None and (record.valid_at, record.provenance) == (OLD, {"source": "chat"})


def test_reconcile_skips_and_confirms_a_duplicate(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index(), ttl={"default_ttl": 10.0})
    window = store.reconcile(NS, None, text(WINDOW), policy=keep_all())
    for table in (store.sql.items, store.sql.vectors):
        admin.execute(
            f"UPDATE {table} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
            (PREFIX,),
        )
    before = store.record(NS, window.key)
    assert before is not None and before.confirmations == 0

    result = store.reconcile(NS, None, text(LONG_FLIGHTS), policy=skip_duplicates())

    assert result == ReconcileResult(
        key=result.key,
        new="skipped",
        version=None,
        superseded=(),
        deleted=(),
        confirmed=(window.key,),
        kept=(),
    )
    record = store.record(NS, window.key)
    assert record is not None and record.confirmed_at is not None
    assert (record.version, record.confirmations) == (1, 1)
    # Confirming changes neither the version nor updated_at, and keeps no history
    assert record.item.updated_at == before.item.updated_at
    assert store.history(NS, window.key) == []
    # Nothing was written for the duplicate
    assert store.record(NS, result.key) is None
    # The memory's expiry and its vector's were renewed from its own TTL
    for table in (store.sql.items, store.sql.vectors):
        row = admin.execute(
            f"SELECT count(*) AS n FROM {table} "
            "WHERE prefix = %s AND expires_at > now() + INTERVAL '5 minutes'",
            (PREFIX,),
        ).fetchone()
        assert row is not None and row["n"] == 1


def test_reconcile_records_a_late_fact_as_superseded(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    live = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")

    late = store.reconcile(
        NS,
        None,
        in_slot(AISLE),
        policy=slot_policy(),
        slot="seat",
        provenance={"source": "old chat"},
        valid_at=OLD,
    )

    assert late == ReconcileResult(
        key=late.key,
        new="superseded",
        version=None,
        superseded=(),
        deleted=(),
        confirmed=(),
        kept=(live.key,),
    )
    # The live fact is unchanged, and the late one never became live
    assert [item.key for item in store.search(NS)] == [live.key]
    assert store.record(NS, late.key) is None
    current = store.record(NS, live.key)
    assert current is not None and current.version == 1
    assert store.history(NS, live.key) == []
    [row] = store.history(NS, late.key)
    assert (row.version, row.reason, row.superseded_by) == (1, "superseded", live.key)
    assert (row.value, row.provenance) == (in_slot(AISLE), {"source": "old chat"})
    # True from its own valid time until the newer fact became true. The store
    # held it for no time
    assert (row.valid_at, row.invalid_at) == (OLD, current.valid_at)
    assert row.written_at == row.ended_at


def test_reconcile_without_history_skips_a_late_fact(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index())
    live = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")

    late = store.reconcile(
        NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat", valid_at=OLD
    )

    # With no history to record it in, the late fact is skipped, and the result says so
    assert (late.new, late.version, late.kept) == ("skipped", None, (live.key,))
    assert [item.key for item in store.search(NS)] == [live.key]
    assert store.record(NS, late.key) is None
    assert vector_keys(admin, store) == [live.key]


def test_reconcile_keep_all_reports_candidates(store_factory, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", text(PEANUTS))
    store.put(NS, "s", text(VEGETARIAN, seat="x"))
    store.put(NS, "w", text(WINDOW))
    store.put(NS, "other", text(VEGETARIAN, seat="y"))

    result = store.reconcile(NS, "k", in_slot(AISLE), policy=keep_all(), slot="seat")

    # The memory under the key is overwritten, so only the slot and similar
    # matches are kept
    assert result == ReconcileResult(
        key="k", new="written", version=2, superseded=(), deleted=(), confirmed=(), kept=("s", "w")
    )
    assert {item.key for item in store.search(NS)} == {"k", "s", "w", "other"}
    [old] = store.history(NS, "k")
    assert (old.version, old.reason, old.value) == (1, "overwritten", text(PEANUTS))
    assert store.history(NS, "s") == [] and store.history(NS, "w") == []


@pytest.mark.parametrize("failure", ["raises", "invalid decision"])
def test_a_policy_exception_rolls_back(store_factory, admin, caplog, mode, failure: str) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    first = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")
    before = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))
    seen: list[list[str]] = []
    raised = RuntimeError("the judge is unreachable")

    def policy(proposal: Any, candidates: Any) -> Decision:
        seen.append([c.key for c in candidates])
        if failure == "raises":
            raise raised
        return Decision(actions={"missing": Action.DELETE})

    expected = RuntimeError if failure == "raises" else ValueError
    with (
        caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
        pytest.raises(expected) as error,
    ):
        store.reconcile(NS, None, in_slot(AISLE), policy=policy, slot="seat")

    # The policy ran once, inside the transaction, and saw the slot fact. The
    # error came out unchanged, nothing was retried and nothing was written
    assert seen == [[first.key]]
    if failure == "raises":
        assert error.value is raised
    else:
        assert "unknown candidate key: missing" in str(error.value)
    assert not retries(caplog, "store reconcile")
    after = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))
    assert after == before
    # The transaction was rolled back, so the store's connection works on
    second = store.reconcile(NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat")
    assert second.superseded == (first.key,)


def test_reconcile_without_index_supersedes_by_slot(store_factory, mode) -> None:
    store = store_factory(mode, history=True)
    first = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")
    store.put(NS, "food", text(VEGETARIAN))

    # No threshold is needed, and one given is ignored
    second = store.reconcile(
        NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat", threshold=5.0
    )

    assert (second.new, second.superseded, second.kept) == ("written", (first.key,), ())
    assert {item.key for item in store.search(NS)} == {second.key, "food"}
    [old] = store.history(NS, first.key)
    assert (old.reason, old.superseded_by) == ("superseded", second.key)


def test_reconcile_on_an_l2_store_with_a_threshold(store_factory, mode) -> None:
    index = seat_index(distance_type="l2")
    store = store_factory(mode, history=True, index=index)
    window = store.reconcile(NS, None, text(WINDOW), policy=keep_all(), threshold=-0.6)
    store.put(NS, "food", text(VEGETARIAN))
    index["embed"].calls = 0

    with pytest.raises(ValueError, match="threshold"):
        store.reconcile(NS, None, text(AISLE), policy=supersede_similar(threshold=-0.6))
    assert index["embed"].calls == 0

    # Scores are negated l2 distances: the window seat scores -0.5477 against
    # the aisle seat, the food far less
    result = store.reconcile(
        NS, None, text(AISLE), policy=supersede_similar(threshold=-0.6), threshold=-0.6
    )
    assert (result.new, result.superseded, result.kept) == ("written", (window.key,), ())
    assert {item.key for item in store.search(NS)} == {result.key, "food"}


def run_together(*calls: Callable[[], Any]) -> list[Any]:
    """Run each call in its own thread, started together, and return their results.

    Raises the first error a call raised.
    """
    barrier = threading.Barrier(len(calls))
    results: list[Any] = [None] * len(calls)
    errors: list[BaseException] = []

    def run(i: int, call: Callable[[], Any]) -> None:
        try:
            barrier.wait(timeout=10)
            results[i] = call()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i, call)) for i, call in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)
    if errors:
        raise errors[0]
    return results


def before_first_decision(policy: Policy, wait: Callable[[], Any]) -> Policy:
    """``policy``, calling ``wait`` before it decides for the first time.

    The policy runs in the reconcile's transaction, after the candidate
    reads, so ``wait`` holds the transaction open between its reads and its
    writes. A retry decides without waiting.
    """
    lock = threading.Lock()
    calls = 0

    def decide(proposal: Any, candidates: Any) -> Decision:
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            wait()
        return policy(proposal, candidates)

    return decide


def test_two_reconciles_in_one_namespace_leave_one_live_fact(store_factory, admin, caplog) -> None:
    # Both read the slot before either writes: each finds it empty. Serializable
    # isolation then retries one of them, and its policy sees the other's fact
    store = store_factory("pool", history=True, index=seat_index())
    both_read = threading.Barrier(2, timeout=10)

    def propose(value: str) -> Callable[[], ReconcileResult]:
        policy = before_first_decision(slot_policy(), both_read.wait)
        return lambda: store.reconcile(NS, None, in_slot(value), policy=policy, slot="seat")

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        results = run_together(propose(WINDOW), propose(AISLE))

    assert retries(caplog, "store reconcile")
    assert [result.new for result in results] == ["written", "written"]
    live = admin.execute(
        f"SELECT key FROM {store.sql.items} WHERE prefix = %s AND value ->> 'seat' = 'x'",
        (PREFIX,),
    ).fetchall()
    assert len(live) == 1
    winner = next(result for result in results if result.key == live[0]["key"])
    loser = next(result for result in results if result is not winner)
    assert winner.superseded == (loser.key,)
    [old] = store.history(NS, loser.key)
    assert (old.reason, old.superseded_by) == ("superseded", winner.key)
    assert store.history(NS, winner.key) == []


def test_a_put_racing_a_reconcile_is_serialized(store_factory, admin, caplog) -> None:
    # The put commits between the reconcile's candidate read and its writes
    store = store_factory("pool", history=True, index=seat_index())
    store.put(NS, "slotted", in_slot(WINDOW))
    read, put_done = threading.Event(), threading.Event()

    def put() -> None:
        assert read.wait(10)
        store.put(NS, "slotted", in_slot(LONG_FLIGHTS))
        put_done.set()

    def wait_for_the_put() -> None:
        read.set()
        assert put_done.wait(10)

    policy = before_first_decision(slot_policy(), wait_for_the_put)

    def reconcile() -> ReconcileResult:
        return store.reconcile(NS, "new", in_slot(AISLE), policy=policy, slot="seat")

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        _, result = run_together(put, reconcile)

    # The reconcile read the first version of the memory in the slot, which
    # the put replaced before the reconcile wrote. It ran again, after the put
    assert retries(caplog, "store reconcile")
    assert result.superseded == ("slotted",)
    rows = admin.execute(
        f"SELECT key, value FROM {store.sql.items} WHERE prefix = %s", (PREFIX,)
    ).fetchall()
    live = {row["key"]: row["value"]["text"] for row in rows}
    archived = [
        (row["key"], row["version"], row["reason"], row["value"]["text"])
        for row in history_rows(admin, store)
    ]
    # The state of one of the two serial orders, with no version archived twice
    put_first = (
        {"new": AISLE},
        [("slotted", 1, "overwritten", WINDOW), ("slotted", 2, "superseded", LONG_FLIGHTS)],
    )
    reconcile_first = (
        {"new": AISLE, "slotted": LONG_FLIGHTS},
        [("slotted", 1, "superseded", WINDOW)],
    )
    assert (live, archived) in (put_first, reconcile_first)
    # Here the put came first
    assert (live, archived) == put_first


def test_the_policy_decides_in_the_write_transaction(store_factory, mode) -> None:
    # The default valid time is the clock of the transaction that read the
    # candidates. It equals the written memory's updated_at only when that
    # transaction is the one that wrote it
    store = store_factory(mode, history=True, index=seat_index())
    seen: list[datetime] = []

    def policy(proposal: Any, candidates: Any) -> Decision:
        seen.append(proposal.valid_at)
        return Decision()

    result = store.reconcile(NS, None, text(WINDOW), policy=policy)

    # The last decision is the one the committed attempt made
    record = store.record(NS, result.key)
    assert record is not None and seen[-1] == record.item.updated_at


def test_reconcile_retries_a_40001(store_factory, unsafe_internals, caplog, mode) -> None:
    faults = PsycopgFaults(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": faults.cursor_class()},
    )
    first = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")
    faults.fail_next("40001", match=CANDIDATE_READ)

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        second = store.reconcile(NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat")

    assert faults.injected == ["40001"]
    found = retries(caplog, "store reconcile")
    assert len(found) == 1 and "SQLSTATE 40001" in found[0]
    assert sum("Retrying" in r.getMessage() for r in caplog.records) == 1
    assert second == ReconcileResult(
        key=second.key,
        new="written",
        version=1,
        superseded=(first.key,),
        deleted=(),
        confirmed=(),
        kept=(),
    )
    assert len(store.history(NS, first.key)) == 1


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_reconcile_retried_after_a_lost_commit_is_already_applied(
    store_factory, admin, unsafe_internals, caplog, mode
) -> None:
    # The reconcile commits, then the store sees 40003 and runs it again. The
    # retry finds the rows that carry its write id and writes nothing
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    first = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")
    clean = store.reconcile(
        ("user", "bob"), None, in_slot(WINDOW), policy=slot_policy(), slot="seat"
    )
    clean_second = store.reconcile(
        ("user", "bob"), None, in_slot(AISLE), policy=slot_policy(), slot="seat"
    )
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        second = store.reconcile(NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat")

    assert fault.fired == 1
    found = retries(caplog, "store reconcile")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    # Rebuilt from the rows: the same superseded keys as a clean run reports
    assert clean_second.superseded == (clean.key,) and not clean_second.already_applied
    assert second == ReconcileResult(
        key=second.key,
        new="written",
        version=1,
        superseded=(first.key,),
        deleted=(),
        confirmed=(),
        kept=(),
        already_applied=True,
    )
    # One history row per superseded key, and one write of the new memory
    assert [(row["key"], row["reason"]) for row in history_rows(admin, store)] == [
        (first.key, "superseded")
    ]
    assert [item.key for item in store.search(NS)] == [second.key]
    assert vector_keys(admin, store) == [second.key]


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_a_confirmation_retried_after_a_lost_commit_is_already_applied(
    store_factory, unsafe_internals, mode
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.reconcile(NS, "w", text(WINDOW), policy=keep_all())
    fault.armed = True

    # The same fact again under its own key: a duplicate of the live memory, confirmed
    again = store.reconcile(NS, "w", text(WINDOW), policy=skip_duplicates())

    assert fault.fired == 1
    # Its confirmation is later than its update, so the rows prove a confirmation
    assert again == ReconcileResult(
        key="w",
        new="skipped",
        version=None,
        superseded=(),
        deleted=(),
        confirmed=("w",),
        kept=(),
        already_applied=True,
    )
    record = store.record(NS, "w")
    assert record is not None and (record.version, record.confirmations) == (1, 1)


def test_reconcile_embeds_once_across_retries(store_factory, admin, unsafe_internals, mode) -> None:
    faults = PsycopgFaults(unsafe_internals)
    index = seat_index()
    store = store_factory(
        mode, history=True, index=index, connect_kwargs={"cursor_factory": faults.cursor_class()}
    )
    store.put(NS, "w", in_slot(WINDOW))
    index["embed"].calls = 0
    faults.fail_next("40001", match=CANDIDATE_READ, times=2)

    result = store.reconcile(NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat")

    assert faults.injected == ["40001", "40001"]
    assert index["embed"].calls == 1
    assert result.superseded == ("w",)
    # The third attempt wrote the vector embedded before the first
    row = admin.execute(
        f"SELECT embedding FROM {store.sql.vectors} WHERE prefix = %s AND key = %s",
        (PREFIX, result.key),
    ).fetchone()
    assert row is not None
    assert parse_vector(row["embedding"]) == pytest.approx(SEATS[AISLE], abs=1e-6)


def test_a_late_fact_under_a_live_key_keeps_every_version(store_factory, mode) -> None:
    # The late fact goes to history under a key whose live memory stays. Its
    # version follows the live one, and the next writes number after it, so
    # archiving them later never collides with it
    store = store_factory(mode, history=True, index=seat_index())
    store.reconcile(NS, "k", text(VEGETARIAN), policy=keep_all(), valid_at=OLDER)
    newer = store.reconcile(NS, None, in_slot(WINDOW), policy=keep_all(), slot="seat")

    late = store.reconcile(NS, "k", in_slot(AISLE), policy=slot_policy(), slot="seat", valid_at=OLD)
    assert (late.new, late.kept) == ("superseded", ("k", newer.key))
    store.put(NS, "k", text(PEANUTS))
    store.put(NS, "k", text(LONG_FLIGHTS))

    versions = [(v.version, v.reason, v.value) for v in store.history(NS, "k")]
    assert versions == [
        (3, "overwritten", text(PEANUTS)),
        (2, "superseded", in_slot(AISLE)),
        (1, "overwritten", text(VEGETARIAN)),
    ]
    record = store.record(NS, "k")
    assert record is not None and (record.version, record.item.value) == (4, text(LONG_FLIGHTS))


def test_a_late_fact_under_a_key_with_history_does_not_collide(store_factory, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", text(VEGETARIAN))
    store.delete(NS, "k")
    newer = store.reconcile(NS, None, in_slot(WINDOW), policy=keep_all(), slot="seat")

    late = store.reconcile(NS, "k", in_slot(AISLE), policy=slot_policy(), slot="seat", valid_at=OLD)

    assert late.new == "superseded"
    versions = [(v.version, v.reason, v.superseded_by) for v in store.history(NS, "k")]
    assert versions == [(2, "superseded", newer.key), (1, "deleted", None)]
    # A write after it continues the versions, and its result reads the version back
    written = store.reconcile(NS, "k", text(PEANUTS), policy=keep_all())
    assert written.version == 3
    record = store.record(NS, "k")
    assert record is not None and record.version == 3


def test_an_archived_version_never_ends_before_it_began(store_factory, mode) -> None:
    # keep_all ignores valid times, so it can write a fact older than the one
    # it overwrites. The overwritten version then ends where it began
    store = store_factory(mode, history=True, index=seat_index())
    store.reconcile(NS, "k", text(WINDOW), policy=keep_all(), valid_at=OLD)

    result = store.reconcile(NS, "k", text(AISLE), policy=keep_all(), valid_at=OLDER)

    assert (result.new, result.version) == ("written", 2)
    [old] = store.history(NS, "k")
    assert (old.reason, old.valid_at, old.invalid_at) == ("overwritten", OLD, OLD)
    # A late fact recorded straight into history is held to the same rule: a
    # policy may name an older candidate as the one that supersedes it
    store.reconcile(
        NS, "older", in_slot(VEGETARIAN), policy=keep_all(), slot="seat", valid_at=OLDER
    )

    def superseded_by_older(proposal: Any, candidates: Any) -> Decision:
        return Decision(new="superseded", superseded_by="older")

    late = store.reconcile(
        NS, "late", in_slot(PEANUTS), policy=superseded_by_older, slot="seat", valid_at=OLD
    )
    assert late.new == "superseded"
    [row] = store.history(NS, "late")
    assert (row.valid_at, row.invalid_at) == (OLD, OLD)


def deciding(decision: Decision) -> Policy:
    """A policy that returns ``decision``, whatever the candidates."""
    return lambda proposal, candidates: decision


def test_reconcile_deletes_a_candidate_into_history(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "gone", in_slot(WINDOW))
    store.put(NS, "food", text(VEGETARIAN))

    result = store.reconcile(
        NS,
        None,
        in_slot(AISLE),
        policy=deciding(Decision(actions={"gone": Action.DELETE})),
        slot="seat",
        valid_at=OLD,
    )

    assert (result.new, result.deleted, result.superseded) == ("written", ("gone",), ())
    assert store.get(NS, "gone") is None
    assert vector_keys(admin, store) == sorted(["food", result.key])
    [old] = store.history(NS, "gone")
    written = store.record(NS, result.key)
    assert written is not None
    assert (old.reason, old.superseded_by, old.value) == ("deleted", None, in_slot(WINDOW))
    # A deleted fact ends when the reconcile writes, not at the new memory's valid time
    assert old.ended_at == old.invalid_at == written.item.updated_at
    assert written.valid_at == OLD


@pytest.mark.parametrize("action", [Action.DELETE, Action.SUPERSEDE])
def test_reconcile_without_history_removes_a_candidate_with_its_vectors(
    store_factory, admin, mode, action: Action
) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "gone", in_slot(WINDOW))
    store.put(NS, "food", text(VEGETARIAN))

    result = store.reconcile(
        NS, None, in_slot(AISLE), policy=deciding(Decision(actions={"gone": action})), slot="seat"
    )

    reported = result.deleted if action == Action.DELETE else result.superseded
    assert reported == ("gone",)
    assert store.get(NS, "gone") is None
    assert {item.key for item in store.search(NS)} == {"food", result.key}
    assert vector_keys(admin, store) == sorted(["food", result.key])
    # Without history nothing was kept anywhere: the store has no history table
    row = admin.execute("SELECT to_regclass(%s) IS NULL AS absent", (store.sql.history,)).fetchone()
    assert row is not None and row["absent"]


def test_a_late_fact_can_delete_the_memory_under_its_own_key(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.reconcile(NS, "k", in_slot(VEGETARIAN), policy=keep_all(), slot="seat", valid_at=OLDER)
    store.reconcile(NS, "newer", in_slot(WINDOW), policy=keep_all(), slot="seat")
    decision = Decision(new="superseded", superseded_by="newer", actions={"k": Action.DELETE})

    late = store.reconcile(
        NS, "k", in_slot(AISLE), policy=deciding(decision), slot="seat", valid_at=OLD
    )

    assert late == ReconcileResult(
        key="k",
        new="superseded",
        version=None,
        superseded=(),
        deleted=("k",),
        confirmed=(),
        kept=("newer",),
    )
    assert store.get(NS, "k") is None
    assert vector_keys(admin, store) == ["newer"]
    # The deleted version and the late fact both stay in history, numbered in turn
    versions = [(v.version, v.reason, v.superseded_by, v.value) for v in store.history(NS, "k")]
    assert versions == [
        (2, "superseded", "newer", in_slot(AISLE)),
        (1, "deleted", None, in_slot(VEGETARIAN)),
    ]


def test_reconcile_confirms_on_a_store_without_an_index(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, ttl={"default_ttl": 10.0})
    store.reconcile(NS, "w", in_slot(WINDOW), policy=keep_all(), slot="seat")
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
        (PREFIX,),
    )
    before = store.record(NS, "w")
    assert before is not None

    # Without an index there are no scores: the slot match with an equal value
    # is the duplicate
    again = store.reconcile(NS, None, in_slot(WINDOW), policy=skip_duplicates(), slot="seat")

    assert (again.new, again.version, again.confirmed) == ("skipped", None, ("w",))
    record = store.record(NS, "w")
    assert record is not None and record.confirmed_at is not None
    assert (record.version, record.confirmations) == (1, 1)
    assert record.item.updated_at == before.item.updated_at
    # Its expiry was renewed from its own TTL
    row = admin.execute(
        f"SELECT expires_at > now() + INTERVAL '5 minutes' AS renewed FROM {store.sql.items} "
        "WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()
    assert row is not None and row["renewed"]


def test_a_late_fact_retried_after_a_lost_commit_is_already_applied(
    store_factory, admin, unsafe_internals, mode
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    live = store.reconcile(NS, None, in_slot(WINDOW), policy=slot_policy(), slot="seat")
    fault.armed = True

    late = store.reconcile(
        NS, None, in_slot(AISLE), policy=slot_policy(), slot="seat", valid_at=OLD
    )

    assert fault.fired == 1
    # The history row under its own key proves the late fact
    assert late == ReconcileResult(
        key=late.key,
        new="superseded",
        version=None,
        superseded=(),
        deleted=(),
        confirmed=(),
        kept=(),
        already_applied=True,
    )
    assert [
        (row["key"], row["reason"], row["superseded_by"]) for row in history_rows(admin, store)
    ] == [(late.key, "superseded", live.key)]
    assert [item.key for item in store.search(NS)] == [live.key]


def test_a_delete_retried_after_a_lost_commit_is_already_applied(
    store_factory, admin, unsafe_internals, mode
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "gone", in_slot(WINDOW))
    fault.armed = True

    result = store.reconcile(
        NS,
        None,
        in_slot(AISLE),
        policy=deciding(Decision(new="skip", actions={"gone": Action.DELETE})),
        slot="seat",
    )

    assert fault.fired == 1
    assert result == ReconcileResult(
        key=result.key,
        new="skipped",
        version=None,
        superseded=(),
        deleted=("gone",),
        confirmed=(),
        kept=(),
        already_applied=True,
    )
    assert [(row["key"], row["reason"]) for row in history_rows(admin, store)] == [
        ("gone", "deleted")
    ]
    assert store.get(NS, "gone") is None and vector_keys(admin, store) == []


def raised_within(call: Callable[[], Any], timeout: float = 60.0) -> BaseException | None:
    """What ``call`` raised in another thread, or None if it returned.

    A store call that waits for a connection the store itself holds never
    returns: the thread stays blocked, and the test fails instead of hanging.
    """
    raised: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as error:  # noqa: BLE001
            raised.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"the call still runs after {timeout} s"
    return raised[0] if raised else None


def test_a_policy_that_calls_the_store_fails_at_once(store_factory, admin, mode) -> None:
    # On a single connection the policy runs while the store holds it, so a
    # store call from the policy would wait forever. On a pool it would read
    # outside the reconcile's transaction. Either way it fails at once
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "w", in_slot(WINDOW))
    before = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))

    def policy(proposal: Any, candidates: Any) -> Decision:
        store.get(NS, "w")
        return Decision()

    error = raised_within(
        lambda: store.reconcile(NS, None, in_slot(AISLE), policy=policy, slot="seat")
    )

    assert isinstance(error, RuntimeError), error
    assert str(error).startswith("A reconcile policy or a confirm TTL function must not call the store")
    after = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))
    assert after == before
    # The transaction was rolled back and the connection given back
    item = store.get(NS, "w")
    assert item is not None and item.value == in_slot(WINDOW)


# resolve


def test_resolve_applies_a_decision(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", text(PEANUTS))
    store.put(NS, "old", in_slot(WINDOW))
    store.put(NS, "gone", in_slot(VEGETARIAN))
    store.put(NS, "dup", text(LONG_FLIGHTS))
    store.put(NS, "food", text(VEGETARIAN))
    conflicts = store.find_conflicts(
        NS, "k", in_slot(AISLE), slot="seat", provenance={"source": "chat"}
    )
    assert reasons(conflicts) == [
        ("k", "key"),
        ("old", "slot"),
        ("gone", "slot"),
        ("dup", "similar"),
    ]
    decision = Decision(
        actions={"old": Action.SUPERSEDE, "gone": Action.DELETE, "dup": Action.CONFIRM}
    )

    result = store.resolve(conflicts, decision)

    assert result == ReconcileResult(
        key="k",
        new="written",
        version=2,
        superseded=("old",),
        deleted=("gone",),
        confirmed=("dup",),
        kept=(),
    )
    # The new memory replaced the target, which went to history, with the
    # provenance and the write id that find_conflicts planned
    record = store.record(NS, "k")
    assert record is not None
    assert (record.version, record.item.value, record.provenance) == (
        2,
        in_slot(AISLE),
        {"source": "chat"},
    )
    assert conflicts._state is not None
    row = admin.execute(
        f"SELECT write_id FROM {store.sql.items} WHERE prefix = %s AND key = 'k'", (PREFIX,)
    ).fetchone()
    assert row is not None and str(row["write_id"]) == conflicts._state.plan.write_id
    assert [(v.version, v.reason, v.value) for v in store.history(NS, "k")] == [
        (1, "overwritten", text(PEANUTS))
    ]
    [old] = store.history(NS, "old")
    assert (old.reason, old.superseded_by, old.value) == ("superseded", "k", in_slot(WINDOW))
    [gone] = store.history(NS, "gone")
    assert (gone.reason, gone.superseded_by) == ("deleted", None)
    assert store.get(NS, "old") is None and store.get(NS, "gone") is None
    # Confirming changed neither the version nor updated_at
    dup = store.record(NS, "dup")
    assert dup is not None and (dup.version, dup.confirmations) == (1, 1)
    assert dup.item.updated_at == conflicts.candidates[3].item.updated_at
    assert vector_keys(admin, store) == ["dup", "food", "k"]


def test_resolve_raises_conflicts_changed_after_a_concurrent_write(
    store_factory, admin, caplog, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "old", in_slot(WINDOW))
    store.put(NS, "dup", text(LONG_FLIGHTS))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    decision = Decision(actions={"old": Action.SUPERSEDE, "dup": Action.CONFIRM})
    # Another writer replaces the memory the decision supersedes
    store.put(NS, "old", in_slot(LONG_FLIGHTS))
    before = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))

    with (
        caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
        pytest.raises(ConflictsChanged) as error,
    ):
        store.resolve(conflicts, decision)

    assert error.value.keys == ("old",)
    # Nothing was written, not even the confirmation, and nothing was retried
    after = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))
    assert after == before
    dup = store.record(NS, "dup")
    assert dup is not None and dup.confirmations == 0
    assert not retries(caplog, "store resolve")
    # Found again, the same decision applies
    again = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    result = store.resolve(again, decision)
    assert (result.new, result.superseded, result.confirmed) == ("written", ("old",), ("dup",))


def test_resolve_detects_a_0_5_style_write_through_updated_at(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "old", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    # A 0.5.x store writes with the 0.5 statement during a rolling upgrade: it
    # sets updated_at and leaves the version as it was
    value = json.dumps(in_slot(LONG_FLIGHTS))
    admin.execute(*store.sql.upsert_items([(PREFIX, "old", value, None)]))
    record = store.record(NS, "old")
    assert record is not None and (record.version, record.item.value) == (1, in_slot(LONG_FLIGHTS))
    before = (item_rows(admin, store), vector_keys(admin, store))

    with pytest.raises(ConflictsChanged) as error:
        store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))

    assert error.value.keys == ("old",)
    assert (item_rows(admin, store), vector_keys(admin, store)) == before


def test_resolve_ignores_changes_to_candidates_it_keeps(store_factory, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "old", in_slot(WINDOW))
    store.put(NS, "also", in_slot(VEGETARIAN))
    store.put(NS, "near", text(LONG_FLIGHTS))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    assert reasons(conflicts) == [("old", "slot"), ("also", "slot"), ("near", "similar")]
    # Another writer changes one candidate the decision keeps and deletes the other
    store.put(NS, "near", text(WINDOW))
    store.delete(NS, "also")

    result = store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))

    assert (result.new, result.superseded, result.kept) == ("written", ("old",), ("also", "near"))
    assert {item.key for item in store.search(NS)} == {result.key, "near"}


def test_resolve_rejects_an_invalid_decision_before_writing(
    store_factory, admin, mode, monkeypatch
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", text(PEANUTS))
    store.put(NS, "old", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, "k", in_slot(AISLE), slot="seat")
    before = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))

    def no_connection(**kwargs: Any) -> Any:
        raise AssertionError("resolve used the connection before rejecting the decision")

    monkeypatch.setattr(store, "_cursor", no_connection)
    for bad in (
        Decision(actions={"missing": Action.DELETE}),
        Decision(actions={"k": Action.SUPERSEDE}),
        Decision(new="skip", actions={"old": Action.SUPERSEDE}),
        Decision(new="superseded", superseded_by="k"),
        {"old": "supersede"},
    ):
        with pytest.raises(ValueError):
            store.resolve(conflicts, bad)  # type: ignore[arg-type]
    monkeypatch.undo()

    after = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))
    assert after == before
    # A valid decision about the same Conflicts still applies
    result = store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))
    assert (result.new, result.version, result.superseded) == ("written", 2, ("old",))


def test_resolve_embeds_nothing(store_factory, admin, mode) -> None:
    index = seat_index()
    store = store_factory(mode, index=index)
    store.put(NS, "old", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    calls = index["embed"].calls

    result = store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))

    assert index["embed"].calls == calls
    # The new memory's vector is the one find_conflicts embedded
    row = admin.execute(
        f"SELECT embedding FROM {store.sql.vectors} WHERE prefix = %s AND key = %s",
        (PREFIX, result.key),
    ).fetchone()
    assert row is not None
    assert parse_vector(row["embedding"]) == pytest.approx(SEATS[AISLE], abs=1e-6)
    assert vector_keys(admin, store) == [result.key]


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_resolve_retried_after_a_lost_commit_is_already_applied(
    store_factory, admin, unsafe_internals, caplog, mode
) -> None:
    # The resolve commits, then the store sees 40003 and runs it again. The
    # retry finds the rows that carry its write id, not a conflict
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "old", in_slot(WINDOW))
    store.put(NS, "dup", text(LONG_FLIGHTS))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = store.resolve(
            conflicts, Decision(actions={"old": Action.SUPERSEDE, "dup": Action.CONFIRM})
        )

    assert fault.fired == 1
    found = retries(caplog, "store resolve")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    assert result == ReconcileResult(
        key=conflicts.proposal.key,
        new="written",
        version=1,
        superseded=("old",),
        deleted=(),
        confirmed=("dup",),
        kept=(),
        already_applied=True,
    )
    # One write: one history row, the new memory once, one confirmation
    assert [(row["key"], row["reason"]) for row in history_rows(admin, store)] == [
        ("old", "superseded")
    ]
    assert {item.key for item in store.search(NS)} == {result.key, "dup"}
    dup = store.record(NS, "dup")
    assert dup is not None and dup.confirmations == 1


def test_resolving_the_same_conflicts_twice_returns_the_first_result(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "old", in_slot(WINDOW))
    store.put(NS, "dup", text(LONG_FLIGHTS))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    first = store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))
    written = (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store))

    # Called again, as after an error that left the outcome unknown, resolve
    # finds its own write rather than a conflict. Another decision about the
    # same Conflicts changes nothing either
    for decision in (
        Decision(actions={"old": Action.SUPERSEDE}),
        Decision(new="skip", actions={"dup": Action.DELETE}),
    ):
        again = store.resolve(conflicts, decision)
        assert again == dataclasses.replace(first, kept=(), already_applied=True)

    assert first.kept == ("dup",)
    assert (item_rows(admin, store), history_rows(admin, store), vector_keys(admin, store)) == (
        written
    )
    assert store.get(NS, "dup") is not None


def test_resolve_counts_a_memory_that_appeared_or_expired_under_the_key_as_changed(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, index=seat_index())

    def move_expiry(key: str, by: str) -> None:
        # Moves only the expiry: the version and updated_at stay as they are
        admin.execute(
            f"UPDATE {store.sql.items} SET expires_at = now() + CAST(%s AS INTERVAL) "
            "WHERE prefix = %s AND key = %s",
            (by, PREFIX, key),
        )

    # No memory had the key when find_conflicts ran, and another writer puts one
    fresh = store.find_conflicts(NS, "new", text(AISLE))
    store.put(NS, "new", text(WINDOW))
    # The memory under the key had expired, and comes back to life
    store.put(NS, "back", text(WINDOW), ttl=60)
    move_expiry("back", "-1 minute")
    revived = store.find_conflicts(NS, "back", text(AISLE))
    assert revived.target_version is None
    move_expiry("back", "1 hour")
    # The memory under the key was live, and expires before resolve
    store.put(NS, "lapsed", text(WINDOW), ttl=60)
    lapsing = store.find_conflicts(NS, "lapsed", text(AISLE))
    assert lapsing.target_version == 1
    move_expiry("lapsed", "-1 minute")
    before = item_rows(admin, store)

    for conflicts, key in ((fresh, "new"), (revived, "back"), (lapsing, "lapsed")):
        with pytest.raises(ConflictsChanged) as error:
            store.resolve(conflicts, Decision())
        assert error.value.keys == (key,)
    assert item_rows(admin, store) == before


def test_resolve_ends_old_facts_at_its_own_write_without_a_valid_time(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "old", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    assert conflicts._state is not None and conflicts._state.plan.valid_at is None

    result = store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))

    # The new memory stores no valid time, so it is valid from its write: the
    # resolve's, not the read find_conflicts made. The fact it superseded
    # ended at that write
    new = store.record(NS, result.key)
    assert new is not None
    assert stored_valid_at(admin, store, result.key) is None
    assert new.valid_at == new.item.updated_at > conflicts.proposal.valid_at
    [old] = store.history(NS, "old")
    assert old.invalid_at == new.item.updated_at


def test_resolve_uses_a_given_valid_time_as_reconcile_does(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.reconcile(NS, "older", in_slot(WINDOW), policy=keep_all(), slot="seat", valid_at=OLDER)
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat", valid_at=OLD)

    result = store.resolve(conflicts, Decision(actions={"older": Action.SUPERSEDE}))

    # The new memory stores its valid time, and the fact it superseded ended at it
    assert stored_valid_at(admin, store, result.key) == OLD
    [old] = store.history(NS, "older")
    assert (old.valid_at, old.invalid_at) == (OLDER, OLD)
    # A late fact goes straight into history, true from its own valid time
    # until the newer fact became true
    late = store.find_conflicts(NS, None, in_slot(PEANUTS), slot="seat", valid_at=OLDER)
    recorded = store.resolve(late, Decision(new="superseded", superseded_by=result.key))
    assert (recorded.new, recorded.version, recorded.kept) == ("superseded", None, (result.key,))
    assert store.record(NS, recorded.key) is None
    [row] = store.history(NS, recorded.key)
    assert (row.reason, row.superseded_by) == ("superseded", result.key)
    assert (row.valid_at, row.invalid_at) == (OLDER, OLD)


def test_resolve_accepts_a_conflicts_and_a_decision_from_a_checkpoint(
    store_factory, admin, mode
) -> None:
    # A LangGraph interrupt keeps both in a checkpoint while a person decides,
    # and the process that resumes the graph has a store of its own
    serde = JsonPlusSerializer()
    finder = store_factory(mode, history=True, index=seat_index())
    finder.put(NS, "k", text(PEANUTS))
    finder.put(NS, "old", in_slot(WINDOW))
    finder.put(NS, "dup", text(LONG_FLIGHTS))
    conflicts = finder.find_conflicts(
        NS, "k", in_slot(AISLE), slot="seat", provenance={"source": "chat"}
    )
    decision = Decision(actions={"old": Action.SUPERSEDE, "dup": Action.CONFIRM})
    saved = serde.loads_typed(serde.dumps_typed(conflicts))
    decided = serde.loads_typed(serde.dumps_typed(decision))
    assert (saved, decided) == (conflicts, decision)
    resumed = store_factory(mode, history=True, index=seat_index())

    result = resumed.resolve(saved, decided)

    assert result == ReconcileResult(
        key="k",
        new="written",
        version=2,
        superseded=("old",),
        deleted=(),
        confirmed=("dup",),
        kept=(),
    )
    assert serde.loads_typed(serde.dumps_typed(result)) == result
    record = resumed.record(NS, "k")
    assert record is not None
    assert (record.item.value, record.provenance) == (in_slot(AISLE), {"source": "chat"})
    row = admin.execute(
        f"SELECT embedding FROM {resumed.sql.vectors} WHERE prefix = %s AND key = 'k'",
        (PREFIX,),
    ).fetchone()
    assert row is not None
    assert parse_vector(row["embedding"]) == pytest.approx(SEATS[AISLE], abs=1e-6)


def test_a_write_racing_a_resolve_is_seen_on_its_retry(
    store_factory, admin, caplog, mode, monkeypatch
) -> None:
    # The check runs in the resolve's transaction. A write that commits after
    # the check and before the resolve's own writes makes serializable
    # isolation run the resolve again, and the check then sees it
    store = store_factory(mode, history=True, index=seat_index())
    other = store_factory("pool", history=True, index=seat_index())
    store.put(NS, "old", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    check = store._check_resolve
    checks: list[int] = []

    def check_then_race(resolving: Any, found: Any, **kwargs: Any) -> Any:
        checks.append(len(checks))
        result = check(resolving, found, **kwargs)
        if len(checks) == 1:
            # Another store replaces the memory the decision supersedes
            other.put(NS, "old", in_slot(LONG_FLIGHTS))
        return result

    monkeypatch.setattr(store, "_check_resolve", check_then_race)
    with (
        caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
        pytest.raises(ConflictsChanged) as error,
    ):
        store.resolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))

    assert retries(caplog, "store resolve") and len(checks) == 2
    assert error.value.keys == ("old",)
    # The racing write stands, and nothing of the resolve was written
    record = store.record(NS, "old")
    assert record is not None and (record.version, record.item.value) == (2, in_slot(LONG_FLIGHTS))
    assert [(row["key"], row["version"], row["reason"]) for row in history_rows(admin, store)] == [
        ("old", 1, "overwritten")
    ]
    assert store.record(NS, conflicts.proposal.key) is None


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_a_resolve_that_only_deletes_finishes_after_a_lost_commit_without_history(
    store_factory, admin, unsafe_internals, caplog, mode
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode, index=seat_index(), connect_kwargs={"cursor_factory": fault.cursor_class()}
    )
    store.put(NS, "taken", in_slot(PEANUTS))
    store.put(NS, "gone", in_slot(WINDOW))
    store.put(NS, "also", in_slot(VEGETARIAN))
    store.put(NS, "kept", text(LONG_FLIGHTS))
    # The first attempt is strict: a memory another writer deleted after
    # find_conflicts is a change, even one the decision deletes
    early = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    store.delete(NS, "taken")
    with pytest.raises(ConflictsChanged) as error:
        store.resolve(early, Decision(new="skip", actions={"taken": Action.DELETE}))
    assert error.value.keys == ("taken",)

    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    assert reasons(conflicts) == [("gone", "slot"), ("also", "slot"), ("kept", "similar")]
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = store.resolve(
            conflicts, Decision(new="skip", actions={"gone": Action.DELETE, "also": Action.DELETE})
        )

    # Without history a decision that only deletes leaves no row with its
    # write id. The retry finds the memories gone, as the decision asked,
    # deletes them again, which changes nothing, and returns normally
    assert fault.fired == 1
    found = retries(caplog, "store resolve")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    # No row proves the earlier commit, so the result is not marked applied
    assert result == ReconcileResult(
        key=conflicts.proposal.key,
        new="skipped",
        version=None,
        superseded=(),
        deleted=("gone", "also"),
        confirmed=(),
        kept=("kept",),
    )
    assert store.get(NS, "gone") is None and store.get(NS, "also") is None
    assert vector_keys(admin, store) == ["kept"]


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_a_resolve_that_deletes_its_target_finishes_after_a_lost_commit_without_history(
    store_factory, admin, unsafe_internals, caplog, mode
) -> None:
    # The decision skips the new memory and deletes the one under its key
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode, index=seat_index(), connect_kwargs={"cursor_factory": fault.cursor_class()}
    )
    store.put(NS, "k", in_slot(WINDOW))
    store.put(NS, "kept", text(LONG_FLIGHTS))
    conflicts = store.find_conflicts(NS, "k", in_slot(AISLE), slot="seat")
    assert reasons(conflicts) == [("k", "key"), ("kept", "similar")]
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = store.resolve(conflicts, Decision(new="skip", actions={"k": Action.DELETE}))

    # The retry finds the memory under the key gone, as the decision asked
    assert fault.fired == 1
    found = retries(caplog, "store resolve")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    assert result == ReconcileResult(
        key="k",
        new="skipped",
        version=None,
        superseded=(),
        deleted=("k",),
        confirmed=(),
        kept=("kept",),
    )
    assert store.get(NS, "k") is None
    assert vector_keys(admin, store) == ["kept"]


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_resolve_called_again_after_an_error_escaped_checks_strictly(
    store_factory, unsafe_internals, mode
) -> None:
    # With one attempt, the error of a lost commit escapes the call. Calling
    # resolve again starts a new call, whose first attempt is strict
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        index=seat_index(),
        retry_policy=RetryPolicy(max_attempts=1),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "gone", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    decision = Decision(new="skip", actions={"gone": Action.DELETE})
    fault.armed = True
    with pytest.raises(psycopg.errors.StatementCompletionUnknown):
        store.resolve(conflicts, decision)

    # Without history nothing tells this call that the earlier one deleted
    # the memory, so it names the key, and find_conflicts shows it gone
    with pytest.raises(ConflictsChanged) as error:
        store.resolve(conflicts, decision)
    assert error.value.keys == ("gone",)
    assert reasons(store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")) == []


def test_confirm_renews_expiry_and_counts(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index(), history=True, ttl={"default_ttl": 10.0})
    store.put(NS, "w", text(WINDOW), ttl=10.0)
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
        (PREFIX,),
    )
    admin.execute(
        f"UPDATE {store.sql.vectors} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
        (PREFIX,),
    )
    admin.commit()
    before = store.record(NS, "w")
    assert before is not None
    assert before.confirmations == 0
    assert before.confirmed_at is None
    before_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    before_vector_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]

    confirmed = store.confirm(NS, "w", ttl=10.0)

    assert confirmed is True
    after = store.record(NS, "w")
    assert after is not None
    assert after.confirmations == 1
    assert after.confirmed_at is not None
    assert after.version == before.version
    assert after.item.updated_at == before.item.updated_at
    # Expiry was renewed
    after_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert after_expiry > before_expiry
    # Vectors' expiry renewed too
    vector_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert vector_expiry is not None
    assert vector_expiry > before_vector_expiry
    assert abs((vector_expiry - after_expiry).total_seconds()) < 1.0
    # No history was added
    assert store.history(NS, "w") == []


def test_confirm_with_a_callable_ttl(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index(), ttl={"default_ttl": 10.0})
    store.put(NS, "w", text(WINDOW), ttl=10.0)
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
        (PREFIX,),
    )
    admin.execute(
        f"UPDATE {store.sql.vectors} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
        (PREFIX,),
    )
    admin.commit()

    # First confirmation: (0 + 1) * 60 = 60 minutes
    confirmed = store.confirm(NS, "w", ttl=lambda r: min((r.confirmations + 1) * 60, 600))
    assert confirmed is True
    first = store.record(NS, "w")
    assert first is not None
    assert first.confirmations == 1
    first_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    first_vector_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert abs((first_vector_expiry - first_expiry).total_seconds()) < 1.0

    # Second confirmation: (1 + 1) * 60 = 120 minutes
    confirmed = store.confirm(NS, "w", ttl=lambda r: min((r.confirmations + 1) * 60, 600))
    assert confirmed is True
    second = store.record(NS, "w")
    assert second is not None
    assert second.confirmations == 2
    second_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    second_vector_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    # Second expiry should be about 60 minutes later than first (120 - 60)
    delta_minutes = (second_expiry - first_expiry).total_seconds() / 60
    assert 59 < delta_minutes < 61
    assert abs((second_vector_expiry - second_expiry).total_seconds()) < 1.0


def test_confirm_with_ttl_none_removes_expiry(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index(), ttl={"default_ttl": 10.0})
    store.put(NS, "w", text(WINDOW), ttl=10.0)
    before = admin.execute(
        f"SELECT expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert before is not None
    before_vector = admin.execute(
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert before_vector is not None

    confirmed = store.confirm(NS, "w", ttl=None)

    assert confirmed is True
    after = admin.execute(
        f"SELECT expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert after is None
    # Vectors too
    vector_expiry = admin.execute(
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    ).fetchone()["expires_at"]
    assert vector_expiry is None


def test_confirm_returns_false_for_missing_or_expired(store_factory, admin, mode) -> None:
    store = store_factory(mode)

    # Missing key with a number
    confirmed = store.confirm(NS, "missing", ttl=10.0)
    assert confirmed is False

    # Missing key with a callable
    confirmed = store.confirm(NS, "missing-callable", ttl=lambda r: 10.0)
    assert confirmed is False

    # Expired key
    store.put(NS, "expired", text(WINDOW), ttl=10.0)
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() - INTERVAL '1 minute' WHERE prefix = %s AND key = 'expired'",
        (PREFIX,),
    )
    admin.commit()
    before_confirmations = admin.execute(
        f"SELECT confirmations FROM {store.sql.items} WHERE prefix = %s AND key = 'expired'",
        (PREFIX,),
    ).fetchone()["confirmations"]

    confirmed = store.confirm(NS, "expired", ttl=10.0)

    assert confirmed is False
    # Expired row is untouched
    after_confirmations = admin.execute(
        f"SELECT confirmations FROM {store.sql.items} WHERE prefix = %s AND key = 'expired'",
        (PREFIX,),
    ).fetchone()["confirmations"]
    assert after_confirmations == before_confirmations


def test_confirm_retried_after_a_lost_commit_counts_once(
    store_factory, admin, unsafe_internals, mode
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        retry_policy=RetryPolicy(max_attempts=3),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "w", text(WINDOW), ttl=10.0)

    fault.armed = True
    confirmed = store.confirm(NS, "w", ttl=20.0)

    assert confirmed is True
    assert fault.fired == 1
    # Should have incremented confirmations only once despite the retry
    record = store.record(NS, "w")
    assert record is not None
    assert record.confirmations == 1


def test_confirm_between_find_and_resolve_does_not_invalidate(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index(), history=True)
    store.put(NS, "w", in_slot(WINDOW))
    conflicts = store.find_conflicts(NS, None, in_slot(AISLE), slot="seat")
    assert len(conflicts.candidates) == 1
    assert conflicts.candidates[0].key == "w"

    # Confirming does not change version or updated_at, so resolve should succeed
    confirmed = store.confirm(NS, "w", ttl=10.0)
    assert confirmed is True

    decision = Decision(new="write", actions={"w": Action.CONFIRM})
    result = store.resolve(conflicts, decision)
    assert result.new == "written"
    assert result.confirmed == ("w",)
