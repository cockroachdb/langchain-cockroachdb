"""find_conflicts and record: the memories a new one may conflict with, read without writing.

Candidates are live memories in the new memory's own namespace: the one
under its key, those in the same slot and, on a store with an index, those
whose vectors score at or above the threshold. Expired memories are never
candidates, and neither method pushes back an expiry. Without a valid_at
from the caller, the new memory is valid from the database's clock.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import PutOp
from psycopg import Connection

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore
from langchain_cockroachdb.store import Conflicts, MemoryRecord
from langchain_cockroachdb.store import base as store_base
from langchain_cockroachdb.store import reconcile as store_reconcile
from langchain_cockroachdb.store import store as store_module
from langchain_cockroachdb.store.sql import StoreSQL
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
