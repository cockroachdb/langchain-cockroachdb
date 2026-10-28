"""find_conflicts and record: the memories a new one may conflict with, read without writing.

Candidates are live memories in the new memory's own namespace: the one
under its key, those in the same slot and, on a store with an index, those
whose vectors score at or above the threshold. Expired memories are never
candidates, and neither method pushes back an expiry. Without a valid_at
from the caller, the new memory is valid from the database's clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from psycopg import Connection

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore
from langchain_cockroachdb.store import Conflicts, MemoryRecord
from langchain_cockroachdb.store import base as store_base
from langchain_cockroachdb.store import reconcile as store_reconcile
from langchain_cockroachdb.store import store as store_module
from tests.integration.store.conftest import SEATS, seat_index
from tests.integration.store.test_schema_upgrade import schema_0_5

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

    # At the limit every vector of the namespace is ranked, with the same result
    exact = store_factory(mode, index=seat_index(), reconcile_scan_limit=5)
    sent = spy_on_reads(exact, monkeypatch)
    assert reasons(exact.find_conflicts(NS, None, text(AISLE))) == [
        ("lf", "similar"),
        ("w", "similar"),
    ]
    [(_, [similar], session)] = [s for s in sent if s[0] == "store find conflicts similar"]
    assert "GROUP BY i.prefix, i.key" in similar and session == []


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
