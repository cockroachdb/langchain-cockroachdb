"""The async store's reconcile, resolve, confirm, conditional writes and history.

AsyncCockroachDBStore plans, checks and decides with the code that
CockroachDBStore uses, and sends the same statements, on the caller's event
loop through the async retry machinery. These tests run its a-prefixed
coroutines against a real CockroachDB, in pool, connection and pipeline
modes: what each writes and returns, how a retry after a lost commit finds
its own write, how two concurrent calls serialize, and that a policy or a
confirm TTL function, plain functions run on the event loop inside the
transaction, cannot call the store.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import PutOp
from psycopg import AsyncConnection

from langchain_cockroachdb import AsyncCockroachDBStore
from langchain_cockroachdb._sql import parse_vector
from langchain_cockroachdb.store import (
    Action,
    ConditionalWrite,
    ConflictsChanged,
    Decision,
    MemoryRecord,
    MemoryVersion,
    ReconcileResult,
    keep_all,
    skip_duplicates,
    supersede_same_slot,
    supersede_similar,
)
from tests.integration.faults import FailAfterCommit
from tests.integration.store.conftest import FAST, SEATS, FixedEmbeddings, seat_index
from tests.integration.store.test_reconcile import (
    AISLE,
    LONG_FLIGHTS,
    NS,
    OLD,
    PEANUTS,
    PREFIX,
    RETRY_LOGGER,
    VEGETARIAN,
    WINDOW,
    in_slot,
    raised_within,
    reasons,
    retries,
    text,
    toward,
)
from tests.integration.store.test_schema_upgrade import schema_0_5

HISTORY = pytest.mark.parametrize("history", [False, True], ids=["no-history", "history"])
# The modes whose writes take the two paths a COMMIT can take: one batch on a
# connection of its own, and the component-wide pipeline
LOST_COMMIT_MODES = pytest.mark.parametrize("mode", ["pipeline", "pool"])
REFUSED = "A reconcile policy or a confirm TTL function must not call the store"


async def fetch(
    aadmin: AsyncConnection[Any], sql: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """The rows ``sql`` returns on the admin connection."""
    cur = await aadmin.execute(sql, params)
    return await cur.fetchall()


async def item_rows(aadmin: AsyncConnection[Any], store: Any) -> list[dict[str, Any]]:
    return await fetch(
        aadmin,
        f"SELECT prefix, key, value, version, updated_at, expires_at FROM {store.sql.items} "
        "ORDER BY prefix, key",
    )


async def history_rows(aadmin: AsyncConnection[Any], store: Any) -> list[dict[str, Any]]:
    """Every history row under PREFIX."""
    return await fetch(
        aadmin,
        f"SELECT key, version, reason, superseded_by, value, valid_at, invalid_at, write_id "
        f"FROM {store.sql.history} WHERE prefix = %s ORDER BY key, version",
        (PREFIX,),
    )


async def vector_keys(aadmin: AsyncConnection[Any], store: Any) -> list[str]:
    rows = await fetch(
        aadmin, f"SELECT key FROM {store.sql.vectors} WHERE prefix = %s ORDER BY key", (PREFIX,)
    )
    return [row["key"] for row in rows]


async def snapshot(aadmin: AsyncConnection[Any], store: Any) -> tuple[Any, ...]:
    """Every item and vector under PREFIX, and its history rows on a store with history."""
    archived = await history_rows(aadmin, store) if store._history is not None else []
    return (await item_rows(aadmin, store), await vector_keys(aadmin, store), archived)


async def stored_vector(aadmin: AsyncConnection[Any], store: Any, key: str) -> list[float]:
    [row] = await fetch(
        aadmin,
        f"SELECT embedding FROM {store.sql.vectors} WHERE prefix = %s AND key = %s",
        (PREFIX, key),
    )
    return parse_vector(row["embedding"])


async def expire_soon(aadmin: AsyncConnection[Any], store: Any) -> None:
    """Move the expiry of every item and vector under PREFIX to a minute from now."""
    for table in (store.sql.items, store.sql.vectors):
        await aadmin.execute(
            f"UPDATE {table} SET expires_at = now() + INTERVAL '1 minute' WHERE prefix = %s",
            (PREFIX,),
        )


async def renewed(aadmin: AsyncConnection[Any], store: Any, key: str) -> list[bool]:
    """For the item of ``key`` and then each of its vectors, whether it expires in over 5 minutes."""
    found: list[bool] = []
    for table in (store.sql.items, store.sql.vectors):
        rows = await fetch(
            aadmin,
            f"SELECT expires_at > now() + INTERVAL '5 minutes' AS renewed FROM {table} "
            "WHERE prefix = %s AND key = %s",
            (PREFIX, key),
        )
        found.extend(row["renewed"] for row in rows)
    return found


def spy_on_reads(store: AsyncCockroachDBStore, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the label, SQL and session statements of every read group the store sends."""
    sent: list[Any] = []
    real = store._read_all

    async def read_all(
        statements: list[tuple[str, list]], label: str, session: list[str] | None = None
    ) -> Any:
        sent.append((label, [sql for sql, _ in statements], session))
        return await real(statements, label, session)

    monkeypatch.setattr(store, "_read_all", read_all)
    return sent


def gate_writes(store: AsyncCockroachDBStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold the first write of each of two tasks until both reach theirs.

    Each task's first attempt then reads in its transaction before either
    task writes, as the sync tests arrange with a barrier in the policy, a
    plain function that cannot wait without blocking the event loop. A
    retry writes at once. The writes of areconcile, aput_if and adelete_if
    start with INSERT or DELETE, and the reads before them with SELECT or WITH.
    """
    send = store._send
    waited: set[asyncio.Task[Any] | None] = set()
    both = asyncio.Event()

    async def gated(conn: Any, statements: Any, cursors: Any) -> Any:
        task = asyncio.current_task()
        first = task not in waited and bool(statements)
        if first and statements[0][0].lstrip().startswith(("INSERT", "DELETE")):
            waited.add(task)
            if len(waited) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), timeout=10)
        return await send(conn, statements, cursors)

    monkeypatch.setattr(store, "_send", gated)


@contextlib.asynccontextmanager
async def open_store(
    libpq_url: str, table_name: str, mode: str, **kwargs: Any
) -> AsyncIterator[AsyncCockroachDBStore]:
    """A store from from_conn_string in ``mode`` on the running event loop, after setup()."""
    options: dict[str, Any] = {"table_name": table_name, "retry_policy": FAST, **kwargs}
    if mode == "pool":
        options["pool_config"] = {"min_size": 1, "max_size": 4}
    elif mode == "pipeline":
        options["pipeline"] = True
    async with AsyncCockroachDBStore.from_conn_string(libpq_url, **options) as store:
        await store.setup()
        yield store


# arecord and afind_conflicts


async def test_arecord(astore_factory, mode) -> None:
    store = await astore_factory(mode, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))

    record = await store.arecord(NS, "k")
    item = await store.aget(NS, "k")
    assert isinstance(record, MemoryRecord) and item is not None
    assert record.item == item
    assert (record.version, record.provenance, record.confirmations) == (1, None, 0)
    assert record.confirmed_at is None
    # A plain put stores no valid time: the memory is valid from its write
    assert record.valid_at == item.updated_at

    await store.aput(NS, "k", text(AISLE))
    record = await store.arecord(NS, "k")
    assert record is not None and (record.version, record.item.value) == (2, text(AISLE))
    assert await store.arecord(NS, "missing") is None
    assert await store.arecord(("user", "bob"), "k") is None


async def test_afind_conflicts_returns_key_slot_and_similar_in_that_order(
    astore_factory, aadmin, mode
) -> None:
    store = await astore_factory(mode, index=seat_index())
    await store.aput(NS, "k", text(PEANUTS))
    await store.aput(NS, "s", text(VEGETARIAN, seat="x"))
    await store.aput(NS, "s2", text(PEANUTS, seat="x"), index=False)  # same slot, no vector
    await store.aput(NS, "w", text(WINDOW))
    await store.aput(NS, "other", text(VEGETARIAN, seat="y"))
    # A descendant namespace holds matches too
    await store.aput(("user", "alice", "trips"), "child", text(WINDOW, seat="x"))
    before = await item_rows(aadmin, store)

    conflicts = await store.afind_conflicts(
        NS, "k", text(AISLE, seat="x"), slot="seat", provenance={"source": "chat"}
    )

    assert reasons(conflicts) == [("k", "key"), ("s", "slot"), ("s2", "slot"), ("w", "similar")]
    k, _, s2, w = conflicts.candidates
    assert w.score == pytest.approx(0.85, abs=1e-3)
    assert k.score == pytest.approx(0.0, abs=1e-6) and s2.score is None
    item = await store.aget(NS, "k")
    assert item is not None
    assert (conflicts.target_version, conflicts.target_updated_at) == (1, item.updated_at)
    assert conflicts._state is not None
    assert conflicts._state.tokens == {
        c.key: (c.version, c.item.updated_at) for c in conflicts.candidates
    }
    proposal = conflicts.proposal
    assert (proposal.key, proposal.slot_value, proposal.provenance) == (
        "k",
        "x",
        {"source": "chat"},
    )
    # Nothing was written
    assert await item_rows(aadmin, store) == before


async def test_afind_conflicts_ignores_expired_memories(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=seat_index())
    await store.aput(NS, "k", text(WINDOW), ttl=60)
    await store.aput(NS, "s", text(VEGETARIAN, seat="x"), ttl=60)
    await store.aput(NS, "w", text(WINDOW), ttl=60)
    await store.aput(NS, "live", text(LONG_FLIGHTS))
    await aadmin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() - INTERVAL '1 minute' "
        "WHERE prefix = %s AND key <> 'live'",
        (PREFIX,),
    )

    conflicts = await store.afind_conflicts(NS, "k", text(AISLE, seat="x"), slot="seat")

    assert reasons(conflicts) == [("live", "similar")]
    # The expired target counts as absent, the same as no row
    assert conflicts.target_version is None and conflicts.target_updated_at is None
    assert conflicts._state is not None and conflicts._state.target is None
    assert await store.arecord(NS, "k") is None
    assert await store.arecord(NS, "live") is not None
    # Nor does areconcile's transaction see them: the live memory is the only candidate
    result = await store.areconcile(NS, None, text(AISLE, seat="x"), policy=keep_all(), slot="seat")
    assert (result.new, result.kept) == ("written", ("live",))


async def test_afind_conflicts_and_areconcile_look_past_a_window_the_filter_emptied(
    astore_factory, mode, monkeypatch
) -> None:
    # Twenty-nine memories of another kind lie nearer to the new memory than
    # the only match, all above the threshold. The index's window of 21 holds
    # none of the match's kind and still ends above the threshold, so matches
    # may lie beyond it, and the store ranks the namespace exactly
    vectors = {AISLE: SEATS[AISLE], "match": toward(SEATS[AISLE], 0.85)}
    for i in range(29):
        vectors[f"near {i}"] = toward(SEATS[AISLE], 0.99 - 0.001 * i)
    index = {"dims": 4, "embed": FixedEmbeddings(vectors), "fields": ["text"], "beam_size": 8}
    store = await astore_factory(mode, history=True, index=index, reconcile_scan_limit=10)
    await store.abatch(
        [PutOp(NS, f"o{i}", {"text": f"near {i}", "kind": "other"}) for i in range(29)]
        + [PutOp(NS, "m", {"text": "match", "kind": "seat"})]
    )
    sent = spy_on_reads(store, monkeypatch)

    conflicts = await store.afind_conflicts(
        NS, None, text(AISLE, kind="seat"), filter={"kind": "seat"}
    )

    assert reasons(conflicts) == [("m", "similar")]
    assert conflicts.candidates[0].score == pytest.approx(0.85, abs=1e-3)
    assert [label for label, _, _ in sent] == [
        "store find conflicts",
        "store find conflicts similar",
        "store find conflicts exact",
    ]
    # The index scan set the beam size in its read's transaction
    assert sent[1][2] == ["SET LOCAL vector_search_beam_size = 8"]

    # areconcile runs the same reads inside its transaction, the beam size
    # set there, and its policy sees the match beyond the window
    result = await store.areconcile(
        NS,
        None,
        text(AISLE, kind="seat"),
        policy=supersede_similar(threshold=0.8),
        filter={"kind": "seat"},
    )
    assert (result.new, result.superseded) == ("written", ("m",))
    [old] = await store.ahistory(NS, "m")
    assert (old.reason, old.superseded_by) == ("superseded", result.key)


# areconcile


async def test_areconcile_supersedes_in_the_same_slot(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    first = await store.areconcile(
        NS, None, in_slot(WINDOW), policy=supersede_same_slot(), slot="seat"
    )
    second = await store.areconcile(
        NS,
        None,
        in_slot(AISLE),
        policy=supersede_same_slot(),
        slot="seat",
        provenance={"source": "chat"},
    )

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
    assert [item.key for item in await store.asearch(NS)] == [second.key]
    [old] = await store.ahistory(NS, first.key)
    assert (old.version, old.reason, old.superseded_by, old.value) == (
        1,
        "superseded",
        second.key,
        in_slot(WINDOW),
    )
    new = await store.arecord(NS, second.key)
    assert new is not None and new.provenance == {"source": "chat"}
    # No valid time was given: the new memory is valid from its write, and
    # the fact it superseded ended at that write
    assert new.valid_at == new.item.updated_at == old.invalid_at
    # The superseded memory's vectors went with it
    assert await vector_keys(aadmin, store) == [second.key]


async def test_areconcile_skips_and_confirms_a_duplicate(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, history=True, index=seat_index(), ttl={"default_ttl": 10.0})
    window = await store.areconcile(NS, None, text(WINDOW), policy=keep_all())
    await expire_soon(aadmin, store)
    before = await store.arecord(NS, window.key)
    assert before is not None and before.confirmations == 0

    result = await store.areconcile(NS, None, text(LONG_FLIGHTS), policy=skip_duplicates())

    assert result == ReconcileResult(
        key=result.key,
        new="skipped",
        version=None,
        superseded=(),
        deleted=(),
        confirmed=(window.key,),
        kept=(),
    )
    record = await store.arecord(NS, window.key)
    assert record is not None and record.confirmed_at is not None
    assert (record.version, record.confirmations) == (1, 1)
    # Confirming changes neither the version nor updated_at, and keeps no history
    assert record.item.updated_at == before.item.updated_at
    assert await store.ahistory(NS, window.key) == []
    # Nothing was written for the duplicate
    assert await store.arecord(NS, result.key) is None
    # The memory's expiry and its vector's were renewed from its own TTL
    assert await renewed(aadmin, store, window.key) == [True, True]


async def test_areconcile_records_a_late_fact_as_superseded(astore_factory, mode) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    live = await store.areconcile(
        NS, None, in_slot(WINDOW), policy=supersede_same_slot(), slot="seat"
    )

    late = await store.areconcile(
        NS,
        None,
        in_slot(AISLE),
        policy=supersede_same_slot(),
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
    assert [item.key for item in await store.asearch(NS)] == [live.key]
    assert await store.arecord(NS, late.key) is None
    current = await store.arecord(NS, live.key)
    assert current is not None and current.version == 1
    assert await store.ahistory(NS, live.key) == []
    [row] = await store.ahistory(NS, late.key)
    assert (row.version, row.reason, row.superseded_by) == (1, "superseded", live.key)
    assert (row.value, row.provenance) == (in_slot(AISLE), {"source": "old chat"})
    # True from its own valid time until the newer fact became true. The store
    # held it for no time
    assert (row.valid_at, row.invalid_at) == (OLD, current.valid_at)
    assert row.written_at == row.ended_at


@pytest.mark.parametrize("failure", ["raises", "invalid decision"])
async def test_an_areconcile_policy_exception_rolls_back(
    astore_factory, aadmin, caplog, mode, failure: str
) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    first = await store.areconcile(
        NS, None, in_slot(WINDOW), policy=supersede_same_slot(), slot="seat"
    )
    before = await snapshot(aadmin, store)
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
        await store.areconcile(NS, None, in_slot(AISLE), policy=policy, slot="seat")

    # The policy ran once, inside the transaction, and saw the slot fact. The
    # error came out unchanged, nothing was retried and nothing was written
    assert seen == [[first.key]]
    if failure == "raises":
        assert error.value is raised
    else:
        assert "unknown candidate key: missing" in str(error.value)
    assert not retries(caplog, "store reconcile")
    assert await snapshot(aadmin, store) == before
    # The transaction was rolled back, so the store's connection works on
    second = await store.areconcile(
        NS, None, in_slot(AISLE), policy=supersede_same_slot(), slot="seat"
    )
    assert second.superseded == (first.key,)


@LOST_COMMIT_MODES
async def test_areconcile_retried_after_a_lost_commit_is_already_applied(
    astore_factory, aadmin, unsafe_internals, caplog, mode
) -> None:
    # The reconcile commits, then the store sees 40003 and runs it again. The
    # retry finds the rows that carry its write id and writes nothing
    fault = FailAfterCommit(unsafe_internals)
    store = await astore_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.async_cursor_class()},
    )
    first = await store.areconcile(
        NS, None, in_slot(WINDOW), policy=supersede_same_slot(), slot="seat"
    )
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        second = await store.areconcile(
            NS, None, in_slot(AISLE), policy=supersede_same_slot(), slot="seat"
        )

    assert fault.fired == 1
    found = retries(caplog, "store reconcile")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    # Rebuilt from the rows, with the superseded key a clean run reports
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
    assert [(row["key"], row["reason"]) for row in await history_rows(aadmin, store)] == [
        (first.key, "superseded")
    ]
    assert [item.key for item in await store.asearch(NS)] == [second.key]
    assert await vector_keys(aadmin, store) == [second.key]


async def test_two_areconciles_in_one_namespace_leave_one_live_fact(
    astore_factory, aadmin, caplog, monkeypatch
) -> None:
    # Both read the slot before either writes: each finds it empty. Serializable
    # isolation then retries one of them, and its policy sees the other's fact
    store = await astore_factory("pool", history=True, index=seat_index())
    gate_writes(store, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        results = await asyncio.gather(
            *(
                store.areconcile(
                    NS, None, in_slot(value), policy=supersede_same_slot(), slot="seat"
                )
                for value in (WINDOW, AISLE)
            )
        )

    assert retries(caplog, "store reconcile")
    assert [result.new for result in results] == ["written", "written"]
    live = await fetch(
        aadmin,
        f"SELECT key FROM {store.sql.items} WHERE prefix = %s AND value ->> 'seat' = 'x'",
        (PREFIX,),
    )
    assert len(live) == 1
    winner = next(result for result in results if result.key == live[0]["key"])
    loser = next(result for result in results if result is not winner)
    assert winner.superseded == (loser.key,)
    [old] = await store.ahistory(NS, loser.key)
    assert (old.reason, old.superseded_by) == ("superseded", winner.key)
    assert await store.ahistory(NS, winner.key) == []


# aresolve


async def test_aresolve_raises_conflicts_changed(astore_factory, aadmin, caplog, mode) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "old", in_slot(WINDOW))
    await store.aput(NS, "dup", text(LONG_FLIGHTS))
    conflicts = await store.afind_conflicts(NS, None, in_slot(AISLE), slot="seat")
    assert reasons(conflicts) == [("old", "slot"), ("dup", "similar")]
    decision = Decision(actions={"old": Action.SUPERSEDE, "dup": Action.CONFIRM})
    # Another writer replaces the memory the decision supersedes
    await store.aput(NS, "old", in_slot(LONG_FLIGHTS))
    before = await snapshot(aadmin, store)

    with (
        caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
        pytest.raises(ConflictsChanged) as error,
    ):
        await store.aresolve(conflicts, decision)

    assert error.value.keys == ("old",)
    # Nothing was written, not even the confirmation, and nothing was retried
    assert await snapshot(aadmin, store) == before
    dup = await store.arecord(NS, "dup")
    assert dup is not None and dup.confirmations == 0
    assert not retries(caplog, "store resolve")
    # Found again, the same decision applies
    again = await store.afind_conflicts(NS, None, in_slot(AISLE), slot="seat")
    result = await store.aresolve(again, decision)
    assert result == ReconcileResult(
        key=again.proposal.key,
        new="written",
        version=1,
        superseded=("old",),
        deleted=(),
        confirmed=("dup",),
        kept=(),
    )
    assert [(v.version, v.reason) for v in await store.ahistory(NS, "old")] == [
        (2, "superseded"),
        (1, "overwritten"),
    ]


async def test_aresolve_detects_a_0_5_style_write_through_updated_at(
    astore_factory, aadmin, mode
) -> None:
    store = await astore_factory(mode, index=seat_index())
    await store.aput(NS, "old", in_slot(WINDOW))
    conflicts = await store.afind_conflicts(NS, None, in_slot(AISLE), slot="seat")
    # A 0.5.x store writes with the 0.5 statement during a rolling upgrade: it
    # sets updated_at and leaves the version as it was
    value = json.dumps(in_slot(LONG_FLIGHTS))
    await aadmin.execute(*store.sql.upsert_items([(PREFIX, "old", value, None)]))
    record = await store.arecord(NS, "old")
    assert record is not None and (record.version, record.item.value) == (1, in_slot(LONG_FLIGHTS))
    before = await snapshot(aadmin, store)

    with pytest.raises(ConflictsChanged) as error:
        await store.aresolve(conflicts, Decision(actions={"old": Action.SUPERSEDE}))

    assert error.value.keys == ("old",)
    assert await snapshot(aadmin, store) == before


async def test_aresolve_accepts_a_conflicts_and_a_decision_from_a_checkpoint(
    astore_factory, aadmin, mode
) -> None:
    # A LangGraph interrupt keeps both in a checkpoint while a person decides,
    # and the process that resumes the graph has a store of its own
    serde = JsonPlusSerializer()
    finder = await astore_factory(mode, history=True, index=seat_index())
    await finder.aput(NS, "k", text(PEANUTS))
    await finder.aput(NS, "old", in_slot(WINDOW))
    await finder.aput(NS, "dup", text(LONG_FLIGHTS))
    conflicts = await finder.afind_conflicts(
        NS, "k", in_slot(AISLE), slot="seat", provenance={"source": "chat"}
    )
    decision = Decision(actions={"old": Action.SUPERSEDE, "dup": Action.CONFIRM})
    saved = serde.loads_typed(serde.dumps_typed(conflicts))
    decided = serde.loads_typed(serde.dumps_typed(decision))
    assert (saved, decided) == (conflicts, decision)
    resumed = await astore_factory(mode, history=True, index=seat_index())

    result = await resumed.aresolve(saved, decided)

    assert result == ReconcileResult(
        key="k",
        new="written",
        version=2,
        superseded=("old",),
        deleted=(),
        confirmed=("dup",),
        kept=(),
    )
    record = await resumed.arecord(NS, "k")
    assert record is not None
    assert (record.item.value, record.provenance) == (in_slot(AISLE), {"source": "chat"})
    # The memory's vector is the one find_conflicts embedded
    assert await stored_vector(aadmin, resumed, "k") == pytest.approx(SEATS[AISLE], abs=1e-6)


@LOST_COMMIT_MODES
async def test_aresolve_retried_after_a_lost_commit_is_already_applied(
    astore_factory, aadmin, unsafe_internals, caplog, mode
) -> None:
    # The resolve commits, then the store sees 40003 and runs it again. The
    # retry finds the rows that carry its write id, not a conflict
    fault = FailAfterCommit(unsafe_internals)
    store = await astore_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.async_cursor_class()},
    )
    await store.aput(NS, "old", in_slot(WINDOW))
    await store.aput(NS, "dup", text(LONG_FLIGHTS))
    conflicts = await store.afind_conflicts(NS, None, in_slot(AISLE), slot="seat")
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = await store.aresolve(
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
    assert [(row["key"], row["reason"]) for row in await history_rows(aadmin, store)] == [
        ("old", "superseded")
    ]
    assert {item.key for item in await store.asearch(NS)} == {result.key, "dup"}
    dup = await store.arecord(NS, "dup")
    assert dup is not None and dup.confirmations == 1


@LOST_COMMIT_MODES
async def test_an_aresolve_that_only_deletes_finishes_after_a_lost_commit_without_history(
    astore_factory, aadmin, unsafe_internals, caplog, mode
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = await astore_factory(
        mode, index=seat_index(), connect_kwargs={"cursor_factory": fault.async_cursor_class()}
    )
    await store.aput(NS, "taken", in_slot(PEANUTS))
    await store.aput(NS, "gone", in_slot(WINDOW))
    await store.aput(NS, "also", in_slot(VEGETARIAN))
    await store.aput(NS, "kept", text(LONG_FLIGHTS))
    # The first attempt is strict: a memory another writer deleted after
    # afind_conflicts is a change, even one the decision deletes
    early = await store.afind_conflicts(NS, None, in_slot(AISLE), slot="seat")
    await store.adelete(NS, "taken")
    with pytest.raises(ConflictsChanged) as error:
        await store.aresolve(early, Decision(new="skip", actions={"taken": Action.DELETE}))
    assert error.value.keys == ("taken",)

    conflicts = await store.afind_conflicts(NS, None, in_slot(AISLE), slot="seat")
    assert reasons(conflicts) == [("gone", "slot"), ("also", "slot"), ("kept", "similar")]
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = await store.aresolve(
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
    assert await store.aget(NS, "gone") is None and await store.aget(NS, "also") is None
    assert await vector_keys(aadmin, store) == ["kept"]


# aconfirm


async def test_aconfirm_renews_and_counts(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=seat_index(), history=True, ttl={"default_ttl": 10.0})
    await store.aput(NS, "w", text(WINDOW))
    await expire_soon(aadmin, store)
    before = await store.arecord(NS, "w")
    assert before is not None and (before.confirmations, before.confirmed_at) == (0, None)

    # Without a ttl the expiry is renewed from the memory's own TTL
    assert await store.aconfirm(NS, "w") is True

    after = await store.arecord(NS, "w")
    assert after is not None and after.confirmed_at is not None
    assert (after.version, after.confirmations) == (before.version, 1)
    assert after.item.updated_at == before.item.updated_at
    assert await renewed(aadmin, store, "w") == [True, True]
    # Confirming keeps no history
    assert await store.ahistory(NS, "w") == []

    # A number sets a new TTL, and None removes the expiry, of the memory and its vectors
    assert await store.aconfirm(NS, "w", ttl=60.0) is True
    [row] = await fetch(
        aadmin,
        f"SELECT ttl_minutes, confirmations FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    )
    assert (row["ttl_minutes"], row["confirmations"]) == (60, 2)
    assert await store.aconfirm(NS, "w", ttl=None) is True
    for table in (store.sql.items, store.sql.vectors):
        [row] = await fetch(
            aadmin, f"SELECT expires_at FROM {table} WHERE prefix = %s AND key = 'w'", (PREFIX,)
        )
        assert row["expires_at"] is None

    # No live memory has the key: nothing to confirm
    assert await store.aconfirm(NS, "missing") is False
    assert await store.aconfirm(NS, "missing", ttl=lambda record: 10.0) is False


async def test_aconfirm_with_a_callable_ttl(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, index=seat_index(), ttl={"default_ttl": 10.0})
    await store.aput(NS, "w", text(WINDOW))
    seen: list[int] = []

    def ttl(record: MemoryRecord) -> float:
        # Longer with each confirmation, from the record read in the transaction
        seen.append(record.confirmations)
        return min((record.confirmations + 1) * 60, 600)

    assert await store.aconfirm(NS, "w", ttl=ttl) is True
    assert await store.aconfirm(NS, "w", ttl=ttl) is True

    assert seen == [0, 1]
    [item] = await fetch(
        aadmin,
        f"SELECT ttl_minutes, expires_at FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    )
    [vector] = await fetch(
        aadmin,
        f"SELECT expires_at FROM {store.sql.vectors} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    )
    assert item["ttl_minutes"] == 120
    assert abs((vector["expires_at"] - item["expires_at"]).total_seconds()) < 1.0
    # A function that returns neither None nor a positive number fails the call
    with pytest.raises(ValueError, match="ttl must be"):
        await store.aconfirm(NS, "w", ttl=lambda record: 0)
    record = await store.arecord(NS, "w")
    assert record is not None and record.confirmations == 2


@LOST_COMMIT_MODES
async def test_a_ttl_function_runs_once_when_an_aconfirm_retries_after_a_lost_commit(
    astore_factory, aadmin, unsafe_internals, caplog, mode
) -> None:
    # The confirm commits and its result is lost. The retry reads the memory,
    # finds its own write id on it, and returns without calling the function again
    fault = FailAfterCommit(unsafe_internals)
    store = await astore_factory(
        mode, connect_kwargs={"cursor_factory": fault.async_cursor_class()}
    )
    await store.aput(NS, "w", text(WINDOW), ttl=10.0)
    calls: list[int] = []

    def ttl(record: MemoryRecord) -> float:
        calls.append(record.confirmations)
        return 20.0

    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        confirmed = await store.aconfirm(NS, "w", ttl=ttl)

    assert confirmed is True and fault.fired == 1
    found = retries(caplog, "store confirm")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    # Called once, with the memory as the first attempt read it
    assert calls == [0]
    record = await store.arecord(NS, "w")
    assert record is not None and record.confirmations == 1
    [row] = await fetch(
        aadmin,
        f"SELECT ttl_minutes FROM {store.sql.items} WHERE prefix = %s AND key = 'w'",
        (PREFIX,),
    )
    assert row["ttl_minutes"] == 20


# ahistory and apurge


async def test_ahistory_and_apurge(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))
    await store.aput(NS, "k", text(AISLE))
    await store.aput(NS, "k", text(LONG_FLIGHTS))
    await store.adelete(NS, "k")

    versions = await store.ahistory(NS, "k")

    # Newest first, the live value never among them
    assert [(v.version, v.reason, v.value) for v in versions] == [
        (3, "deleted", text(LONG_FLIGHTS)),
        (2, "overwritten", text(AISLE)),
        (1, "overwritten", text(WINDOW)),
    ]
    assert all(isinstance(v, MemoryVersion) and (v.namespace, v.key) == (NS, "k") for v in versions)
    assert [v.version for v in await store.ahistory(NS, "k", limit=2)] == [3, 2]
    assert await store.ahistory(NS, "k", limit=0) == []

    # The key's versions go on after its history, and purge forgets all of
    # it: the memory, its vectors and its history
    await store.aput(NS, "k", text(VEGETARIAN))
    await store.aput(NS, "other", text(PEANUTS))
    record = await store.arecord(NS, "k")
    assert record is not None and record.version == 4
    assert await store.apurge(NS, "k") is True
    assert await store.aget(NS, "k") is None and await store.ahistory(NS, "k") == []
    assert await vector_keys(aadmin, store) == ["other"]
    assert [row["key"] for row in await history_rows(aadmin, store)] == []
    # Nothing left to purge
    assert await store.apurge(NS, "k") is False

    # A store without history has neither
    plain = await astore_factory(mode, index=seat_index())
    with pytest.raises(ValueError, match="history=True"):
        await plain.ahistory(NS, "other")
    with pytest.raises(ValueError, match="history=True"):
        await plain.apurge(NS, "other")


# aput_if and adelete_if


@HISTORY
async def test_aput_if_applies_and_returns_the_current_record_on_a_conflict(
    astore_factory, aadmin, mode, history
) -> None:
    store = await astore_factory(mode, history=history, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))
    record = await store.arecord(NS, "k")
    assert record is not None

    applied = await store.aput_if(
        NS, "k", text(AISLE), expected=record, provenance={"source": "chat"}
    )

    assert applied == ConditionalWrite(applied=True, key="k", version=2, current=None)
    now = await store.arecord(NS, "k")
    assert now is not None
    assert (now.version, now.item.value, now.provenance) == (2, text(AISLE), {"source": "chat"})
    assert await stored_vector(aadmin, store, "k") == pytest.approx(SEATS[AISLE], abs=1e-6)
    if history:
        [old] = await store.ahistory(NS, "k")
        assert (old.version, old.reason, old.value) == (1, "overwritten", text(WINDOW))

    # With the record read before that write: nothing is written, and the
    # current memory comes back, to merge or to reject
    before = await snapshot(aadmin, store)
    conflict = await store.aput_if(NS, "k", text(LONG_FLIGHTS), expected=record)
    assert conflict == ConditionalWrite(applied=False, key="k", version=None, current=now)
    assert await snapshot(aadmin, store) == before

    # Create only: the first call creates the memory, the second finds it
    created = await store.aput_if(NS, "new", text(VEGETARIAN), expected=None)
    assert created == ConditionalWrite(applied=True, key="new", version=1, current=None)
    again = await store.aput_if(NS, "new", text(PEANUTS), expected=None)
    assert again == ConditionalWrite(
        applied=False, key="new", version=None, current=await store.arecord(NS, "new")
    )


@HISTORY
async def test_aput_if_detects_a_0_5_style_write_through_updated_at(
    astore_factory, aadmin, mode, history
) -> None:
    store = await astore_factory(mode, history=history, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))
    record = await store.arecord(NS, "k")
    # A 0.5.x store writes with the 0.5 statement during a rolling upgrade: it
    # sets updated_at and leaves the version as it was
    await aadmin.execute(
        *store.sql.upsert_items([(PREFIX, "k", json.dumps(text(LONG_FLIGHTS)), None)])
    )
    current = await store.arecord(NS, "k")
    assert record is not None and current is not None
    assert (current.version, current.item.value) == (1, text(LONG_FLIGHTS))
    before = await snapshot(aadmin, store)

    result = await store.aput_if(NS, "k", text(AISLE), expected=record)

    assert result == ConditionalWrite(applied=False, key="k", version=None, current=current)
    assert await snapshot(aadmin, store) == before


@HISTORY
async def test_adelete_if(astore_factory, aadmin, mode, history) -> None:
    store = await astore_factory(mode, history=history, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))
    stale = await store.arecord(NS, "k")
    await store.aput(NS, "k", text(AISLE))
    before = await snapshot(aadmin, store)

    # Written since the record was read: nothing is deleted, and the current memory comes back
    conflict = await store.adelete_if(NS, "k", expected=stale)
    current = await store.arecord(NS, "k")
    assert current is not None and current.version == 2
    assert conflict == ConditionalWrite(applied=False, key="k", version=None, current=current)
    assert await snapshot(aadmin, store) == before

    # With the current record it deletes as adelete does
    deleted = await store.adelete_if(NS, "k", expected=current)
    assert deleted == ConditionalWrite(applied=True, key="k", version=None, current=None)
    assert await store.aget(NS, "k") is None and await vector_keys(aadmin, store) == []
    if history:
        assert [(v.version, v.reason) for v in await store.ahistory(NS, "k")] == [
            (2, "deleted"),
            (1, "overwritten"),
        ]

    # Gone since: a conflict with no current memory
    gone = await store.adelete_if(NS, "k", expected=current)
    assert gone == ConditionalWrite(applied=False, key="k", version=None, current=None)
    # With expected=None it applies only while no live memory has the key
    absent = await store.adelete_if(NS, "k", expected=None)
    assert absent == ConditionalWrite(applied=True, key="k", version=None, current=None)
    await store.aput(NS, "k", text(VEGETARIAN))
    live = await store.adelete_if(NS, "k", expected=None)
    assert live == ConditionalWrite(
        applied=False, key="k", version=None, current=await store.arecord(NS, "k")
    )
    assert await store.aget(NS, "k") is not None


@HISTORY
async def test_two_aput_ifs_from_the_same_record_one_wins(
    astore_factory, aadmin, caplog, monkeypatch, history
) -> None:
    # Both read the memory in their transactions before either writes, so both
    # find it unchanged. Serializable isolation then retries one of them, and
    # its check sees the other's write
    store = await astore_factory("pool", history=history, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))
    record = await store.arecord(NS, "k")
    gate_writes(store, monkeypatch)
    values = [text(AISLE), text(LONG_FLIGHTS)]

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        results = await asyncio.gather(
            *(store.aput_if(NS, "k", value, expected=record) for value in values)
        )

    assert retries(caplog, "store put_if")
    assert sorted(result.applied for result in results) == [False, True]
    won = next(i for i, result in enumerate(results) if result.applied)
    assert results[won] == ConditionalWrite(applied=True, key="k", version=2, current=None)
    # The other gets the winner's memory, to merge or to reject
    current = await store.arecord(NS, "k")
    assert current is not None and current.item.value == values[won]
    assert results[1 - won] == ConditionalWrite(
        applied=False, key="k", version=None, current=current
    )
    vector = await stored_vector(aadmin, store, "k")
    assert vector == pytest.approx(SEATS[values[won]["text"]], abs=1e-6)
    if history:
        assert [(v.version, v.reason, v.value) for v in await store.ahistory(NS, "k")] == [
            (1, "overwritten", text(WINDOW))
        ]


@HISTORY
async def test_two_adelete_ifs_from_the_same_record_one_wins(
    astore_factory, aadmin, caplog, monkeypatch, history
) -> None:
    # Both read the memory before either deletes it. Serializable isolation
    # rolls one of them back with a 40001, so its retry checks as strictly
    # as its first attempt and finds the memory gone: the other removed it
    store = await astore_factory("pool", history=history, index=seat_index())
    await store.aput(NS, "k", text(WINDOW))
    record = await store.arecord(NS, "k")
    gate_writes(store, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        results = await asyncio.gather(
            *(store.adelete_if(NS, "k", expected=record) for _ in range(2))
        )

    found = retries(caplog, "store delete_if")
    assert found and all("SQLSTATE 40001" in message for message in found)
    assert sorted(result.applied for result in results) == [False, True]
    assert ConditionalWrite(applied=False, key="k", version=None, current=None) in results
    assert await store.aget(NS, "k") is None and await vector_keys(aadmin, store) == []
    if history:
        assert [(v.version, v.reason) for v in await store.ahistory(NS, "k")] == [(1, "deleted")]


@LOST_COMMIT_MODES
@HISTORY
async def test_aput_if_and_adelete_if_retried_after_a_lost_commit(
    astore_factory, aadmin, unsafe_internals, caplog, mode, history
) -> None:
    # Each commits, then the store sees 40003 and runs it again. The retry
    # finds its write: put_if's write id on the row, delete_if's memory gone
    fault = FailAfterCommit(unsafe_internals)
    index = seat_index()
    store = await astore_factory(
        mode,
        history=history,
        index=index,
        connect_kwargs={"cursor_factory": fault.async_cursor_class()},
    )
    await store.aput(NS, "k", text(WINDOW))
    record = await store.arecord(NS, "k")
    embedded = index["embed"].calls
    fault.armed = True
    # Numbers too: the retry compares the stored value with its own
    value = text(AISLE, stars=1.0, count=3)

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = await store.aput_if(NS, "k", value, expected=record)

    assert fault.fired == 1
    found = retries(caplog, "store put_if")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    assert result == ConditionalWrite(
        applied=True, key="k", version=2, current=None, already_applied=True
    )
    # Embedded once, before the first attempt: the retry reused the vectors
    assert index["embed"].calls == embedded + 1
    # One version bump and, with history, one archived version
    now = await store.arecord(NS, "k")
    assert now is not None and (now.version, now.item.value) == (2, value)
    if history:
        assert [(v.version, v.reason) for v in await store.ahistory(NS, "k")] == [
            (1, "overwritten")
        ]

    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        deleted = await store.adelete_if(NS, "k", expected=now)

    assert fault.fired == 2
    assert len(retries(caplog, "store delete_if")) == 1
    # The retry finds the memory gone, the state delete_if asked for. With
    # history its own history row proves the earlier attempt deleted it
    assert deleted == ConditionalWrite(
        applied=True, key="k", version=None, current=None, already_applied=history
    )
    assert await store.aget(NS, "k") is None and await vector_keys(aadmin, store) == []


# Before setup(), and a store call from a policy or a TTL function


async def test_the_async_methods_need_setup(astore_factory, aadmin, mode) -> None:
    index = seat_index()
    store = await astore_factory(mode, setup=False, index=index)
    for statement in schema_0_5(store):
        await aadmin.execute(statement)

    for method, args, kwargs in (
        ("afind_conflicts", (NS, "k", text(AISLE)), {}),
        ("arecord", (NS, "k"), {}),
        ("areconcile", (NS, "k", text(AISLE)), {"policy": keep_all()}),
        ("aconfirm", (NS, "k"), {}),
        ("aput_if", (NS, "k", text(AISLE)), {"expected": None}),
        ("adelete_if", (NS, "k"), {"expected": None}),
    ):
        with pytest.raises(ValueError, match=r"setup\(\)"):
            await getattr(store, method)(*args, **kwargs)
    # Refused before anything was embedded or written
    assert index["embed"].calls == 0
    [row] = await fetch(aadmin, f"SELECT count(*) AS n FROM {store.sql.items}")
    assert row["n"] == 0

    await store.setup()
    result = await store.areconcile(NS, "k", text(AISLE), policy=keep_all())
    assert (result.new, result.version) == ("written", 1)
    record = await store.arecord(NS, "k")
    assert record is not None and record.item.value == text(AISLE)


def test_a_policy_or_ttl_function_that_calls_the_async_store_fails_at_once(
    libpq_url, table_name, mode
) -> None:
    # Both are plain functions that run on the store's event loop, inside the
    # transaction. A store call from either fails at once, with the advice
    # the sync store gives, and nothing is written. The store runs on an event
    # loop of its own in another thread, so a call that blocked that loop
    # would fail this test after the timeout instead of hanging the suite
    async def scenario() -> None:
        async with open_store(
            libpq_url, table_name, mode, history=True, index=seat_index()
        ) as store:
            await store.aput(NS, "w", in_slot(WINDOW))
            ran: list[str] = []

            def policy(proposal: Any, candidates: Any) -> Decision:
                ran.append("policy")
                store.get(NS, "w")
                return Decision()

            def ttl(record: MemoryRecord) -> float:
                ran.append("ttl")
                store.get(NS, "w")
                return 10.0

            with pytest.raises(RuntimeError, match=REFUSED):
                await store.areconcile(NS, None, in_slot(AISLE), policy=policy, slot="seat")
            with pytest.raises(RuntimeError, match=REFUSED):
                await store.aconfirm(NS, "w", ttl=ttl)

            # Each ran once: the error is not retried. Both transactions were
            # rolled back and the connection given back, so the store works on
            assert ran == ["policy", "ttl"]
            assert [item.key for item in await store.asearch(NS)] == ["w"]
            record = await store.arecord(NS, "w")
            assert record is not None and (record.version, record.confirmations) == (1, 0)
            assert await store.ahistory(NS, "w") == []

    assert raised_within(lambda: asyncio.run(scenario()), timeout=60.0) is None
