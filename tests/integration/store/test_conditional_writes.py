"""put_if and delete_if: writes that apply only to the memory the caller read.

Two threads that read the same memory with record() and both write it back
must not lose one update without a word. put_if and delete_if check, in the
transaction of the write, that the live memory still has the version and
updated_at of the record the caller read, as resolve checks them. On a
match they write as put and delete do. On a mismatch they write nothing and
return the current memory, so the application decides which update wins.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from psycopg import Connection

from langchain_cockroachdb._sql import parse_vector
from langchain_cockroachdb.store import ConditionalWrite
from tests.integration.faults import FailAfterCommit
from tests.integration.store.conftest import SEATS, seat_index
from tests.integration.store.test_reconcile import (
    AISLE,
    LONG_FLIGHTS,
    NS,
    OLD,
    PREFIX,
    RETRY_LOGGER,
    VEGETARIAN,
    WINDOW,
    history_rows,
    item_rows,
    retries,
    run_together,
    stored_valid_at,
    text,
    vector_keys,
)
from tests.integration.store.test_schema_upgrade import schema_0_5

HISTORY = pytest.mark.parametrize("history", [False, True], ids=["no-history", "history"])


def snapshot(admin: Connection[Any], store: Any) -> tuple[Any, ...]:
    """Every item and vector under PREFIX, and its history rows on a store with history."""
    archived = history_rows(admin, store) if store._history is not None else []
    return (item_rows(admin, store), vector_keys(admin, store), archived)


def stored_vector(admin: Connection[Any], store: Any, key: str) -> list[float]:
    row = admin.execute(
        f"SELECT embedding FROM {store.sql.vectors} WHERE prefix = %s AND key = %s",
        (PREFIX, key),
    ).fetchone()
    assert row is not None
    return parse_vector(row["embedding"])


@HISTORY
def test_put_if_applies_when_unchanged(store_factory, admin, mode, history) -> None:
    store = store_factory(mode, history=history, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    assert record is not None and record.version == 1

    result = store.put_if(NS, "k", text(AISLE), expected=record, provenance={"source": "chat"})

    assert result == ConditionalWrite(applied=True, key="k", version=2, current=None)
    now = store.record(NS, "k")
    assert now is not None
    assert (now.version, now.item.value, now.provenance) == (2, text(AISLE), {"source": "chat"})
    # The write id is on the row, and the vectors are the new value's
    row = admin.execute(
        f"SELECT write_id FROM {store.sql.items} WHERE prefix = %s AND key = 'k'", (PREFIX,)
    ).fetchone()
    assert row is not None and row["write_id"] is not None
    assert vector_keys(admin, store) == ["k"]
    assert stored_vector(admin, store, "k") == pytest.approx(SEATS[AISLE], abs=1e-6)
    if history:
        [old] = store.history(NS, "k")
        assert (old.version, old.reason, old.value) == (1, "overwritten", text(WINDOW))
        # Archived in the transaction that wrote the new version
        assert old.ended_at == now.item.updated_at


def test_put_if_stores_provenance_and_valid_time_as_reconcile_does(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())

    created = store.put_if(NS, "k", text(WINDOW), expected=None, valid_at=OLD)
    first = store.record(NS, "k")
    assert created.applied and first is not None
    assert (first.valid_at, first.provenance) == (OLD, None)
    later = OLD + timedelta(days=1)
    written = store.put_if(
        NS, "k", text(AISLE), expected=first, provenance={"source": "chat"}, valid_at=later
    )

    assert written == ConditionalWrite(applied=True, key="k", version=2, current=None)
    # The version it replaced ends at the new fact's valid time
    [old] = store.history(NS, "k")
    assert (old.valid_at, old.invalid_at) == (OLD, later)
    # Without a valid time the memory stores none: it is valid from its write
    plain = store.put_if(NS, "k", text(LONG_FLIGHTS), expected=store.record(NS, "k"))
    assert plain.applied and stored_valid_at(admin, store, "k") is None


@HISTORY
def test_put_if_returns_the_current_record_on_a_conflict(
    store_factory, admin, mode, history
) -> None:
    store = store_factory(mode, history=history, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    # Another writer puts between the record and the put_if
    store.put(NS, "k", text(LONG_FLIGHTS))
    before = snapshot(admin, store)

    result = store.put_if(NS, "k", text(AISLE), expected=record)

    current = store.record(NS, "k")
    assert current is not None and (current.version, current.item.value) == (2, text(LONG_FLIGHTS))
    assert result == ConditionalWrite(applied=False, key="k", version=None, current=current)
    # Nothing was written
    assert snapshot(admin, store) == before
    # The application merges, and writes with the current record
    merged = store.put_if(NS, "k", text(AISLE), expected=result.current)
    assert merged == ConditionalWrite(applied=True, key="k", version=3, current=None)


@HISTORY
def test_put_if_with_none_creates_once(store_factory, admin, mode, history) -> None:
    store = store_factory(mode, history=history, index=seat_index())

    first = store.put_if(NS, "k", text(WINDOW), expected=None)
    second = store.put_if(NS, "k", text(AISLE), expected=None)

    assert first == ConditionalWrite(applied=True, key="k", version=1, current=None)
    current = store.record(NS, "k")
    assert current is not None and current.item.value == text(WINDOW)
    assert second == ConditionalWrite(applied=False, key="k", version=None, current=current)
    assert [row["key"] for row in item_rows(admin, store)] == ["k"]
    assert stored_vector(admin, store, "k") == pytest.approx(SEATS[WINDOW], abs=1e-6)

    # A deleted memory counts as absent. With history the key's versions go on after its history
    store.delete(NS, "k")
    again = store.put_if(NS, "k", text(AISLE), expected=None)
    assert again == ConditionalWrite(
        applied=True, key="k", version=2 if history else 1, current=None
    )
    # So does an expired one, which is not archived
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() - INTERVAL '1 minute' "
        "WHERE prefix = %s AND key = 'k'",
        (PREFIX,),
    )
    revived = store.put_if(NS, "k", text(VEGETARIAN), expected=None)
    assert revived == ConditionalWrite(
        applied=True, key="k", version=3 if history else 2, current=None
    )
    if history:
        assert [(v.version, v.reason) for v in store.history(NS, "k")] == [(1, "deleted")]


@HISTORY
def test_put_if_detects_a_0_5_style_write_through_updated_at(
    store_factory, admin, mode, history
) -> None:
    store = store_factory(mode, history=history, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    # A 0.5.x store writes with the 0.5 statement during a rolling upgrade: it
    # sets updated_at and leaves the version as it was
    admin.execute(*store.sql.upsert_items([(PREFIX, "k", json.dumps(text(LONG_FLIGHTS)), None)]))
    current = store.record(NS, "k")
    assert record is not None and current is not None
    assert (current.version, current.item.value) == (1, text(LONG_FLIGHTS))
    before = snapshot(admin, store)

    result = store.put_if(NS, "k", text(AISLE), expected=record)

    assert result == ConditionalWrite(applied=False, key="k", version=None, current=current)
    assert snapshot(admin, store) == before


@HISTORY
def test_two_threads_put_if_from_the_same_record_one_wins(
    store_factory, admin, caplog, monkeypatch, history
) -> None:
    # Both read the memory in their transactions before either writes, so both
    # find it unchanged. Serializable isolation then retries one of them, and
    # its check sees the other's write
    store = store_factory("pool", history=history, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    both_read = threading.Barrier(2, timeout=10)
    check = store._check_conditional
    lock = threading.Lock()
    checked: set[int] = set()

    def check_once_both_read(plan: Any, found: Any, **kwargs: Any) -> Any:
        # The first check of each thread waits for the other's. A retry checks at once
        with lock:
            first = threading.get_ident() not in checked
            checked.add(threading.get_ident())
        if first:
            both_read.wait()
        return check(plan, found, **kwargs)

    monkeypatch.setattr(store, "_check_conditional", check_once_both_read)
    values = [text(AISLE), text(LONG_FLIGHTS)]

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        results = run_together(
            *(lambda value=value: store.put_if(NS, "k", value, expected=record) for value in values)
        )

    assert retries(caplog, "store put_if")
    assert sorted(result.applied for result in results) == [False, True]
    won = next(i for i, result in enumerate(results) if result.applied)
    assert results[won] == ConditionalWrite(applied=True, key="k", version=2, current=None)
    # The other gets the winner's memory, to merge or to reject
    current = store.record(NS, "k")
    assert current is not None and current.item.value == values[won]
    assert results[1 - won] == ConditionalWrite(
        applied=False, key="k", version=None, current=current
    )
    assert stored_vector(admin, store, "k") == pytest.approx(SEATS[values[won]["text"]], abs=1e-6)
    if history:
        assert [(v.version, v.reason, v.value) for v in store.history(NS, "k")] == [
            (1, "overwritten", text(WINDOW))
        ]


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
@HISTORY
def test_put_if_retried_after_a_lost_commit_is_already_applied(
    store_factory, admin, unsafe_internals, caplog, mode, history
) -> None:
    # The put_if commits, then the store sees 40003 and runs it again. The
    # retry finds its write id on the row and writes nothing
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=history,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = store.put_if(NS, "k", text(AISLE), expected=record)

    assert fault.fired == 1
    found = retries(caplog, "store put_if")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    assert result == ConditionalWrite(
        applied=True, key="k", version=2, current=None, already_applied=True
    )
    # One version bump and, with history, one archived version
    now = store.record(NS, "k")
    assert now is not None and (now.version, now.item.value) == (2, text(AISLE))
    if history:
        assert [(v.version, v.reason) for v in store.history(NS, "k")] == [(1, "overwritten")]
    assert vector_keys(admin, store) == ["k"]

    # Create only, the same way: the retry finds its own memory, not a conflict
    fault.armed = True
    created = store.put_if(NS, "new", text(VEGETARIAN), expected=None)
    assert fault.fired == 2
    assert created == ConditionalWrite(
        applied=True, key="new", version=1, current=None, already_applied=True
    )


@HISTORY
def test_delete_if_applies_and_conflicts(store_factory, admin, mode, history) -> None:
    store = store_factory(mode, history=history, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    stale = store.record(NS, "k")
    store.put(NS, "k", text(AISLE))
    before = snapshot(admin, store)

    # Written since the record was read: nothing is deleted, and the current memory comes back
    conflict = store.delete_if(NS, "k", expected=stale)
    current = store.record(NS, "k")
    assert current is not None and current.version == 2
    assert conflict == ConditionalWrite(applied=False, key="k", version=None, current=current)
    assert snapshot(admin, store) == before

    # With the current record it deletes as delete does
    deleted = store.delete_if(NS, "k", expected=current)
    assert deleted == ConditionalWrite(applied=True, key="k", version=None, current=None)
    assert store.get(NS, "k") is None and vector_keys(admin, store) == []
    if history:
        assert [(v.version, v.reason) for v in store.history(NS, "k")] == [
            (2, "deleted"),
            (1, "overwritten"),
        ]

    # Gone since: a conflict with no current memory
    gone = store.delete_if(NS, "k", expected=current)
    assert gone == ConditionalWrite(applied=False, key="k", version=None, current=None)
    # With expected=None it applies only while no live memory has the key
    absent = store.delete_if(NS, "k", expected=None)
    assert absent == ConditionalWrite(applied=True, key="k", version=None, current=None)
    store.put(NS, "k", text(VEGETARIAN))
    live = store.delete_if(NS, "k", expected=None)
    assert live == ConditionalWrite(
        applied=False, key="k", version=None, current=store.record(NS, "k")
    )
    assert store.get(NS, "k") is not None


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
@HISTORY
def test_delete_if_retried_after_a_lost_commit_is_applied(
    store_factory, admin, unsafe_internals, caplog, mode, history
) -> None:
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=history,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    fault.armed = True

    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        result = store.delete_if(NS, "k", expected=record)

    assert fault.fired == 1
    found = retries(caplog, "store delete_if")
    assert len(found) == 1 and "SQLSTATE 40003" in found[0]
    # The retry finds the memory gone, the state delete_if asked for. With
    # history its own history row proves that the earlier attempt deleted it.
    # Without history no row proves which attempt did
    assert result == ConditionalWrite(
        applied=True, key="k", version=None, current=None, already_applied=history
    )
    assert store.get(NS, "k") is None and vector_keys(admin, store) == []
    if history:
        assert [(v.version, v.reason) for v in store.history(NS, "k")] == [(1, "deleted")]


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
@pytest.mark.parametrize("method", ["put_if", "delete_if"])
def test_a_retry_after_another_writer_reports_the_current_memory(
    store_factory, unsafe_internals, monkeypatch, mode, method
) -> None:
    # The write commits and its result is lost. Before the retry another
    # writer writes the key, so the retry reports a conflict with that
    # writer's memory: the current state
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    other = store_factory("pool", history=True, index=seat_index())
    store.put(NS, "k", text(WINDOW))
    record = store.record(NS, "k")
    real = store._cursor
    attempts = 0

    @contextlib.contextmanager
    def cursor(**kwargs: Any) -> Iterator[Any]:
        nonlocal attempts
        if kwargs.get("label") == f"store {method}":
            attempts += 1
            if attempts == 2:
                other.put(NS, "k", text(LONG_FLIGHTS))
        with real(**kwargs) as cur:
            yield cur

    monkeypatch.setattr(store, "_cursor", cursor)
    fault.armed = True

    if method == "put_if":
        result = store.put_if(NS, "k", text(AISLE), expected=record)
    else:
        result = store.delete_if(NS, "k", expected=record)

    assert fault.fired == 1 and attempts == 2
    current = store.record(NS, "k")
    assert current is not None and current.item.value == text(LONG_FLIGHTS)
    assert result == ConditionalWrite(applied=False, key="k", version=None, current=current)


@HISTORY
def test_put_if_needs_setup(store_factory, admin, mode, history) -> None:
    index = seat_index()
    store = store_factory(mode, setup=False, history=history, index=index)
    for statement in schema_0_5(store):
        admin.execute(statement)

    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.put_if(NS, "k", text(AISLE), expected=None)
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.delete_if(NS, "k", expected=None)
    assert index["embed"].calls == 0
    row = admin.execute(f"SELECT count(*) AS n FROM {store.sql.items}").fetchone()
    assert row is not None and row["n"] == 0

    store.setup()
    assert store.put_if(NS, "k", text(AISLE), expected=None).applied
    assert store.delete_if(NS, "k", expected=store.record(NS, "k")).applied
