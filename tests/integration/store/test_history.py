"""Version history: what a store with history keeps when it overwrites or deletes.

With history on, setup() creates the history table, and every put batch
first copies the live versions it overwrites or deletes there, in the same
transaction. delete becomes a soft delete; purge() and delete_namespace()
remove history too. A batch retried after its commit archives nothing twice.
A version never ends before it was written, and no version of a key starts
before the one it follows ended, across an overwrite, a delete or a late
fact, even for a transaction whose now() is behind them.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import pytest
from langgraph.store.base import PutOp
from psycopg import Connection

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore
from langchain_cockroachdb.store import (
    ConditionalWrite,
    Decision,
    MemoryRecord,
    MemoryVersion,
    keep_all,
)
from tests.integration.faults import FailAfterCommit
from tests.integration.store.conftest import seat_index

RETRY_LOGGER = "langchain_cockroachdb.retry"
NS = ("user", "alice")
PREFIX = "user.alice"
WINDOW = {"text": "Prefers window seats"}
AISLE = {"text": "Prefers aisle seats"}
VEGETARIAN = {"text": "Vegetarian"}
PEANUTS = {"text": "Allergic to peanuts"}


def count(admin: Connection[Any], table: str, prefix: str, key: str | None = None) -> int:
    """Rows of ``table`` under exactly ``prefix``, or for one key there."""
    if key is None:
        row = admin.execute(
            f"SELECT count(*) AS n FROM {table} WHERE prefix = %s", (prefix,)
        ).fetchone()
    else:
        row = admin.execute(
            f"SELECT count(*) AS n FROM {table} WHERE prefix = %s AND key = %s", (prefix, key)
        ).fetchone()
    assert row is not None
    return row["n"]


def item_version(admin: Connection[Any], store: Any, key: str) -> int:
    row = admin.execute(
        f"SELECT version FROM {store.sql.items} WHERE prefix = %s AND key = %s", (PREFIX, key)
    ).fetchone()
    assert row is not None
    return row["version"]


def history_rows(admin: Connection[Any], store: Any, key: str) -> list[dict[str, Any]]:
    return admin.execute(
        f"SELECT version, reason, value, write_id FROM {store.sql.history} "
        "WHERE prefix = %s AND key = %s ORDER BY version DESC",
        (PREFIX, key),
    ).fetchall()


def retries(caplog: pytest.LogCaptureFixture, label: str) -> list[str]:
    prefix = f"Retrying {label} after "
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


WRITERS = pytest.mark.parametrize("writer", ["put", "reconcile", "put_if"])


def move_ahead(store: Any, key: str) -> tuple[str, tuple[str, str]]:
    """The statement that moves a live memory's updated_at and valid_at an hour ahead.

    CockroachDB fixes now() when a transaction starts. A transaction that is
    pushed past a later writer, and refreshes its reads, can end that
    writer's version with a now() earlier than the version's updated_at.
    A version an hour ahead of the clock stands in for it. A memory with no
    valid time keeps none.
    """
    return (
        f"UPDATE {store.sql.items} SET updated_at = updated_at + INTERVAL '1 hour', "
        "valid_at = valid_at + INTERVAL '1 hour' WHERE prefix = %s AND key = %s",
        (PREFIX, key),
    )


def check_never_back_in_time(
    old: MemoryVersion, before: MemoryRecord, new: MemoryRecord | None
) -> None:
    """Check the history row and live memory a write left after replacing ``before``."""
    assert (old.version, old.value, old.reason) == (1, WINDOW, "overwritten")
    assert old.written_at == before.item.updated_at
    assert old.ended_at >= old.written_at
    assert old.invalid_at is not None and old.invalid_at >= old.valid_at
    assert new is not None and (new.version, new.item.value) == (2, AISLE)
    assert new.item.updated_at >= before.item.updated_at
    # The archive and the write ran in one transaction, and the old memory
    # had no valid time of its own, so its fact ended when the store replaced it
    assert old.ended_at == new.item.updated_at == old.invalid_at


def write_as(writer: str, store: CockroachDBStore, value: dict[str, Any]) -> None:
    """Write ``value`` under "k" as ``writer`` does: put, reconcile (keep_all) or put_if."""
    if writer == "put":
        store.put(NS, "k", value)
    elif writer == "reconcile":
        assert store.reconcile(NS, "k", value, policy=keep_all()).new == "written"
    else:
        assert store.put_if(NS, "k", value, expected=store.record(NS, "k")).applied


async def awrite_as(writer: str, store: AsyncCockroachDBStore, value: dict[str, Any]) -> None:
    """Write ``value`` under "k" as ``writer`` does, on the async store."""
    if writer == "put":
        await store.aput(NS, "k", value)
    elif writer == "reconcile":
        assert (await store.areconcile(NS, "k", value, policy=keep_all())).new == "written"
    else:
        expected = await store.arecord(NS, "k")
        assert (await store.aput_if(NS, "k", value, expected=expected)).applied


RECREATORS = pytest.mark.parametrize("writer", ["put", "reconcile"])
OLD = datetime(2025, 1, 1, tzinfo=timezone.utc)


def move_history_ahead(store: Any, key: str, version: int) -> tuple[str, tuple[str, str, int]]:
    """The statement that moves one history row's ended_at an hour ahead, returning it."""
    return (
        f"UPDATE {store.sql.history} SET ended_at = ended_at + INTERVAL '1 hour' "
        "WHERE prefix = %s AND key = %s AND version = %s RETURNING ended_at",
        (PREFIX, key, version),
    )


def superseded_by_newer(proposal: Any, candidates: Any) -> Decision:
    """A policy that records the new memory as a late fact that "newer" supersedes."""
    return Decision(new="superseded", superseded_by="newer")


def test_put_over_an_existing_key_keeps_the_old_version(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)

    [old] = store.history(NS, "k")
    assert isinstance(old, MemoryVersion)
    assert (old.namespace, old.key, old.version) == (NS, "k", 1)
    assert old.value == WINDOW
    assert old.reason == "overwritten"
    assert old.superseded_by is None and old.provenance is None
    assert old.written_at <= old.ended_at
    # A plain put has no valid time: the version was valid from when it was
    # written, until the new value was written
    assert old.valid_at == old.written_at
    assert old.invalid_at == old.ended_at
    item = store.get(NS, "k")
    assert item is not None and item.value == AISLE
    # The archive and the overwrite ran in one transaction
    assert old.ended_at == item.updated_at
    assert item_version(admin, store, "k") == 2


def test_delete_is_soft_with_history(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.delete(NS, "k")

    assert store.get(NS, "k") is None
    assert store.search(NS, query="Prefers window seats") == []
    [old] = store.history(NS, "k")
    assert (old.version, old.reason, old.value) == (1, "deleted", WINDOW)
    assert old.invalid_at == old.ended_at
    # The item and its vectors are gone; only history keeps the memory
    assert count(admin, store.sql.items, PREFIX, "k") == 0
    assert count(admin, store.sql.vectors, PREFIX, "k") == 0


def test_delete_is_hard_without_history(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.delete(NS, "k")

    assert count(admin, store.sql.items, PREFIX, "k") == 0
    with pytest.raises(ValueError, match="history=True"):
        store.history(NS, "k")
    with pytest.raises(ValueError, match="history=True"):
        store.purge(NS, "k")
    # setup() made no history table
    row = admin.execute(*store.sql.history_exists()).fetchone()
    assert row is not None and row["ready"] is False


def test_purge_removes_the_memory_its_vectors_and_history(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)
    store.put(NS, "other", VEGETARIAN)
    assert count(admin, store.sql.vectors, PREFIX, "k") == 1
    assert count(admin, store.sql.history, PREFIX, "k") == 1

    assert store.purge(NS, "k") is True
    for table in (store.sql.items, store.sql.vectors, store.sql.history):
        assert count(admin, table, PREFIX, "k") == 0
    assert store.purge(NS, "k") is False
    # Other memories in the namespace stay
    assert store.get(NS, "other") is not None

    # A deleted memory has only its history left, which purge removes
    store.delete(NS, "other")
    assert count(admin, store.sql.history, PREFIX, "other") == 1
    assert store.purge(NS, "other") is True
    assert count(admin, store.sql.history, PREFIX, "other") == 0


def test_history_on_a_store_without_an_index(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True)
    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)
    store.delete(NS, "k")
    assert [(v.version, v.reason) for v in store.history(NS, "k")] == [
        (2, "deleted"),
        (1, "overwritten"),
    ]
    assert store.purge(NS, "k") is True
    assert count(admin, store.sql.history, PREFIX, "k") == 0
    assert store.purge(NS, "k") is False


def test_delete_namespace_removes_history_under_the_prefix(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    for ns in [("user", "a"), ("user", "a", "b"), ("user", "ab")]:
        store.put(ns, "k", WINDOW)
        store.put(ns, "k", AISLE)
        store.put(ns, "gone", VEGETARIAN)
        store.delete(ns, "gone")

    assert store.delete_namespace(("user", "a")) == 2
    assert count(admin, store.sql.history, "user.a") == 0
    assert count(admin, store.sql.history, "user.a.b") == 0
    assert count(admin, store.sql.history, "user.ab") == 2
    assert [v.reason for v in store.history(("user", "ab"), "k")] == ["overwritten"]
    assert store.get(("user", "ab"), "k") is not None


def test_a_store_without_history_deletes_the_history_left_under_a_namespace(
    store_factory, admin, mode
) -> None:
    keeper = store_factory(mode, history=True, index=seat_index())
    for ns in [("user", "a"), ("user", "ab")]:
        keeper.put(ns, "k", WINDOW)
        keeper.put(ns, "k", AISLE)

    # Later the application turns history off, on the same tables. Forgetting
    # a namespace still forgets its history
    store = store_factory(mode, index=seat_index())
    assert store.delete_namespace(("user", "a")) == 1
    assert count(admin, store.sql.history, "user.a") == 0
    assert count(admin, store.sql.history, "user.ab") == 1
    assert store.get(("user", "ab"), "k") is not None
    # purge still needs a store with history
    with pytest.raises(ValueError, match="history=True"):
        store.purge(("user", "ab"), "k")


def test_history_lists_versions_newest_first_up_to_limit(store_factory, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    values = [WINDOW, AISLE, VEGETARIAN, PEANUTS]
    for value in values:
        store.put(NS, "k", value)

    versions = store.history(NS, "k", limit=2)
    assert [v.version for v in versions] == [3, 2]
    assert [v.value for v in versions] == [VEGETARIAN, AISLE]
    assert [v.version for v in store.history(NS, "k")] == [3, 2, 1]
    assert store.history(NS, "k", limit=0) == []
    assert store.history(NS, "missing") == []
    with pytest.raises(ValueError, match="limit"):
        store.history(NS, "k", limit=-1)


def test_history_ttl_sets_row_level_ttl_on_the_history_table(store_factory, admin, mode) -> None:
    store = store_factory(mode, history={"ttl_days": 30}, index=seat_index())
    row = admin.execute(
        f"SELECT create_statement FROM [SHOW CREATE TABLE {store.sql.history}]"
    ).fetchone()
    assert row is not None
    assert "ttl_expiration_expression = 'expires_at'" in row["create_statement"]

    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)
    row = admin.execute(
        f"SELECT extract(epoch FROM expires_at - now()) AS left_seconds FROM {store.sql.history} "
        "WHERE prefix = %s AND key = %s",
        (PREFIX, "k"),
    ).fetchone()
    assert row is not None
    assert 30 * 86400 - 600 < float(row["left_seconds"]) <= 30 * 86400


def test_history_leaves_out_versions_past_their_ttl(store_factory, admin, mode) -> None:
    store = store_factory(mode, history={"ttl_days": 30}, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)
    store.put(NS, "k", VEGETARIAN)
    admin.execute(
        f"UPDATE {store.sql.history} SET expires_at = now() - INTERVAL '1 hour' "
        "WHERE prefix = %s AND key = %s AND version = 1",
        (PREFIX, "k"),
    )

    assert [v.version for v in store.history(NS, "k")] == [2]
    # The expired version is still in the table after history() ran: it was
    # left out, not yet deleted by the TTL job
    assert count(admin, store.sql.history, PREFIX, "k") == 2


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_a_put_retried_after_a_lost_commit_archives_once(
    store_factory, admin, unsafe_internals, caplog, mode: str
) -> None:
    # The first attempt commits, then the store sees 40003 and runs the batch
    # again. Its archive skips the row that carries the batch's write id
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode,
        history=True,
        index=seat_index(),
        connect_kwargs={"cursor_factory": fault.cursor_class()},
    )
    store.put(NS, "k", WINDOW)
    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        store.put(NS, "k", AISLE)
    assert fault.fired == 1
    assert len(retries(caplog, "store put")) == 1
    [old] = store.history(NS, "k")
    assert (old.version, old.value) == (1, WINDOW)
    assert item_version(admin, store, "k") == 2

    # A delete retried the same way finds the row gone and archives nothing more
    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        store.delete(NS, "k")
    assert fault.fired == 2
    assert [(v.version, v.reason) for v in store.history(NS, "k")] == [
        (2, "deleted"),
        (1, "overwritten"),
    ]


def test_history_needs_setup(store_factory, mode) -> None:
    store_factory(mode, index=seat_index())  # tables set up without history
    store = store_factory(mode, setup=False, history=True, index=seat_index())

    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.put(NS, "k", WINDOW)
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.history(NS, "k")
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.purge(NS, "k")
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.delete_namespace(NS)
    # Reads do not need history
    assert store.get(NS, "k") is None

    store.setup()
    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)
    assert [v.value for v in store.history(NS, "k")] == [WINDOW]


def test_a_store_sees_a_history_table_another_process_created(
    store_factory, mode, monkeypatch
) -> None:
    store_factory(mode, index=seat_index())
    store = store_factory(mode, setup=False, history=True, index=seat_index())
    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.put(NS, "k", WINDOW)

    # Another process runs setup() with history on
    store_factory(mode, history=True, index=seat_index())

    # The store trusts a missing table for a while, but reads the schema
    # again before it refuses a write
    monkeypatch.setattr(store, "_SCHEMA_RECHECK_SECONDS", 3600.0)
    store.put(NS, "k", WINDOW)
    store.put(NS, "k", AISLE)
    assert [v.value for v in store.history(NS, "k")] == [WINDOW]


def test_expired_rows_are_not_archived_when_overwritten(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW, ttl=60)
    store.put(NS, "gone", AISLE, ttl=60)
    admin.execute(
        f"UPDATE {store.sql.items} SET expires_at = now() - INTERVAL '1 minute' WHERE prefix = %s",
        (PREFIX,),
    )

    store.put(NS, "k", AISLE)
    store.delete(NS, "gone")
    assert store.history(NS, "k") == []
    assert store.history(NS, "gone") == []
    item = store.get(NS, "k")
    assert item is not None and item.value == AISLE


def test_a_key_written_again_after_a_delete_continues_its_versions(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.delete(NS, "k")
    store.put(NS, "k", AISLE)
    # Version 1 is in history already, so the new memory is version 2
    assert item_version(admin, store, "k") == 2

    store.put(NS, "k", VEGETARIAN)
    assert [(v.version, v.reason, v.value) for v in store.history(NS, "k")] == [
        (2, "overwritten", AISLE),
        (1, "deleted", WINDOW),
    ]


def test_a_batch_archives_every_key_it_overwrites_or_deletes(store_factory, admin, mode) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    bob = ("user", "bob")
    store.batch([PutOp(NS, "a", WINDOW), PutOp(NS, "b", AISLE), PutOp(bob, "c", VEGETARIAN)])
    store.batch(
        [
            PutOp(NS, "a", AISLE),
            PutOp(NS, "b", None),
            PutOp(bob, "c", PEANUTS),
            PutOp(NS, "new", WINDOW),
        ]
    )

    assert [(v.reason, v.value) for v in store.history(NS, "a")] == [("overwritten", WINDOW)]
    assert [(v.reason, v.value) for v in store.history(NS, "b")] == [("deleted", AISLE)]
    assert [(v.reason, v.value) for v in store.history(bob, "c")] == [("overwritten", VEGETARIAN)]
    assert store.history(NS, "new") == []
    # Every history row carries the write id of the batch that ended its version
    [a], [b] = history_rows(admin, store, "a"), history_rows(admin, store, "b")
    assert a["write_id"] is not None and a["write_id"] == b["write_id"]


@WRITERS
def test_a_write_whose_clock_is_behind_never_goes_back_in_time(
    store_factory, admin, mode, writer
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW)
    assert admin.execute(*move_ahead(store, "k")).rowcount == 1
    before = store.record(NS, "k")
    assert before is not None

    write_as(writer, store, AISLE)

    [old] = store.history(NS, "k")
    new = store.record(NS, "k")
    check_never_back_in_time(old, before, new)
    # The version moved, so a put_if from the record read before the write is
    # refused, even where updated_at stayed where it was
    stale = store.put_if(NS, "k", VEGETARIAN, expected=before)
    assert stale == ConditionalWrite(applied=False, key="k", version=None, current=new)


@RECREATORS
def test_a_key_written_again_after_a_delete_never_starts_before_it_ended(
    store_factory, admin, mode, writer
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", WINDOW)
    store.delete(NS, "k")
    row = admin.execute(*move_history_ahead(store, "k", 1)).fetchone()
    assert row is not None

    write_as(writer, store, AISLE)

    new = store.record(NS, "k")
    assert new is not None and (new.version, new.item.value) == (2, AISLE)
    assert new.item.updated_at >= row["ended_at"]
    # A new row is still never updated after it was created
    assert new.item.created_at == new.item.updated_at


def test_a_late_fact_is_never_written_before_the_memory_live_under_its_key(
    store_factory, admin, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", VEGETARIAN)
    store.put(NS, "newer", WINDOW)
    assert admin.execute(*move_ahead(store, "k")).rowcount == 1
    live = store.record(NS, "k")
    assert live is not None

    late = store.reconcile(NS, "k", AISLE, policy=superseded_by_newer, valid_at=OLD)

    assert late.new == "superseded"
    [row] = store.history(NS, "k")
    assert (row.version, row.reason, row.superseded_by) == (2, "superseded", "newer")
    assert row.written_at >= live.item.updated_at
    assert row.ended_at == row.written_at


@WRITERS
def test_a_version_written_after_a_late_fact_never_starts_before_it_ended(
    store_factory, admin, mode, writer
) -> None:
    store = store_factory(mode, history=True, index=seat_index())
    store.put(NS, "k", VEGETARIAN)
    store.put(NS, "newer", WINDOW)
    late = store.reconcile(NS, "k", AISLE, policy=superseded_by_newer, valid_at=OLD)
    assert late.new == "superseded"
    row = admin.execute(*move_history_ahead(store, "k", 2)).fetchone()
    assert row is not None

    write_as(writer, store, PEANUTS)

    new = store.record(NS, "k")
    assert new is not None and (new.version, new.item.value) == (3, PEANUTS)
    assert new.item.updated_at >= row["ended_at"]


@pytest.mark.timeout(240)
def test_parallel_setup_with_history_converges(libpq_url, table_name, admin) -> None:
    # Four processes starting together create the history table and change
    # its TTL at once. On v25.4 the TTL changes fail each other until retried
    errors: list[BaseException] = []
    stores: list[CockroachDBStore] = []

    def run() -> None:
        try:
            with CockroachDBStore.from_conn_string(
                libpq_url, history={"ttl_days": 30}, table_name=table_name
            ) as store:
                store.setup()
                stores.append(store)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180)
    assert not any(thread.is_alive() for thread in threads), "a setup thread did not finish"
    assert errors == []
    row = admin.execute(
        f"SELECT create_statement FROM [SHOW CREATE TABLE {stores[0].sql.history}]"
    ).fetchone()
    assert row is not None
    assert row["create_statement"].count("ttl_expiration_expression = 'expires_at'") == 1


def test_from_conn_string_takes_history(libpq_url, table_name) -> None:
    with CockroachDBStore.from_conn_string(libpq_url, table_name=table_name, history=True) as store:
        store.setup()
        store.put(NS, "k", WINDOW)
        store.put(NS, "k", AISLE)
        assert [v.value for v in store.history(NS, "k")] == [WINDOW]


async def test_async_store_archives_overwrites_and_deletes(astore_factory, aadmin, mode) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "k", WINDOW)
    await store.aput(NS, "k", AISLE)
    await store.adelete(NS, "k")

    assert await store.aget(NS, "k") is None
    cur = await aadmin.execute(
        f"SELECT version, reason, value FROM {store.sql.history} "
        "WHERE prefix = %s AND key = %s ORDER BY version DESC",
        (PREFIX, "k"),
    )
    assert [(r["version"], r["reason"], r["value"]) for r in await cur.fetchall()] == [
        (2, "deleted", AISLE),
        (1, "overwritten", WINDOW),
    ]

    # delete_namespace removes the history under the prefix too
    await store.aput(("user", "bob"), "j", VEGETARIAN)
    assert await store.adelete_namespace(("user",)) == 1
    cur = await aadmin.execute(f"SELECT count(*) AS n FROM {store.sql.history}")
    row = await cur.fetchone()
    assert row is not None and row["n"] == 0


async def test_async_store_without_history_deletes_the_history_left_under_a_namespace(
    astore_factory, aadmin, mode
) -> None:
    keeper = await astore_factory(mode, history=True, index=seat_index())
    for ns in [("user", "a"), ("user", "ab")]:
        await keeper.aput(ns, "k", WINDOW)
        await keeper.aput(ns, "k", AISLE)

    store = await astore_factory(mode, index=seat_index())
    assert await store.adelete_namespace(("user", "a")) == 1
    cur = await aadmin.execute(
        f"SELECT prefix, count(*) AS n FROM {store.sql.history} GROUP BY prefix"
    )
    assert {row["prefix"]: row["n"] for row in await cur.fetchall()} == {"user.ab": 1}


async def test_async_store_history_needs_setup(astore_factory, aadmin, mode) -> None:
    await astore_factory(mode, index=seat_index())  # tables set up without history
    store = await astore_factory(mode, setup=False, history=True, index=seat_index())
    with pytest.raises(ValueError, match=r"setup\(\)"):
        await store.aput(NS, "k", WINDOW)
    with pytest.raises(ValueError, match=r"setup\(\)"):
        await store.adelete_namespace(NS)

    await store.setup()
    await store.aput(NS, "k", WINDOW)
    await store.aput(NS, "k", AISLE)
    cur = await aadmin.execute(
        f"SELECT version, value FROM {store.sql.history} WHERE prefix = %s AND key = %s",
        (PREFIX, "k"),
    )
    assert [(r["version"], r["value"]) for r in await cur.fetchall()] == [(1, WINDOW)]


async def test_async_from_conn_string_takes_history(libpq_url, table_name, aadmin) -> None:
    async with AsyncCockroachDBStore.from_conn_string(
        libpq_url, table_name=table_name, history={"ttl_days": 7}
    ) as store:
        await store.setup()
        await store.aput(NS, "k", WINDOW)
        await store.aput(NS, "k", AISLE)
        cur = await aadmin.execute(
            f"SELECT extract(epoch FROM expires_at - now()) AS left_seconds "
            f"FROM {store.sql.history} WHERE prefix = %s AND key = %s",
            (PREFIX, "k"),
        )
        [row] = await cur.fetchall()
        assert 7 * 86400 - 600 < float(row["left_seconds"]) <= 7 * 86400


@WRITERS
async def test_an_async_write_whose_clock_is_behind_never_goes_back_in_time(
    astore_factory, aadmin, mode, writer
) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "k", WINDOW)
    assert (await aadmin.execute(*move_ahead(store, "k"))).rowcount == 1
    before = await store.arecord(NS, "k")
    assert before is not None

    await awrite_as(writer, store, AISLE)

    [old] = await store.ahistory(NS, "k")
    new = await store.arecord(NS, "k")
    check_never_back_in_time(old, before, new)
    stale = await store.aput_if(NS, "k", VEGETARIAN, expected=before)
    assert stale == ConditionalWrite(applied=False, key="k", version=None, current=new)


@RECREATORS
async def test_an_async_key_written_again_after_a_delete_never_starts_before_it_ended(
    astore_factory, aadmin, mode, writer
) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "k", WINDOW)
    await store.adelete(NS, "k")
    row = await (await aadmin.execute(*move_history_ahead(store, "k", 1))).fetchone()
    assert row is not None

    await awrite_as(writer, store, AISLE)

    new = await store.arecord(NS, "k")
    assert new is not None and (new.version, new.item.value) == (2, AISLE)
    assert new.item.updated_at >= row["ended_at"]
    assert new.item.created_at == new.item.updated_at


async def test_an_async_late_fact_is_never_written_before_the_memory_live_under_its_key(
    astore_factory, aadmin, mode
) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "k", VEGETARIAN)
    await store.aput(NS, "newer", WINDOW)
    assert (await aadmin.execute(*move_ahead(store, "k"))).rowcount == 1
    live = await store.arecord(NS, "k")
    assert live is not None

    late = await store.areconcile(NS, "k", AISLE, policy=superseded_by_newer, valid_at=OLD)

    assert late.new == "superseded"
    [row] = await store.ahistory(NS, "k")
    assert (row.version, row.reason, row.superseded_by) == (2, "superseded", "newer")
    assert row.written_at >= live.item.updated_at
    assert row.ended_at == row.written_at


@WRITERS
async def test_an_async_version_written_after_a_late_fact_never_starts_before_it_ended(
    astore_factory, aadmin, mode, writer
) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())
    await store.aput(NS, "k", VEGETARIAN)
    await store.aput(NS, "newer", WINDOW)
    late = await store.areconcile(NS, "k", AISLE, policy=superseded_by_newer, valid_at=OLD)
    assert late.new == "superseded"
    row = await (await aadmin.execute(*move_history_ahead(store, "k", 2))).fetchone()
    assert row is not None

    await awrite_as(writer, store, PEANUTS)

    new = await store.arecord(NS, "k")
    assert new is not None and (new.version, new.item.value) == (3, PEANUTS)
    assert new.item.updated_at >= row["ended_at"]
