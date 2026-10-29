"""Migration 2 on a 0.5 table, and the write id every put batch carries.

A store reads whether the items table has migration 2's columns before its
first write, and trusts them for good once it has seen them. Until setup()
applies migration 2 the store writes with the 0.5 statements, so a table
that 0.5.x set up keeps working. It reads the columns again after
_SCHEMA_RECHECK_SECONDS, or at once for a method that needs migration 2, so
it sees a setup() that another process ran. After migration 2, every write
bumps the row's version and stores the batch's write id, and a batch retried
after its commit changes nothing.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from langgraph.store.base import PutOp
from psycopg import Connection

from tests.integration.faults import FailAfterCommit
from tests.integration.store.conftest import seat_index

RETRY_LOGGER = "langchain_cockroachdb.retry"
NS = ("user", "alice")

# Migration 2's columns: (data type, nullable) as information_schema reports them
RECONCILE_COLUMNS = {
    "version": ("bigint", "NO"),
    "provenance": ("jsonb", "YES"),
    "confirmed_at": ("timestamp with time zone", "YES"),
    "confirmations": ("bigint", "NO"),
    "write_id": ("uuid", "YES"),
    "valid_at": ("timestamp with time zone", "YES"),
}
COLUMNS_SQL = (
    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
    "WHERE table_schema = 'public' AND table_name = %s"
)


def schema_0_5(store: Any) -> list[str]:
    """The statements that leave the tables 0.5.x setup() created: migrations 0 and 1."""
    return [
        *store.sql.migrations()[:2],
        f"INSERT INTO {store.sql.migrations_table} (v) VALUES (0), (1)",
        *store.sql.vector_statements(),
    ]


def row_sql(store: Any) -> str:
    return (
        "SELECT version, write_id, provenance, valid_at, confirmations "
        f"FROM {store.sql.items} WHERE prefix = %s AND key = %s"
    )


def row_of(admin: Connection[Any], store: Any, key: str) -> dict[str, Any]:
    row = admin.execute(row_sql(store), (".".join(NS), key)).fetchone()
    assert row is not None
    return row


def columns_of(admin: Connection[Any], table_name: str) -> dict[str, tuple[str, str]]:
    rows = admin.execute(COLUMNS_SQL, (table_name,)).fetchall()
    return {row["column_name"]: (row["data_type"], row["is_nullable"]) for row in rows}


def latest_migration(admin: Connection[Any], store: Any) -> int:
    row = admin.execute(f"SELECT max(v) AS v FROM {store.sql.migrations_table}").fetchone()
    assert row is not None
    return row["v"]


def retries(caplog: pytest.LogCaptureFixture, label: str) -> list[str]:
    prefix = f"Retrying {label} after "
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def test_setup_upgrades_a_0_5_table_with_data(store_factory, admin, table_name, mode) -> None:
    store = store_factory(mode, setup=False, index=seat_index())
    for statement in schema_0_5(store):
        admin.execute(statement)
    admin.execute(
        *store.sql.upsert_items([("user.alice", "seat", '{"text": "Prefers window seats"}', None)])
    )
    assert not set(RECONCILE_COLUMNS) & set(columns_of(admin, table_name))

    store.setup()

    columns = columns_of(admin, table_name)
    assert {name: columns.get(name) for name in RECONCILE_COLUMNS} == RECONCILE_COLUMNS
    row = row_of(admin, store, "seat")
    assert row["version"] == 1
    assert row["confirmations"] == 0
    assert row["valid_at"] is None
    assert latest_migration(admin, store) == 2
    item = store.get(NS, "seat")
    assert item is not None and item.value == {"text": "Prefers window seats"}


def test_plain_writes_work_before_setup_applies_migration_2(
    store_factory, admin, table_name, mode
) -> None:
    store = store_factory(mode, setup=False, index=seat_index())
    for statement in schema_0_5(store):
        admin.execute(statement)

    store.put(NS, "seat", {"text": "Prefers window seats"})
    store.put(NS, "food", {"text": "Vegetarian"})
    store.put(NS, "seat", {"text": "Prefers aisle seats"})
    item = store.get(NS, "seat")
    assert item is not None and item.value == {"text": "Prefers aisle seats"}
    results = store.search(NS, query="Prefers window seats")
    assert [result.key for result in results] == ["seat", "food"]
    assert results[0].score == pytest.approx(0.85, abs=1e-3)
    store.delete(NS, "food")
    assert store.get(NS, "food") is None
    # The writes used the 0.5 statements: the table has no new columns
    assert not set(RECONCILE_COLUMNS) & set(columns_of(admin, table_name))

    with pytest.raises(ValueError, match=r"setup\(\)"):
        store.reconcile(
            NS,
            "seat",
            {"text": "Prefers window seats"},
            policy=lambda proposal, candidates: None,
        )


def test_setup_turns_on_versions_for_a_store_that_wrote_before_it(
    store_factory, admin, mode
) -> None:
    # The first put caches that migration 2 is missing; setup() clears that
    store = store_factory(mode, setup=False, index=seat_index())
    for statement in schema_0_5(store):
        admin.execute(statement)
    store.put(NS, "seat", {"text": "Prefers window seats"})

    store.setup()

    assert row_of(admin, store, "seat")["version"] == 1
    store.put(NS, "seat", {"text": "Prefers aisle seats"})
    row = row_of(admin, store, "seat")
    assert row["version"] == 2
    assert row["write_id"] is not None


def test_a_store_that_checked_before_setup_sees_migration_2_later(
    store_factory, admin, mode, monkeypatch
) -> None:
    # Two stores write to a 0.5 table, so each caches that migration 2 is missing
    store = store_factory(mode, setup=False, index=seat_index())
    for statement in schema_0_5(store):
        admin.execute(statement)
    other = store_factory(mode, setup=False, index=seat_index())
    store.put(NS, "seat", {"text": "Prefers window seats"})
    other.put(NS, "food", {"text": "Vegetarian"})

    # Then another process runs setup()
    store_factory(mode, index=seat_index())

    # A method that needs migration 2 reads the schema again when it is called,
    # and plain writes use what it read
    with pytest.raises(NotImplementedError):
        store.reconcile(
            NS,
            "seat",
            {"text": "Prefers aisle seats"},
            policy=lambda proposal, candidates: None,
        )
    store.put(NS, "seat", {"text": "Prefers aisle seats"})
    assert row_of(admin, store, "seat")["version"] == 2

    # A plain write trusts the missing migration for _SCHEMA_RECHECK_SECONDS,
    # then reads the schema again
    monkeypatch.setattr(other, "_SCHEMA_RECHECK_SECONDS", 3600.0)
    other.put(NS, "food", {"text": "Allergic to peanuts"})
    assert row_of(admin, other, "food")["version"] == 1
    monkeypatch.setattr(other, "_SCHEMA_RECHECK_SECONDS", 0.0)
    other.put(NS, "food", {"text": "Vegetarian"})
    assert row_of(admin, other, "food")["version"] == 2


def test_every_write_bumps_the_version(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.put(NS, "seat", {"text": "Prefers window seats"})
    first = row_of(admin, store, "seat")
    assert first["version"] == 1
    store.put(NS, "seat", {"text": "Prefers aisle seats"})
    second = row_of(admin, store, "seat")
    assert second["version"] == 2
    # Each put batch stores its own write id; a plain put has no provenance or valid time
    assert first["write_id"] is not None and second["write_id"] is not None
    assert first["write_id"] != second["write_id"]
    assert second["provenance"] is None and second["valid_at"] is None


def test_a_batch_stores_one_write_id_on_every_row(store_factory, admin, mode) -> None:
    store = store_factory(mode, index=seat_index())
    store.batch(
        [
            PutOp(NS, "seat", {"text": "Prefers window seats"}),
            PutOp(NS, "food", {"text": "Vegetarian"}),
        ]
    )
    seat, food = row_of(admin, store, "seat"), row_of(admin, store, "food")
    assert seat["write_id"] is not None
    assert seat["write_id"] == food["write_id"]
    assert seat["version"] == food["version"] == 1


@pytest.mark.parametrize("mode", ["pipeline", "pool"])
def test_a_put_retried_after_a_lost_commit_bumps_the_version_once(
    store_factory, admin, unsafe_internals, caplog, mode: str
) -> None:
    # The first attempt commits, then the store sees 40003 and runs the batch
    # again. Its upsert finds the row carrying its write id and changes nothing
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode, index=seat_index(), connect_kwargs={"cursor_factory": fault.cursor_class()}
    )
    store.put(NS, "seat", {"text": "Prefers window seats"})
    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        store.put(NS, "seat", {"text": "Prefers aisle seats"})
    assert fault.fired == 1
    found = retries(caplog, "store put")
    assert len(found) == 1
    assert "SQLSTATE 40003" in found[0]
    assert row_of(admin, store, "seat")["version"] == 2
    item = store.get(NS, "seat")
    assert item is not None and item.value == {"text": "Prefers aisle seats"}


async def test_async_store_writes_0_5_statements_until_setup(
    astore_factory, aadmin, table_name, mode
) -> None:
    store = await astore_factory(mode, setup=False, index=seat_index())
    for statement in schema_0_5(store):
        await aadmin.execute(statement)

    await store.aput(NS, "seat", {"text": "Prefers window seats"})
    await store.aput(NS, "seat", {"text": "Prefers aisle seats"})
    item = await store.aget(NS, "seat")
    assert item is not None and item.value == {"text": "Prefers aisle seats"}
    assert [result.key for result in await store.asearch(NS, query="Vegetarian")] == ["seat"]
    await store.aput(NS, "food", {"text": "Vegetarian"})
    await store.adelete(NS, "food")
    assert await store.aget(NS, "food") is None
    cur = await aadmin.execute(COLUMNS_SQL, (table_name,))
    assert not set(RECONCILE_COLUMNS) & {row["column_name"] for row in await cur.fetchall()}

    await store.setup()

    # Migration 2 gave the row version 1; the put after setup() bumps it
    await store.aput(NS, "seat", {"text": "Prefers window seats"})
    cur = await aadmin.execute(row_sql(store), (".".join(NS), "seat"))
    row = await cur.fetchone()
    assert row is not None
    assert row["version"] == 2
    assert row["write_id"] is not None
