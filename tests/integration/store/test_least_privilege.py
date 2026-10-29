"""A store run by a role granted only the store's own tables.

An admin role runs setup(). The application's role then gets SELECT, INSERT,
UPDATE and DELETE on the items, vectors and history tables, and no grant on
the migrations table, as 0.5.x allowed. The schema check reads the
migrations table first. When that read is denied, the store learns that
migration 2 is applied from the items table's own columns, so this role
still writes with versions and history. A role that can read the migrations
table keeps reading it.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
import warnings
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg import Connection, sql

from tests.integration.store.conftest import seat_index

NS = ("user", "alice")
PREFIX = "user.alice"
WINDOW = {"text": "Prefers window seats"}
AISLE = {"text": "Prefers aisle seats"}  # 0.85 against WINDOW
LONG_FLIGHTS = {"text": "Prefers window seats on long flights"}
VEGETARIAN = {"text": "Vegetarian"}  # 0.0 against the others


@pytest.fixture
def app_role(admin: Connection[Any]) -> Iterator[str]:
    """A login role with CONNECT on the test database and no grant on any table.

    The role and all its grants are dropped after the test, even when it fails.
    """
    role = f"lcrdb_app_{uuid.uuid4().hex[:10]}"
    row = admin.execute("SELECT current_database() AS name").fetchone()
    assert row is not None
    admin.execute(f"CREATE ROLE {role} WITH LOGIN")
    try:
        admin.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(row["name"]), sql.Identifier(role)
            )
        )
        yield role
    finally:
        try:
            # Revokes the role's grants on the database and on every table
            admin.execute(f"DROP OWNED BY {role}")
            admin.execute(f"DROP ROLE {role}")
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"role cleanup failed: {error!r}", UserWarning, stacklevel=1)


def grant_store_tables(admin: Connection[Any], libpq_url: str, role: str, store: Any) -> None:
    """Grant ``role`` the store's items, vectors and history tables, and nothing else.

    Then check that the role cannot read the migrations table.
    """
    tables = ", ".join((store.sql.items, store.sql.vectors, store.sql.history))
    admin.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {tables} TO {role}")
    with (
        Connection.connect(libpq_url, user=role, autocommit=True) as conn,
        pytest.raises(psycopg.errors.InsufficientPrivilege),
    ):
        conn.execute(f"SELECT v FROM {store.sql.migrations_table}")


def test_a_role_granted_only_the_store_tables_uses_migration_2(
    app_role, store_factory, admin, libpq_url, mode
) -> None:
    owner = store_factory(mode, history=True, index=seat_index())  # root runs setup()
    grant_store_tables(admin, libpq_url, app_role, owner)
    store = store_factory(
        mode, setup=False, history=True, index=seat_index(), connect_kwargs={"user": app_role}
    )

    store.put(NS, "seat", WINDOW)
    # The migrations read was denied, and the check fell back to the columns
    assert store._migrations_denied is True
    store.put(NS, "seat", AISLE)
    store.put(NS, "food", VEGETARIAN)
    item = store.get(NS, "seat")
    assert item is not None and item.value == AISLE
    assert [result.key for result in store.search(NS, query="Prefers window seats")] == [
        "seat",
        "food",
    ]

    # The second put bumped the version and kept the first value in history
    record = store.record(NS, "seat")
    assert record is not None and record.version == 2
    [old] = store.history(NS, "seat")
    assert (old.version, old.value, old.reason) == (1, WINDOW, "overwritten")
    row = admin.execute(
        f"SELECT version, write_id FROM {store.sql.items} WHERE prefix = %s AND key = %s",
        (PREFIX, "seat"),
    ).fetchone()
    assert row is not None and row["version"] == 2 and row["write_id"] is not None

    conflicts = store.find_conflicts(NS, "seat", LONG_FLIGHTS)
    assert [(c.key, c.reason) for c in conflicts.candidates] == [("seat", "key")]
    assert conflicts.target_version == 2

    # With history, a delete keeps the version it ends
    store.delete(NS, "food")
    assert store.get(NS, "food") is None
    [gone] = store.history(NS, "food")
    assert (gone.version, gone.value, gone.reason) == (1, VEGETARIAN, "deleted")


async def test_async_role_granted_only_the_store_tables_uses_migration_2(
    app_role, astore_factory, admin, aadmin, libpq_url, mode
) -> None:
    owner = await astore_factory(mode, history=True, index=seat_index())  # root runs setup()
    await asyncio.to_thread(grant_store_tables, admin, libpq_url, app_role, owner)
    store = await astore_factory(
        mode, setup=False, history=True, index=seat_index(), connect_kwargs={"user": app_role}
    )

    await store.aput(NS, "seat", WINDOW)
    assert store._migrations_denied is True
    await store.aput(NS, "seat", AISLE)
    await store.aput(NS, "food", VEGETARIAN)
    item = await store.aget(NS, "seat")
    assert item is not None and item.value == AISLE
    results = await store.asearch(NS, query="Prefers window seats")
    assert [result.key for result in results] == ["seat", "food"]
    await store.adelete(NS, "food")
    assert await store.aget(NS, "food") is None

    # The writes bumped versions, stored write ids and kept history
    cur = await aadmin.execute(
        f"SELECT key, version, write_id FROM {store.sql.items} WHERE prefix = %s", (PREFIX,)
    )
    [row] = await cur.fetchall()
    assert (row["key"], row["version"]) == ("seat", 2) and row["write_id"] is not None
    cur = await aadmin.execute(
        f"SELECT key, version, reason, value FROM {store.sql.history} "
        "WHERE prefix = %s ORDER BY key",
        (PREFIX,),
    )
    assert [(r["key"], r["version"], r["reason"], r["value"]) for r in await cur.fetchall()] == [
        ("food", 1, "deleted", VEGETARIAN),
        ("seat", 1, "overwritten", WINDOW),
    ]


def schema_checks(store: Any, monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[str, list]]]:
    """Record the statements of every schema check round trip a sync or async store sends."""
    checks: list[list[tuple[str, list]]] = []
    real = store._read_all

    def record(statements: list[tuple[str, list]], label: str) -> None:
        if label == "store schema check":
            checks.append(statements)

    def read_all(statements: list[tuple[str, list]], label: str, session: Any = None) -> Any:
        record(statements, label)
        return real(statements, label, session)

    async def aread_all(statements: list[tuple[str, list]], label: str, session: Any = None) -> Any:
        record(statements, label)
        return await real(statements, label, session)

    is_async = inspect.iscoroutinefunction(real)
    monkeypatch.setattr(store, "_read_all", aread_all if is_async else read_all)
    return checks


def test_a_role_that_can_read_the_migrations_table_keeps_reading_it(
    store_factory, monkeypatch, mode
) -> None:
    store = store_factory(mode, history=True, index=seat_index())  # root
    checks = schema_checks(store, monkeypatch)
    store.put(NS, "seat", WINDOW)
    store.put(NS, "seat", AISLE)
    assert checks == [[store.sql.reconcile_migration_recorded(), store.sql.history_exists()]]
    assert store._migrations_denied is False
    [old] = store.history(NS, "seat")
    assert old.value == WINDOW


async def test_async_role_that_can_read_the_migrations_table_keeps_reading_it(
    astore_factory, monkeypatch, mode
) -> None:
    store = await astore_factory(mode, history=True, index=seat_index())  # root
    checks = schema_checks(store, monkeypatch)
    await store.aput(NS, "seat", WINDOW)
    await store.aput(NS, "seat", AISLE)
    assert checks == [[store.sql.reconcile_migration_recorded(), store.sql.history_exists()]]
    assert store._migrations_denied is False


def test_a_denied_migrations_read_in_a_callers_transaction_goes_to_the_caller(
    app_role, store_factory, admin, libpq_url
) -> None:
    owner = store_factory("connection", history=True, index=seat_index())
    grant_store_tables(admin, libpq_url, app_role, owner)
    store = store_factory(
        "connection",
        setup=False,
        history=True,
        index=seat_index(),
        connect_kwargs={"user": app_role},
    )
    # The denial aborts the caller's transaction, so it raises there
    with pytest.raises(psycopg.errors.InsufficientPrivilege), store.conn.transaction():
        store.put(NS, "seat", WINDOW)
    # The caller's retry reads the columns
    with store.conn.transaction():
        store.put(NS, "seat", WINDOW)
    record = store.record(NS, "seat")
    assert record is not None and record.version == 1


async def test_async_denied_migrations_read_in_a_callers_transaction_goes_to_the_caller(
    app_role, astore_factory, admin, libpq_url
) -> None:
    owner = await astore_factory("connection", history=True, index=seat_index())
    await asyncio.to_thread(grant_store_tables, admin, libpq_url, app_role, owner)
    store = await astore_factory(
        "connection",
        setup=False,
        history=True,
        index=seat_index(),
        connect_kwargs={"user": app_role},
    )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with store.conn.transaction():
            await store.aput(NS, "seat", WINDOW)
    async with store.conn.transaction():
        await store.aput(NS, "seat", WINDOW)
    item = await store.aget(NS, "seat")
    assert item is not None and item.value == WINDOW
