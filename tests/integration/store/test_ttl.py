"""TTL on the store, checked against a real CockroachDB.

Puts write expires_at from the item's or the default TTL, reads push it
back when refresh_ttl resolves to true, and expired rows are hidden from
reads, searches and listings until something deletes them. Three things
delete them: CockroachDB's row-level TTL job, which setup() turns on for
both tables when the store has a TTL configuration, sweep_ttl(), and the
client-side sweeper. The slow test that waits for the TTL job has
"ttl_expiration" in its name, so -k "not ttl_expiration" skips it.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import pytest
from psycopg import AsyncConnection, Connection

from tests.integration.store.test_parity import WAIT
from tests.integration.store.test_semantic import TWO_MESSAGES, index_config

# 0.01 minutes is 0.6 seconds; sleeping EXPIRED seconds lets such a row expire
SHORT = 0.01
EXPIRED = 1.0
# Long enough that nothing written with it expires during a test
LONG = 60.0
SETTING = "ttl_expiration_expression = 'expires_at'"


def create_statement(admin: Connection[Any], table: str) -> str:
    row = admin.execute(f"SELECT create_statement FROM [SHOW CREATE TABLE {table}]").fetchone()
    assert row is not None
    return row["create_statement"]


async def acreate_statement(aadmin: AsyncConnection[Any], table: str) -> str:
    cur = await aadmin.execute(f"SELECT create_statement FROM [SHOW CREATE TABLE {table}]")
    row = await cur.fetchone()
    assert row is not None
    return row["create_statement"]


EXPIRY_SQL = "SELECT key, expires_at, ttl_minutes, {left} AS left_s FROM {table} ORDER BY key"
LEFT = "CAST(EXTRACT(EPOCH FROM expires_at - now()) AS FLOAT8)"


def expiries(admin: Connection[Any], table: str) -> dict[str, list[dict[str, Any]]]:
    """The expiry rows of ``table``, grouped by key."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in admin.execute(EXPIRY_SQL.format(left=LEFT, table=table)).fetchall():
        out.setdefault(row["key"], []).append(row)
    return out


async def aexpiries(aadmin: AsyncConnection[Any], table: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    cur = await aadmin.execute(EXPIRY_SQL.format(left=LEFT, table=table))
    for row in await cur.fetchall():
        out.setdefault(row["key"], []).append(row)
    return out


def expires_at(rows: dict[str, list[dict[str, Any]]], key: str) -> list[datetime]:
    return [row["expires_at"] for row in rows[key]]


def snapshot(admin: Connection[Any], store: Any) -> dict[str, dict[str, list[datetime]]]:
    """expires_at of every item and vector row, by table then key."""
    out = {}
    for name, table in (("items", store.sql.items), ("vectors", store.sql.vectors)):
        rows = expiries(admin, table)
        out[name] = {key: expires_at(rows, key) for key in rows}
    return out


async def asnapshot(
    aadmin: AsyncConnection[Any], store: Any
) -> dict[str, dict[str, list[datetime]]]:
    out = {}
    for name, table in (("items", store.sql.items), ("vectors", store.sql.vectors)):
        rows = await aexpiries(aadmin, table)
        out[name] = {key: expires_at(rows, key) for key in rows}
    return out


def assert_extended(before: dict, after: dict, keys: set[str], unchanged: set[str]) -> None:
    """Rows of ``keys`` moved later in both tables; rows of ``unchanged`` did not move."""
    for table in ("items", "vectors"):
        for key in keys:
            assert all(a > b for a, b in zip(after[table][key], before[table][key], strict=True)), (
                table,
                key,
            )
        for key in unchanged:
            assert after[table][key] == before[table][key], (table, key)


def assert_put_expiry(items: dict, vectors: dict) -> None:
    # No ttl argument: the default of 2 minutes. ttl=10: ten minutes.
    # ttl=None: no expiry. Vectors carry the same expiry as their item.
    for rows in (items, vectors):
        for row in rows["default"]:
            assert float(row["ttl_minutes"]) == 2.0
            assert 100 < row["left_s"] <= 120
        for row in rows["ten"]:
            assert float(row["ttl_minutes"]) == 10.0
            assert 580 < row["left_s"] <= 600
        for row in rows["never"]:
            assert row["ttl_minutes"] is None and row["expires_at"] is None
    assert len(vectors["default"]) == 3


def test_put_with_ttl_sets_expires_at_and_default_ttl_applies(store_factory, admin) -> None:
    store = store_factory("pool", index=index_config(), ttl={"default_ttl": 2})
    store.put(("t",), "default", TWO_MESSAGES)
    store.put(("t",), "ten", TWO_MESSAGES, ttl=10)
    store.put(("t",), "never", TWO_MESSAGES, ttl=None)
    assert_put_expiry(expiries(admin, store.sql.items), expiries(admin, store.sql.vectors))


async def test_async_put_with_ttl_sets_expires_at_and_default_ttl_applies(
    astore_factory, aadmin
) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"default_ttl": 2})
    await store.aput(("t",), "default", TWO_MESSAGES)
    await store.aput(("t",), "ten", TWO_MESSAGES, ttl=10)
    await store.aput(("t",), "never", TWO_MESSAGES, ttl=None)
    assert_put_expiry(
        await aexpiries(aadmin, store.sql.items), await aexpiries(aadmin, store.sql.vectors)
    )


def test_get_refreshes_expiry_and_refresh_false_does_not(store_factory, admin) -> None:
    store = store_factory("pool", index=index_config(), ttl={"default_ttl": 1})
    store.put(("t",), "k", TWO_MESSAGES)
    store.put(("t",), "never", TWO_MESSAGES, ttl=None)
    before = snapshot(admin, store)
    time.sleep(0.1)
    assert store.get(("t",), "k", refresh_ttl=False) is not None
    assert snapshot(admin, store) == before
    assert store.get(("t",), "k") is not None
    after = snapshot(admin, store)
    assert_extended(before, after, {"k"}, set())
    # Rows without a TTL stay without one
    assert store.get(("t",), "never") is not None
    assert snapshot(admin, store)["items"]["never"] == [None]

    # refresh_on_read=False changes the default, refresh_ttl=True still refreshes
    quiet = store_factory(
        "pool", index=index_config(), ttl={"default_ttl": 1, "refresh_on_read": False}
    )
    time.sleep(0.1)
    assert quiet.get(("t",), "k") is not None
    assert snapshot(admin, store) == after
    assert quiet.get(("t",), "k", refresh_ttl=True) is not None
    assert_extended(after, snapshot(admin, store), {"k"}, set())


async def test_async_get_refreshes_expiry_and_refresh_false_does_not(
    astore_factory, aadmin
) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"default_ttl": 1})
    await store.aput(("t",), "k", TWO_MESSAGES)
    before = await asnapshot(aadmin, store)
    await asyncio.sleep(0.1)
    assert await store.aget(("t",), "k", refresh_ttl=False) is not None
    assert await asnapshot(aadmin, store) == before
    assert await store.aget(("t",), "k") is not None
    after = await asnapshot(aadmin, store)
    assert_extended(before, after, {"k"}, set())

    quiet = await astore_factory(
        "pool", index=index_config(), ttl={"default_ttl": 1, "refresh_on_read": False}
    )
    await asyncio.sleep(0.1)
    assert await quiet.aget(("t",), "k") is not None
    assert await asnapshot(aadmin, store) == after
    assert await quiet.aget(("t",), "k", refresh_ttl=True) is not None
    assert_extended(after, await asnapshot(aadmin, store), {"k"}, set())


KEYS = {"a", "b", "c"}


def test_search_refreshes_only_returned_rows(store_factory, admin) -> None:
    store = store_factory("pool", index=index_config(), ttl={"default_ttl": 1})
    for key in sorted(KEYS):
        store.put(("t",), key, {"text": f"memory {key}"})
    # A plain search returns the two most recently updated: c, then b
    before = snapshot(admin, store)
    time.sleep(0.1)
    assert [item.key for item in store.search(("t",), limit=2)] == ["c", "b"]
    after = snapshot(admin, store)
    assert_extended(before, after, {"b", "c"}, {"a"})

    # A semantic search refreshes the two it returns, whichever they are
    time.sleep(0.1)
    found = {item.key for item in store.search(("t",), query="memory a", limit=2)}
    assert len(found) == 2
    assert_extended(after, snapshot(admin, store), found, KEYS - found)


async def test_async_search_refreshes_only_returned_rows(astore_factory, aadmin) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"default_ttl": 1})
    for key in sorted(KEYS):
        await store.aput(("t",), key, {"text": f"memory {key}"})
    before = await asnapshot(aadmin, store)
    await asyncio.sleep(0.1)
    assert [item.key for item in await store.asearch(("t",), limit=2)] == ["c", "b"]
    after = await asnapshot(aadmin, store)
    assert_extended(before, after, {"b", "c"}, {"a"})

    await asyncio.sleep(0.1)
    found = {item.key for item in await store.asearch(("t",), query="memory a", limit=2)}
    assert len(found) == 2
    assert_extended(after, await asnapshot(aadmin, store), found, KEYS - found)


def count(admin: Connection[Any], table: str, key: str) -> int:
    row = admin.execute(f"SELECT count(*) AS n FROM {table} WHERE key = %s", (key,)).fetchone()
    assert row is not None
    return int(row["n"])


async def acount(aadmin: AsyncConnection[Any], table: str, key: str) -> int:
    cur = await aadmin.execute(f"SELECT count(*) AS n FROM {table} WHERE key = %s", (key,))
    row = await cur.fetchone()
    assert row is not None
    return int(row["n"])


def test_expired_rows_are_hidden_from_get_search_and_list_before_the_job_runs(
    store_factory, admin
) -> None:
    # The default cron is hourly, so the job does not run during the test
    store = store_factory("pool", index=index_config(), ttl={"default_ttl": LONG})
    store.put(("gone",), "x", {"text": "old memory"}, ttl=SHORT)
    store.put(("live",), "y", {"text": "old memory"}, ttl=None)
    time.sleep(EXPIRED)
    assert store.get(("gone",), "x") is None
    assert store.search(("gone",)) == []
    assert [item.key for item in store.search(())] == ["y"]
    assert store.search(("gone",), query="old memory") == []
    assert [item.key for item in store.search((), query="old memory")] == ["y"]
    assert store.list_namespaces() == [("live",)]
    # The rows are still there, still expired: hidden reads did not revive them
    assert count(admin, store.sql.items, "x") == 1
    assert count(admin, store.sql.vectors, "x") == 1
    assert expiries(admin, store.sql.items)["x"][0]["left_s"] < 0
    assert expiries(admin, store.sql.vectors)["x"][0]["left_s"] < 0


async def test_async_expired_rows_are_hidden_from_get_search_and_list_before_the_job_runs(
    astore_factory, aadmin
) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"default_ttl": LONG})
    await store.aput(("gone",), "x", {"text": "old memory"}, ttl=SHORT)
    await store.aput(("live",), "y", {"text": "old memory"}, ttl=None)
    await asyncio.sleep(EXPIRED)
    assert await store.aget(("gone",), "x") is None
    assert await store.asearch(("gone",)) == []
    assert [item.key for item in await store.asearch(())] == ["y"]
    assert await store.asearch(("gone",), query="old memory") == []
    assert [item.key for item in await store.asearch((), query="old memory")] == ["y"]
    assert await store.alist_namespaces() == [("live",)]
    assert await acount(aadmin, store.sql.items, "x") == 1
    assert await acount(aadmin, store.sql.vectors, "x") == 1
    assert (await aexpiries(aadmin, store.sql.items))["x"][0]["left_s"] < 0
    assert (await aexpiries(aadmin, store.sql.vectors))["x"][0]["left_s"] < 0


# refresh_on_read is off so that reading the expired row does not revive it
SHOW_EXPIRED = {"default_ttl": LONG, "omit_expired": False, "refresh_on_read": False}


def test_omit_expired_false_returns_expired_rows_until_swept(store_factory) -> None:
    store = store_factory("pool", index=index_config(), ttl=SHOW_EXPIRED)
    store.put(("gone",), "x", {"text": "old memory"}, ttl=SHORT)
    time.sleep(EXPIRED)
    assert store.get(("gone",), "x") is not None
    assert [item.key for item in store.search(("gone",))] == ["x"]
    assert [item.key for item in store.search(("gone",), query="old memory")] == ["x"]
    assert store.list_namespaces() == [("gone",)]
    assert store.sweep_ttl() == 1
    assert store.get(("gone",), "x") is None
    assert store.search(("gone",), query="old memory") == []
    assert store.list_namespaces() == []


async def test_async_omit_expired_false_returns_expired_rows_until_swept(astore_factory) -> None:
    store = await astore_factory("pool", index=index_config(), ttl=SHOW_EXPIRED)
    await store.aput(("gone",), "x", {"text": "old memory"}, ttl=SHORT)
    await asyncio.sleep(EXPIRED)
    assert await store.aget(("gone",), "x") is not None
    assert [item.key for item in await store.asearch(("gone",))] == ["x"]
    assert [item.key for item in await store.asearch(("gone",), query="old memory")] == ["x"]
    assert await store.alist_namespaces() == [("gone",)]
    assert await store.sweep_ttl() == 1
    assert await store.aget(("gone",), "x") is None
    assert await store.asearch(("gone",), query="old memory") == []
    assert await store.alist_namespaces() == []


def test_store_without_ttl_config_still_hides_expired_rows_written_by_another_store(
    store_factory,
) -> None:
    writer = store_factory("pool", ttl={"default_ttl": LONG})
    writer.put(("gone",), "x", {"v": 1}, ttl=SHORT)
    writer.put(("live",), "y", {"v": 2})
    time.sleep(EXPIRED)
    plain = store_factory("pool")
    assert plain.get(("gone",), "x") is None
    assert [item.key for item in plain.search(())] == ["y"]
    assert plain.list_namespaces() == [("live",)]
    shows = store_factory("pool", ttl={"omit_expired": False})
    assert {item.key for item in shows.search(())} == {"x", "y"}
    assert shows.list_namespaces() == [("gone",), ("live",)]
    assert shows.get(("gone",), "x") is not None


async def test_async_store_without_ttl_config_still_hides_expired_rows_written_by_another_store(
    astore_factory,
) -> None:
    writer = await astore_factory("pool", ttl={"default_ttl": LONG})
    await writer.aput(("gone",), "x", {"v": 1}, ttl=SHORT)
    await writer.aput(("live",), "y", {"v": 2})
    await asyncio.sleep(EXPIRED)
    plain = await astore_factory("pool")
    assert await plain.aget(("gone",), "x") is None
    assert [item.key for item in await plain.asearch(())] == ["y"]
    assert await plain.alist_namespaces() == [("live",)]
    shows = await astore_factory("pool", ttl={"omit_expired": False})
    assert {item.key for item in await shows.asearch(())} == {"x", "y"}
    assert await shows.alist_namespaces() == [("gone",), ("live",)]
    assert await shows.aget(("gone",), "x") is not None


def test_store_without_ttl_config_never_writes_or_refreshes_an_expiry(store_factory, admin) -> None:
    writer = store_factory("pool", index=index_config(), ttl={"default_ttl": 1})
    writer.put(("t",), "x", TWO_MESSAGES)
    plain = store_factory("pool", index=index_config())
    plain.put(("t",), "y", TWO_MESSAGES)
    before = snapshot(admin, writer)
    assert before["items"]["y"] == [None] and before["vectors"]["y"] == [None] * 3
    time.sleep(0.1)
    # refresh_ttl resolves to true here (BaseStore's default without a
    # configuration), but a store without TTL settings leaves expiry alone
    assert plain.get(("t",), "x") is not None
    assert {item.key for item in plain.search(("t",))} == {"x", "y"}
    assert {item.key for item in plain.search(("t",), query="hello")} == {"x", "y"}
    assert snapshot(admin, writer) == before


async def test_async_store_without_ttl_config_never_writes_or_refreshes_an_expiry(
    astore_factory, aadmin
) -> None:
    writer = await astore_factory("pool", index=index_config(), ttl={"default_ttl": 1})
    await writer.aput(("t",), "x", TWO_MESSAGES)
    plain = await astore_factory("pool", index=index_config())
    await plain.aput(("t",), "y", TWO_MESSAGES)
    before = await asnapshot(aadmin, writer)
    assert before["items"]["y"] == [None] and before["vectors"]["y"] == [None] * 3
    await asyncio.sleep(0.1)
    assert await plain.aget(("t",), "x") is not None
    assert {item.key for item in await plain.asearch(("t",))} == {"x", "y"}
    assert {item.key for item in await plain.asearch(("t",), query="hello")} == {"x", "y"}
    assert await asnapshot(aadmin, writer) == before


def cron_setting(cron: str) -> str:
    return f"ttl_job_cron = '{cron}'"


def test_setup_applies_ttl_settings_to_both_tables(store_factory, admin) -> None:
    store = store_factory("pool", index=index_config(), ttl={"sweep_interval_minutes": 5})
    for table in (store.sql.items, store.sql.vectors):
        statement = create_statement(admin, table)
        assert SETTING in statement and cron_setting("*/5 * * * *") in statement
    # setup() is safe to repeat, and a changed cron applies on the next one
    store.setup()
    store_factory("pool", index=index_config(), ttl={"sweep_interval_minutes": 10})
    for table in (store.sql.items, store.sql.vectors):
        assert cron_setting("*/10 * * * *") in create_statement(admin, table)
    store_factory(
        "pool",
        index=index_config(),
        ttl={"sweep_interval_minutes": 10, "ttl_job_cron": "0 3 * * *"},
    )
    for table in (store.sql.items, store.sql.vectors):
        statement = create_statement(admin, table)
        assert SETTING in statement and cron_setting("0 3 * * *") in statement


async def test_async_setup_applies_ttl_settings_to_both_tables(astore_factory, aadmin) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"sweep_interval_minutes": 5})
    for table in (store.sql.items, store.sql.vectors):
        statement = await acreate_statement(aadmin, table)
        assert SETTING in statement and cron_setting("*/5 * * * *") in statement
    await store.setup()
    await astore_factory("pool", index=index_config(), ttl={"ttl_job_cron": "0 3 * * *"})
    for table in (store.sql.items, store.sql.vectors):
        statement = await acreate_statement(aadmin, table)
        assert SETTING in statement and cron_setting("0 3 * * *") in statement


def test_setup_without_ttl_leaves_existing_settings_and_disable_ttl_resets_them(
    store_factory, admin
) -> None:
    with_ttl = store_factory("pool", index=index_config(), ttl={"sweep_interval_minutes": 5})
    tables = (with_ttl.sql.items, with_ttl.sql.vectors)
    store_factory("pool", index=index_config())
    for table in tables:
        statement = create_statement(admin, table)
        assert SETTING in statement and cron_setting("*/5 * * * *") in statement
    with_ttl.disable_ttl()
    for table in tables:
        assert "ttl_expiration_expression" not in create_statement(admin, table)
    # Safe to repeat
    with_ttl.disable_ttl()


async def test_async_setup_without_ttl_leaves_existing_settings_and_adisable_ttl_resets_them(
    astore_factory, aadmin
) -> None:
    with_ttl = await astore_factory("pool", index=index_config(), ttl={"sweep_interval_minutes": 5})
    tables = (with_ttl.sql.items, with_ttl.sql.vectors)
    await astore_factory("pool", index=index_config())
    for table in tables:
        assert SETTING in await acreate_statement(aadmin, table)
    await with_ttl.adisable_ttl()
    for table in tables:
        assert "ttl_expiration_expression" not in await acreate_statement(aadmin, table)
    await with_ttl.adisable_ttl()


def put_sweep_rows(store: Any) -> None:
    """Two items about to expire, one with a long TTL, one without, all with vectors."""
    store.put(("t",), "e1", TWO_MESSAGES, ttl=SHORT)
    store.put(("u",), "e2", TWO_MESSAGES, ttl=SHORT)
    store.put(("t",), "live", TWO_MESSAGES, ttl=LONG)
    store.put(("t",), "never", TWO_MESSAGES, ttl=None)


def keys_in(admin: Connection[Any], table: str) -> set[str]:
    return {row["key"] for row in admin.execute(f"SELECT key FROM {table}").fetchall()}


async def akeys_in(aadmin: AsyncConnection[Any], table: str) -> set[str]:
    cur = await aadmin.execute(f"SELECT key FROM {table}")
    return {row["key"] for row in await cur.fetchall()}


def test_sweep_ttl_deletes_expired_items_and_vectors_and_returns_count(
    store_factory, admin
) -> None:
    store = store_factory("pool", index=index_config(), ttl={"default_ttl": LONG})
    put_sweep_rows(store)
    time.sleep(EXPIRED)
    assert store.sweep_ttl() == 2
    assert keys_in(admin, store.sql.items) == {"live", "never"}
    assert keys_in(admin, store.sql.vectors) == {"live", "never"}
    assert store.sweep_ttl() == 0


async def test_async_sweep_ttl_deletes_expired_items_and_vectors_and_returns_count(
    astore_factory, aadmin
) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"default_ttl": LONG})
    await store.aput(("t",), "e1", TWO_MESSAGES, ttl=SHORT)
    await store.aput(("u",), "e2", TWO_MESSAGES, ttl=SHORT)
    await store.aput(("t",), "live", TWO_MESSAGES, ttl=LONG)
    await store.aput(("t",), "never", TWO_MESSAGES, ttl=None)
    await asyncio.sleep(EXPIRED)
    assert await store.sweep_ttl() == 2
    assert await akeys_in(aadmin, store.sql.items) == {"live", "never"}
    assert await akeys_in(aadmin, store.sql.vectors) == {"live", "never"}
    assert await store.sweep_ttl() == 0


# CockroachDB's TTL job starts at a minute boundary and takes a while to
# delete, so the test polls for up to four minutes
JOB_WAIT = 240
JOB_POLL = 5


@pytest.mark.timeout(JOB_WAIT + 60)
def test_ttl_expiration_job_removes_expired_items_and_vectors(store_factory, admin) -> None:
    store = store_factory(
        "pool", index=index_config(), ttl={"default_ttl": LONG, "ttl_job_cron": "* * * * *"}
    )
    store.put(("t",), "e1", TWO_MESSAGES, ttl=0.02)
    store.put(("u",), "e2", TWO_MESSAGES, ttl=0.02)
    store.put(("t",), "never", TWO_MESSAGES, ttl=None)
    started = time.monotonic()
    items: set[str] = set()
    vectors: set[str] = set()
    while time.monotonic() - started < JOB_WAIT:
        items = keys_in(admin, store.sql.items)
        vectors = keys_in(admin, store.sql.vectors)
        if items == {"never"} and vectors == {"never"}:
            break
        time.sleep(JOB_POLL)
    print(f"TTL job wait: {time.monotonic() - started:.0f} s")
    assert items == {"never"}
    assert vectors == {"never"}
    assert count(admin, store.sql.vectors, "never") == 3


def wait_for_keys(admin: Connection[Any], tables: tuple[str, ...], keys: set[str]) -> None:
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if all(keys_in(admin, table) == keys for table in tables):
            return
        time.sleep(0.2)
    assert [keys_in(admin, table) for table in tables] == [keys] * len(tables)


@pytest.mark.timeout(60)
def test_start_and_stop_ttl_sweeper_parity(store_factory, admin) -> None:
    store = store_factory("pool", index=index_config(), ttl={"default_ttl": LONG})
    tables = (store.sql.items, store.sql.vectors)
    store.put(("t",), "short", TWO_MESSAGES, ttl=0.005)
    store.put(("t",), "long", TWO_MESSAGES)
    # About 0.6 seconds between sweeps. A second start while it runs gives
    # another future, as PostgresStore does; cancelling either stops it.
    first = store.start_ttl_sweeper(0.01)  # type: ignore[arg-type]
    second = store.start_ttl_sweeper(0.01)  # type: ignore[arg-type]
    assert second is not first and not second.done()
    wait_for_keys(admin, tables, {"long"})
    assert second.cancel()
    assert first.result(timeout=WAIT) is None
    assert store.stop_ttl_sweeper(timeout=WAIT) is True
    # It starts again after a stop, and stopping twice is fine
    store.put(("t",), "again", TWO_MESSAGES, ttl=0.005)
    third = store.start_ttl_sweeper(0.01)  # type: ignore[arg-type]
    wait_for_keys(admin, tables, {"long"})
    assert store.stop_ttl_sweeper(timeout=WAIT) is True
    assert third.result(timeout=WAIT) is None
    assert store.stop_ttl_sweeper() is True


async def await_keys(aadmin: AsyncConnection[Any], tables: tuple[str, ...], keys: set[str]) -> None:
    for _ in range(WAIT * 5):
        if all([await akeys_in(aadmin, table) == keys for table in tables]):
            return
        await asyncio.sleep(0.2)
    assert [await akeys_in(aadmin, table) for table in tables] == [keys] * len(tables)


@pytest.mark.timeout(60)
async def test_async_start_and_stop_ttl_sweeper_parity(astore_factory, aadmin) -> None:
    store = await astore_factory("pool", index=index_config(), ttl={"default_ttl": LONG})
    tables = (store.sql.items, store.sql.vectors)
    await store.aput(("t",), "short", TWO_MESSAGES, ttl=0.005)
    await store.aput(("t",), "long", TWO_MESSAGES)
    task = await store.start_ttl_sweeper(0.01)  # type: ignore[arg-type]
    assert await store.start_ttl_sweeper(0.01) is task  # type: ignore[arg-type]
    await await_keys(aadmin, tables, {"long"})
    assert await store.stop_ttl_sweeper(timeout=WAIT) is True
    assert task.done() and task.exception() is None
    await store.aput(("t",), "again", TWO_MESSAGES, ttl=0.005)
    again = await store.start_ttl_sweeper(0.01)  # type: ignore[arg-type]
    assert again is not task
    await await_keys(aadmin, tables, {"long"})
    assert await store.stop_ttl_sweeper(timeout=WAIT) is True
    assert await store.stop_ttl_sweeper() is True


# Moved from test_parity.py and test_async.py


def test_ttl_expiry_refresh_and_sweep(store_factory, admin, table_name, mode) -> None:
    # A TTL configuration also turns on row-level TTL in setup()
    store = store_factory(mode, ttl={"default_ttl": 1.0, "refresh_on_read": True})
    store.put(("t",), "short", {"v": 1}, ttl=0.01)
    store.put(("t",), "long", {"v": 2})
    store.put(("t",), "none", {"v": 3}, ttl=None)
    store.put(("t",), "one", {"v": 4}, ttl=1)

    def expires(key: str) -> Any:
        row = admin.execute(
            f"SELECT expires_at, ttl_minutes FROM {table_name} WHERE key = %s", (key,)
        ).fetchone()
        return row

    # No ttl argument: the default. ttl=None: no expiry. ttl=1: one minute
    assert expires("long")["ttl_minutes"] == 1
    assert expires("none")["ttl_minutes"] is None
    assert expires("none")["expires_at"] is None
    assert expires("one")["ttl_minutes"] == 1
    before = expires("long")["expires_at"]
    time.sleep(1.2)
    assert store.get(("t",), "short") is None
    assert {item.key for item in store.search(("t",))} == {"long", "none", "one"}
    after = expires("long")["expires_at"]
    assert after > before
    assert store.get(("t",), "long", refresh_ttl=False) is not None
    assert expires("long")["expires_at"] == after
    time.sleep(0.05)
    assert store.get(("t",), "long") is not None
    assert expires("long")["expires_at"] > after
    assert store.sweep_ttl() == 1
    assert store.sweep_ttl() == 0
    assert expires("short") is None
    assert "ttl_expiration_expression" in create_statement(admin, table_name)
    store.disable_ttl()
    assert "ttl_expiration_expression" not in create_statement(admin, table_name)


def test_store_without_ttl_config_has_no_sweeper(store_factory) -> None:
    store = store_factory("connection")
    future = store.start_ttl_sweeper()
    assert future.done() and future.result() is None


async def test_async_ttl_expiry_refresh_and_sweep(astore_factory, aadmin, table_name, mode) -> None:
    store = await astore_factory(mode, ttl={"default_ttl": 1.0, "refresh_on_read": True})
    await store.aput(("t",), "short", {"v": 1}, ttl=0.01)
    await store.aput(("t",), "long", {"v": 2})
    await store.aput(("t",), "none", {"v": 3}, ttl=None)

    async def expires(key: str) -> Any:
        cur = await aadmin.execute(
            f"SELECT expires_at, ttl_minutes FROM {table_name} WHERE key = %s", (key,)
        )
        return await cur.fetchone()

    assert (await expires("long"))["ttl_minutes"] == 1
    assert (await expires("none"))["expires_at"] is None
    before = (await expires("long"))["expires_at"]
    await asyncio.sleep(1.2)
    assert await store.aget(("t",), "short") is None
    assert {item.key for item in await store.asearch(("t",))} == {"long", "none"}
    after = (await expires("long"))["expires_at"]
    assert after > before
    assert await store.aget(("t",), "long", refresh_ttl=False) is not None
    assert (await expires("long"))["expires_at"] == after
    assert await store.sweep_ttl() == 1
    assert await store.sweep_ttl() == 0
    assert "ttl_expiration_expression" in await acreate_statement(aadmin, table_name)
    await store.adisable_ttl()
    assert "ttl_expiration_expression" not in await acreate_statement(aadmin, table_name)


async def test_async_store_without_ttl_config_has_no_sweeper(astore_factory) -> None:
    store = await astore_factory("connection")
    task = await store.start_ttl_sweeper()
    await task
    assert task.result() is None


async def test_aexit_signals_the_sweeper_to_stop(astore_factory) -> None:
    store = await astore_factory("pool", ttl={"default_ttl": 1.0})
    async with store:
        task = await store.start_ttl_sweeper(1)
    await asyncio.wait_for(task, WAIT)
