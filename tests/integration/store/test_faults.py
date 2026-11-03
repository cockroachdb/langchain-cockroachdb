"""The store under real CockroachDB faults, concurrency and a shared connection.

The store runs every database call through the retry machinery it shares
with the checkpointers. These tests make CockroachDB raise real errors,
kill pooled sessions, hold row locks and race writers, and check that the
store retries what it should, writes each row once, and leaves errors in a
caller's own transaction to the caller.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import threading
import time
import uuid
from typing import Any

import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import PutOp
from psycopg import Connection

from langchain_cockroachdb import AsyncCockroachDBStore, CockroachDBStore, RetryPolicy, get_sqlstate
from langchain_cockroachdb._sql import parse_vector
from langchain_cockroachdb.store.sql import SCHEMA_CHANGE_JOB_TYPES, UNFINISHED_JOB_STATES
from tests.integration.faults import (
    FailAfterCommit,
    PsycopgFaults,
    hold_row_lock,
    kill_sessions,
    wait_for_statement,
)
from tests.integration.store.conftest import FAST
from tests.integration.store.test_schema_upgrade import NS, schema_0_5

DIMS = 4
EMBEDDINGS = DeterministicFakeEmbedding(size=DIMS)
STUCK_WRITE = RetryPolicy(attempt_timeout=0.5, initial_backoff=0.05, max_attempts=8)
RETRY_LOGGER = "langchain_cockroachdb.retry"
TTL_SETTING = "ttl_expiration_expression = 'expires_at'"


def index_config() -> dict[str, Any]:
    return {"dims": DIMS, "embed": EMBEDDINGS, "fields": ["text"]}


def app_name() -> str:
    return f"store_{uuid.uuid4().hex[:8]}"


def with_phrases(policy: RetryPolicy, phrases: str) -> RetryPolicy:
    """``policy``, or ``policy`` without transient phrases when ``phrases`` is "none".

    Without phrases, a killed pooled connection is retried only because
    psycopg reports the connection broken.
    """
    return dataclasses.replace(policy, transient_messages=()) if phrases == "none" else policy


def retries(caplog: pytest.LogCaptureFixture, label: str) -> list[str]:
    """The retry warnings logged for ``label``."""
    prefix = f"Retrying {label} after "
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def items_match(store: Any) -> str:
    """Text that only the store's item upsert contains."""
    return f"insert into {store.sql.items} (".lower()


def counts(admin: Connection[Any], store: Any) -> tuple[int, int, int, int]:
    """Item rows, distinct item keys, vector rows and distinct vector keys."""
    items = admin.execute(
        f"SELECT count(*) AS n, count(DISTINCT (prefix, key)) AS d FROM {store.sql.items}"
    ).fetchone()
    vectors = admin.execute(
        f"SELECT count(*) AS n, count(DISTINCT (prefix, key, field_name)) AS d "
        f"FROM {store.sql.vectors}"
    ).fetchone()
    assert items is not None and vectors is not None
    return items["n"], items["d"], vectors["n"], vectors["d"]


async def acounts(aadmin: Any, store: Any) -> tuple[int, int, int, int]:
    cur = await aadmin.execute(
        f"SELECT count(*) AS n, count(DISTINCT (prefix, key)) AS d FROM {store.sql.items}"
    )
    items = await cur.fetchone()
    cur = await aadmin.execute(
        f"SELECT count(*) AS n, count(DISTINCT (prefix, key, field_name)) AS d "
        f"FROM {store.sql.vectors}"
    )
    vectors = await cur.fetchone()
    return items["n"], items["d"], vectors["n"], vectors["d"]


# 40001 and 40003 on a write batch


@pytest.mark.parametrize("mode", ["pool", "connection", "pipeline"])
def test_put_batch_retries_40001_and_writes_once(
    store_factory, admin, unsafe_internals, caplog, mode: str
) -> None:
    # The item upsert of the first attempt fails, which aborts the whole
    # BEGIN..COMMIT batch. The retry writes the item and its vector once
    faults = PsycopgFaults(unsafe_internals)
    store = store_factory(
        mode, index=index_config(), connect_kwargs={"cursor_factory": faults.cursor_class()}
    )
    faults.fail_next("40001", match=items_match(store))
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        store.put(("docs",), "k1", {"text": "hello"})
    assert faults.injected == ["40001"]
    found = retries(caplog, "store put")
    assert len(found) == 1
    assert "SQLSTATE 40001" in found[0]
    assert sum("Retrying" in r.getMessage() for r in caplog.records) == 1
    assert counts(admin, store) == (1, 1, 1, 1)
    item = store.get(("docs",), "k1")
    assert item is not None and item.value == {"text": "hello"}


async def test_async_put_batch_retries_40001_and_writes_once(
    astore_factory, aadmin, unsafe_internals, caplog, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    store = await astore_factory(
        mode, index=index_config(), connect_kwargs={"cursor_factory": faults.async_cursor_class()}
    )
    faults.fail_next("40001", match=items_match(store))
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        await store.aput(("docs",), "k1", {"text": "hello"})
    assert faults.injected == ["40001"]
    found = retries(caplog, "store put")
    assert len(found) == 1
    assert "SQLSTATE 40001" in found[0]
    assert sum("Retrying" in r.getMessage() for r in caplog.records) == 1
    assert await acounts(aadmin, store) == (1, 1, 1, 1)
    item = await store.aget(("docs",), "k1")
    assert item is not None and item.value == {"text": "hello"}


@pytest.mark.parametrize("mode", ["pool", "connection", "pipeline"])
def test_put_batch_retries_40003_without_duplicates(
    store_factory, admin, unsafe_internals, caplog, mode: str
) -> None:
    # Each faulted batch commits, then the store sees 40003 and runs it again.
    # The upserts make the second run a no-op change, never a second row
    fault = FailAfterCommit(unsafe_internals)
    store = store_factory(
        mode, index=index_config(), connect_kwargs={"cursor_factory": fault.cursor_class()}
    )
    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        store.batch(
            [
                PutOp(("docs",), "k1", {"text": "first"}),
                PutOp(("docs",), "k2", {"text": "first"}),
            ]
        )
    fault.armed = True
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        store.put(("docs",), "k1", {"text": "last"})
    assert fault.fired == 2
    found = retries(caplog, "store put")
    assert len(found) == 2
    assert all("SQLSTATE 40003" in message for message in found)
    assert counts(admin, store) == (2, 2, 2, 2)
    k1, k2 = store.get(("docs",), "k1"), store.get(("docs",), "k2")
    assert k1 is not None and k1.value == {"text": "last"}
    assert k2 is not None and k2.value == {"text": "first"}


# A killed pooled connection during a read


@pytest.mark.parametrize("phrases", ["default", "none"])
@pytest.mark.parametrize("delay", [0.0, 0.3], ids=["immediate", "after_300ms"])
def test_read_batch_retries_a_killed_pooled_connection(
    store_factory, libpq_url, caplog, delay: float, phrases: str
) -> None:
    app = app_name()
    store = store_factory(
        "pool",
        retry_policy=with_phrases(FAST, phrases),
        connect_kwargs={"application_name": app},
    )
    store.put(("docs",), "k1", {"text": "hello"})
    assert kill_sessions(libpq_url, app) >= 1
    time.sleep(delay)
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        item = store.get(("docs",), "k1")
    assert item is not None and item.value == {"text": "hello"}
    assert retries(caplog, "store get")


@pytest.mark.parametrize("phrases", ["default", "none"])
async def test_async_read_batch_retries_a_killed_pooled_connection(
    astore_factory, libpq_url, caplog, phrases: str
) -> None:
    app = app_name()
    store = await astore_factory(
        "pool",
        retry_policy=with_phrases(FAST, phrases),
        connect_kwargs={"application_name": app},
    )
    await store.aput(("docs",), "k1", {"text": "hello"})
    assert kill_sessions(libpq_url, app) >= 1
    with caplog.at_level(logging.WARNING, logger=RETRY_LOGGER):
        item = await store.aget(("docs",), "k1")
    assert item is not None and item.value == {"text": "hello"}
    assert retries(caplog, "store get")


# A write stuck behind a row lock


@pytest.mark.timeout(60)
def test_stuck_write_is_abandoned_and_retried(store_factory, libpq_url, caplog) -> None:
    app = app_name()
    store = store_factory(
        "pool", retry_policy=STUCK_WRITE, connect_kwargs={"application_name": app}
    )
    store.put(("docs",), "k1", {"text": "old"})
    lock_sql = f"UPDATE {store.sql.items} SET updated_at = now() WHERE prefix = %s AND key = %s"
    errors: list[BaseException] = []

    def put() -> None:
        try:
            store.put(("docs",), "k1", {"text": "new"})
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    writer = threading.Thread(target=put, daemon=True)
    try:
        with (
            caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
            hold_row_lock(libpq_url, lock_sql, ("docs", "k1")),
        ):
            writer.start()
            # The first attempt reaches the upsert and waits on the lock
            wait_for_statement(libpq_url, app, f"insert into {store.sql.items}")
            # Release the lock only once the store gave up on that attempt
            deadline = time.monotonic() + 10
            while not retries(caplog, "store put"):
                assert time.monotonic() < deadline, "the stuck attempt was not abandoned"
                time.sleep(0.05)
    finally:
        # The lock is released by now, so a started writer can finish
        if writer.ident is not None:
            writer.join(30)
    assert not writer.is_alive()
    assert errors == []
    assert "attempt timeout" in caplog.text
    item = store.get(("docs",), "k1")
    assert item is not None and item.value == {"text": "new"}


# Two writers on one key


@pytest.mark.timeout(120)
def test_concurrent_puts_to_one_key_both_succeed(store_factory, admin) -> None:
    stores = [store_factory("pool", index=index_config()) for _ in range(2)]
    puts = 50
    errors: list[BaseException] = []
    # Both writers start together, so their puts overlap
    start = threading.Barrier(2)

    def write(writer: int) -> None:
        try:
            start.wait(30)
            for n in range(puts):
                stores[writer].put(("docs",), "shared", {"text": f"writer {writer} put {n}"})
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(writer,)) for writer in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(90)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert counts(admin, stores[0]) == (1, 1, 1, 1)
    item = stores[0].get(("docs",), "shared")
    assert item is not None
    last = {f"writer {writer} put {puts - 1}" for writer in range(2)}
    assert item.value["text"] in last
    # The item and its vector come from the same put
    row = admin.execute(f"SELECT embedding FROM {stores[0].sql.vectors}").fetchone()
    assert row is not None
    expected = EMBEDDINGS.embed_documents([item.value["text"]])[0]
    assert parse_vector(row["embedding"]) == pytest.approx(expected, abs=1e-6)


# A caller's transaction on the store's single connection


@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
def test_error_inside_callers_transaction_is_raised_at_once(
    store_factory, admin, unsafe_internals, caplog, mode: str
) -> None:
    # The caller owns the transaction. A retry inside it would keep it open
    # through the backoff and hide the 40001 from the caller's retry logic
    faults = PsycopgFaults(unsafe_internals)
    store = store_factory(
        mode, index=index_config(), connect_kwargs={"cursor_factory": faults.cursor_class()}
    )
    with (
        caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
        pytest.raises(psycopg.errors.SerializationFailure) as info,
        store.conn.transaction(),
    ):
        # The caller's own statement runs first, so the transaction is the
        # caller's and already in use when the store's put joins it
        assert store.conn.execute("SELECT 1").fetchone() is not None
        faults.fail_next("40001", match=items_match(store))
        store.put(("docs",), "k1", {"text": "hello"})
    assert get_sqlstate(info.value) == "40001"
    assert getattr(info.value, "in_caller_transaction", False) is True
    assert faults.count(items_match(store)) == 1
    assert not any("Retrying" in r.getMessage() for r in caplog.records)
    assert counts(admin, store) == (0, 0, 0, 0)
    # The caller retries its whole transaction
    with store.conn.transaction():
        store.put(("docs",), "k1", {"text": "hello"})
    assert counts(admin, store) == (1, 1, 1, 1)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
async def test_async_error_inside_callers_transaction_is_raised_at_once(
    astore_factory, aadmin, unsafe_internals, caplog, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    store = await astore_factory(
        mode, index=index_config(), connect_kwargs={"cursor_factory": faults.async_cursor_class()}
    )
    with (
        caplog.at_level(logging.WARNING, logger=RETRY_LOGGER),
        pytest.raises(psycopg.errors.SerializationFailure) as info,
    ):
        async with store.conn.transaction():
            cur = await store.conn.execute("SELECT 1")
            assert await cur.fetchone() is not None
            faults.fail_next("40001", match=items_match(store))
            await store.aput(("docs",), "k1", {"text": "hello"})
    assert get_sqlstate(info.value) == "40001"
    assert getattr(info.value, "in_caller_transaction", False) is True
    assert faults.count(items_match(store)) == 1
    assert not any("Retrying" in r.getMessage() for r in caplog.records)
    assert await acounts(aadmin, store) == (0, 0, 0, 0)
    async with store.conn.transaction():
        await store.aput(("docs",), "k1", {"text": "hello"})
    assert await acounts(aadmin, store) == (1, 1, 1, 1)


def test_single_connection_store_does_not_retry_when_closed(store_factory, caplog) -> None:
    store = store_factory("connection")
    store.put(("docs",), "k1", {"text": "hello"})
    store.conn.close()
    caplog.clear()
    with caplog.at_level(logging.WARNING), pytest.raises(psycopg.OperationalError):
        store.get(("docs",), "k1")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("the store's only connection is closed" in message for message in warnings) == 1
    assert not any("Retrying" in message for message in warnings)


# Several processes running setup() at once


def assert_set_up_once(libpq_url: str, store: Any) -> None:
    """The store's tables, vector index, migrations and TTL settings exist once."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        versions = [
            row[0] for row in conn.execute(f"SELECT v FROM {store.sql.migrations_table} ORDER BY v")
        ]
        assert versions == list(range(len(store.sql.migrations())))
        statements = {}
        for table in (store.sql.items, store.sql.vectors):
            row = conn.execute(f"SELECT create_statement FROM [SHOW CREATE TABLE {table}]")
            found = row.fetchone()
            assert found is not None
            statements[table] = found[0]
    assert statements[store.sql.vectors].count("VECTOR INDEX") == 1
    for statement in statements.values():
        assert statement.count(TTL_SETTING) == 1, statement
        assert statement.count(f"ttl_job_cron = '{store.ttl.cron}'") == 1, statement


@pytest.mark.timeout(240)
def test_parallel_setup_converges(libpq_url, table_name) -> None:
    # Six stores with the default retry policy, as six processes starting
    # together would. On v25.4 their TTL steps fail each other until retried,
    # or skip the change another one made
    errors: list[BaseException] = []
    stores: list[CockroachDBStore] = []

    def run() -> None:
        try:
            with CockroachDBStore.from_conn_string(
                libpq_url,
                index=index_config(),
                ttl={"default_ttl": 5},
                table_name=table_name,
            ) as store:
                store.setup()
                stores.append(store)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180)
    assert not any(thread.is_alive() for thread in threads), "a setup thread did not finish"
    assert errors == []
    assert len(stores) == 6
    assert_set_up_once(libpq_url, stores[0])


@pytest.mark.timeout(240)
async def test_async_parallel_setup_converges(libpq_url, table_name) -> None:
    async def run() -> AsyncCockroachDBStore:
        async with AsyncCockroachDBStore.from_conn_string(
            libpq_url,
            index=index_config(),
            ttl={"default_ttl": 5},
            table_name=table_name,
        ) as store:
            await store.setup()
            return store

    stores = await asyncio.wait_for(asyncio.gather(*(run() for _ in range(6))), 180)
    assert len(stores) == 6
    assert_set_up_once(libpq_url, stores[0])


# A 0.5 table with this many items takes migration 2 seconds to backfill.
# The processes whose migration statement returns at once wait for it in
# setup(), and their TTL steps then meet each other's TTL changes
ROWS_CONVERGE = 100_000
# The tests that watch a waiting setup() need the backfill to run longer
ROWS_BACKFILL = 200_000
HISTORIES = pytest.mark.parametrize(
    "history", [None, {"ttl_days": 30}], ids=["without-history", "with-history"]
)


def build_0_5_table(libpq_url: str, table_name: str, rows: int) -> None:
    """The tables that 0.5.x setup() left, with ``rows`` items in them."""
    with (
        CockroachDBStore.from_conn_string(
            libpq_url, index=index_config(), table_name=table_name
        ) as store,
        psycopg.connect(libpq_url, autocommit=True) as conn,
    ):
        for statement in schema_0_5(store):
            conn.execute(statement)
        conn.execute(
            f"INSERT INTO {store.sql.items} (prefix, key, value) "
            "SELECT 'user.' || CAST(i %% 100 AS STRING), 'k' || CAST(i AS STRING), "
            "CAST(%s AS JSONB) FROM generate_series(1, %s) AS i",
            ('{"text": "hello"}', rows),
        )


def upgrade_settings(table_name: str, history: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "index": index_config(),
        "ttl": {"default_ttl": 5},
        "history": history,
        "table_name": table_name,
    }


def assert_upgraded_once(libpq_url: str, store: Any, rows: int) -> None:
    """Migration 2 is recorded once and backfilled all ``rows``, and each TTL setting is in place."""
    assert_set_up_once(libpq_url, store)
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        row = conn.execute(
            f"SELECT count(*) FROM {store.sql.items} WHERE version = 1 AND confirmations = 0"
        ).fetchone()
        assert row is not None and row[0] == rows
        if store._history is not None:
            found = conn.execute(
                f"SELECT create_statement FROM [SHOW CREATE TABLE {store.sql.history}]"
            ).fetchone()
            assert found is not None
            assert found[0].count(TTL_SETTING) == 1, found[0]
            assert found[0].count("ttl_job_cron = '@hourly'") == 1, found[0]


@pytest.mark.timeout(300)
@HISTORIES
def test_parallel_setup_converges_while_migration_2_backfills(
    libpq_url, table_name, history
) -> None:
    # Six processes start on 0.6.0 together over a 0.5 table that holds
    # data. One backfills migration 2's columns. The others' migration
    # statement returns at once, so they wait in setup() until it is done,
    # then their TTL steps meet each other's TTL changes, which v25.4 refuses
    # while another runs
    build_0_5_table(libpq_url, table_name, ROWS_CONVERGE)
    errors: list[BaseException] = []
    stores: list[CockroachDBStore] = []

    def run() -> None:
        try:
            with CockroachDBStore.from_conn_string(
                libpq_url, **upgrade_settings(table_name, history)
            ) as store:
                store.setup()
                stores.append(store)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(240)
    assert not any(thread.is_alive() for thread in threads), "a setup thread did not finish"
    assert errors == []
    assert len(stores) == 6
    assert_upgraded_once(libpq_url, stores[0], ROWS_CONVERGE)


@pytest.mark.timeout(300)
@HISTORIES
async def test_async_parallel_setup_converges_while_migration_2_backfills(
    libpq_url, table_name, history
) -> None:
    await asyncio.to_thread(build_0_5_table, libpq_url, table_name, ROWS_CONVERGE)

    async def run() -> AsyncCockroachDBStore:
        async with AsyncCockroachDBStore.from_conn_string(
            libpq_url, **upgrade_settings(table_name, history)
        ) as store:
            await store.setup()
            return store

    stores = await asyncio.wait_for(asyncio.gather(*(run() for _ in range(6))), 240)
    assert len(stores) == 6
    await asyncio.to_thread(assert_upgraded_once, libpq_url, stores[0], ROWS_CONVERGE)


# A setup() whose migration 2 statement meets the same migration still
# backfilling in another process gets an answer at once, before the columns
# exist. It must wait for that migration before it records migration 2.
STORE_LOGGER = "langchain_cockroachdb.store"
WAITING = "is still applying migration 2"


def migration_2_recorded(conn: Connection[Any], store: Any) -> bool:
    row = conn.execute(
        f"SELECT EXISTS (SELECT 1 FROM {store.sql.migrations_table} WHERE v = 2)"
    ).fetchone()
    return bool(row and row[0])


def write_id_usable(conn: Connection[Any], store: Any) -> bool:
    try:
        conn.execute(f"SELECT write_id FROM {store.sql.items} LIMIT 0")
    except psycopg.errors.UndefinedColumn:
        return False
    return True


def wait_for_backfill(libpq_url: str, table_name: str, seconds: float = 30.0) -> bool:
    """Wait until a schema change job names the table, and say if one did.

    It matches more loosely than the store does: the table's name anywhere
    in the description.
    """
    sql = (
        "SELECT count(*) FROM [SHOW JOBS] WHERE job_type = ANY(CAST(%s AS STRING[])) "
        "AND status = ANY(CAST(%s AS STRING[])) AND strpos(description, %s) > 0"
    )
    params = (list(SCHEMA_CHANGE_JOB_TYPES), list(UNFINISHED_JOB_STATES), table_name)
    deadline = time.monotonic() + seconds
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        while time.monotonic() < deadline:
            row = conn.execute(sql, params).fetchone()
            if row is not None and row[0] > 0:
                return True
            time.sleep(0.05)
    return False


@pytest.mark.timeout(300)
def test_a_setup_that_did_not_run_migration_2_waits_before_recording_it(
    libpq_url, table_name, caplog, monkeypatch
) -> None:
    # Two processes start on 0.6.0 over a 0.5 table that holds data. The
    # second one's migration statement returns while the first one is still
    # backfilling. A store that finds migration 2 recorded writes its columns
    # at once, so it must not be recorded before they can be used
    build_0_5_table(libpq_url, table_name, ROWS_BACKFILL)
    settings = {"index": index_config(), "table_name": table_name}
    # (migration 2 recorded, its columns usable), as a third process sees them
    seen: list[tuple[bool, bool]] = []
    stop = threading.Event()
    # What each setup() found when it began to wait: whether the columns were
    # already public, which is False after a statement that returned early
    found_public: list[bool] = []
    wait_for_migration = CockroachDBStore._wait_for_migration

    def recording_wait(store: CockroachDBStore, migration: str) -> None:
        [[row]] = store._read_all([store.sql.reconcile_columns_exist()], "test")
        found_public.append(bool(row["ready"]))
        wait_for_migration(store, migration)

    monkeypatch.setattr(CockroachDBStore, "_wait_for_migration", recording_wait)

    def setup() -> None:
        with CockroachDBStore.from_conn_string(libpq_url, **settings) as store:
            store.setup()

    def observe(conn: Connection[Any], store: Any) -> tuple[bool, bool]:
        recorded = migration_2_recorded(conn, store)
        return recorded, write_id_usable(conn, store)

    def watch(store: Any) -> None:
        with psycopg.connect(libpq_url, autocommit=True) as conn:
            while not stop.is_set():
                seen.append(observe(conn, store))
                time.sleep(0.05)

    with (
        caplog.at_level(logging.INFO, logger=STORE_LOGGER),
        CockroachDBStore.from_conn_string(libpq_url, **settings) as reader,
        concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool,
    ):
        watcher = pool.submit(watch, reader)
        try:
            first = pool.submit(setup)
            # The second starts once the first one's backfill shows in SHOW JOBS
            assert wait_for_backfill(libpq_url, table_name), "no backfill job showed up"
            second = pool.submit(setup)
            concurrent.futures.wait([first, second], return_when=concurrent.futures.FIRST_COMPLETED)
            # A store created as soon as either setup() returns writes with the
            # migration 2 statements
            with CockroachDBStore.from_conn_string(libpq_url, **settings) as store:
                store.put(NS, "k", {"text": "hello"})
            first.result()
            second.result()
        finally:
            stop.set()
        watcher.result()
        with psycopg.connect(libpq_url, autocommit=True) as conn:
            seen.append(observe(conn, reader))

    # A setup() whose statement returned early waited for the backfill
    if False in found_public:
        assert any(WAITING in record.getMessage() for record in caplog.records)
    assert seen[0] == (False, False) and seen[-1] == (True, True)
    assert (True, False) not in seen
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        row = conn.execute(
            f"SELECT write_id FROM {reader.sql.items} WHERE prefix = %s AND key = %s",
            (".".join(NS), "k"),
        ).fetchone()
    assert row is not None and row[0] is not None


def test_the_migration_wait_takes_over_when_no_job_applies_it(
    libpq_url, table_name, caplog
) -> None:
    # A peer's migration failed or was reverted, or never ran: the columns
    # are not there and no job applies them. The wait applies the migration
    # itself instead of waiting out its budget
    build_0_5_table(libpq_url, table_name, 1_000)
    with (
        caplog.at_level(logging.INFO, logger=STORE_LOGGER),
        CockroachDBStore.from_conn_string(
            libpq_url, index=index_config(), table_name=table_name
        ) as store,
    ):
        store._wait_for_migration(store.sql.migrations()[2])
        with psycopg.connect(libpq_url, autocommit=True) as conn:
            assert write_id_usable(conn, store)
            assert not migration_2_recorded(conn, store)
    messages = [record.getMessage() for record in caplog.records]
    assert any("its columns are not public yet" in message for message in messages)
    assert any("Migration 2 on" in message and "is in place" in message for message in messages)


@pytest.mark.timeout(300)
def test_the_migration_wait_sees_a_backfill_running(libpq_url, table_name) -> None:
    # The checks setup() waits on, during a real backfill: the job is found
    # in SHOW JOBS and the columns are not public until it is done
    build_0_5_table(libpq_url, table_name, ROWS_BACKFILL)
    with CockroachDBStore.from_conn_string(
        libpq_url, index=index_config(), table_name=table_name
    ) as store:
        migration = store.sql.migrations()[2]

        def run_migration() -> None:
            with psycopg.connect(libpq_url, autocommit=True) as conn:
                conn.execute(migration)

        backfill = threading.Thread(target=run_migration)
        backfill.start()
        checks = [store.sql.schema_changes_running(), store.sql.reconcile_columns_exist()]
        seen = []
        while backfill.is_alive():
            [[jobs], [columns]] = store._read_all(checks, "test")
            seen.append((jobs["running"], columns["ready"]))
            time.sleep(0.1)
        backfill.join()
        [[jobs], [columns]] = store._read_all(checks, "test")
    assert any(running >= 1 and not ready for running, ready in seen), seen
    assert (jobs["running"], columns["ready"]) == (0, True)
