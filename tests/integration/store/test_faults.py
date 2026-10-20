"""The store under real CockroachDB faults, concurrency and a shared connection.

The store runs every database call through the retry machinery it shares
with the checkpointers. These tests make CockroachDB raise real errors,
kill pooled sessions, hold row locks and race writers, and check that the
store retries what it should, writes each row once, and leaves errors in a
caller's own transaction to the caller.
"""

from __future__ import annotations

import asyncio
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
from tests.integration.faults import (
    FORCE_ERROR,
    PsycopgFaults,
    hold_row_lock,
    kill_sessions,
    wait_for_statement,
)
from tests.integration.store.conftest import FAST

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


class FailAfterCommit:
    """Make the client see 40003 after a write batch committed.

    Once armed, the cursor sends the next COMMIT and then
    crdb_internal.force_error with 40003 in the same pipeline. The batch
    commits, and the error reaches the store as the batch's result: an
    ambiguous result after the write applied, as when a connection drops
    while the COMMIT's answer is on its way.
    """

    def __init__(self, unsafe_internals: bool) -> None:
        self.unsafe_internals = unsafe_internals
        self.armed = False
        self.fired = 0

    def cursor_class(self) -> type[psycopg.Cursor[Any]]:
        fault = self

        class FailAfterCommitCursor(psycopg.Cursor):  # type: ignore[type-arg]
            def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                result = super().execute(query, params, **kwargs)
                if fault.armed and str(query).strip().upper() == "COMMIT":
                    fault.armed = False
                    fault.fired += 1
                    if fault.unsafe_internals:
                        super().execute("SET allow_unsafe_internals = true")
                    super().execute(
                        FORCE_ERROR,
                        {"fault_code": "40003", "fault_message": "injected 40003 after COMMIT"},
                    )
                return result

        return FailAfterCommitCursor


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
    # together would. On v25.4 their TTL steps fail each other until retried
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
