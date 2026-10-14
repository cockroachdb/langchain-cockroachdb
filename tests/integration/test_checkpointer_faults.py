"""Checkpointers survive real CockroachDB faults."""

import logging
import threading
import time
import uuid
from urllib.parse import urlsplit

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import CockroachDBSaver, RetryPolicy, get_sqlstate
from tests.integration.faults import PsycopgFaults, hold_row_lock, kill_sessions

FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)
RETRYABLE_CODES = ["40001", "40003", "57P01", "58C01", "08006"]
# A real change, so the open transaction holds a write intent that blocks the saver's UPSERT
LOCK_SQL = "UPDATE checkpoints SET metadata = '{}' WHERE thread_id = %s"


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def pool_options(application_name: str) -> dict:
    return {
        "autocommit": True,
        "row_factory": dict_row,
        "prepare_threshold": 0,
        "application_name": application_name,
    }


@pytest.fixture
def sync_pool(libpq_url):
    pools: list[ConnectionPool] = []

    def make(application_name: str, faults: PsycopgFaults | None = None) -> ConnectionPool:
        options = pool_options(application_name)
        if faults is not None:
            options["cursor_factory"] = faults.cursor_class()
        pool = ConnectionPool(libpq_url, min_size=1, max_size=4, kwargs=options, open=True)
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        pool.close()


def app_name() -> str:
    return f"saver_{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize("code", RETRYABLE_CODES)
def test_sync_saver_recovers_from_retryable_errors(sync_pool, unsafe_internals, code: str) -> None:
    faults = PsycopgFaults(unsafe_internals)
    saver = CockroachDBSaver(sync_pool(app_name(), faults), retry_policy=FAST)
    saver.setup()
    thread = str(uuid.uuid4())
    faults.fail_next(code, match="insert into checkpoints (")
    saved = saver.put(config(thread), empty_checkpoint(), {"step": 1}, {})
    faults.fail_next(code, match="from checkpoints ")
    assert saver.get_tuple(saved) is not None
    faults.fail_next(code, match="from checkpoints ")
    assert len(list(saver.list(config(thread)))) == 1
    faults.fail_next(code, match="insert into checkpoint_writes")
    saver.put_writes(saved, [("channel", "value")], task_id="t1")
    faults.fail_next(code, match="delete from checkpoints")
    saver.delete_thread(thread)
    assert saver.get_tuple(config(thread)) is None
    assert faults.injected == [code] * 5


def test_sync_saver_raises_non_retryable_errors_once(sync_pool, unsafe_internals) -> None:
    faults = PsycopgFaults(unsafe_internals)
    saver = CockroachDBSaver(sync_pool(app_name(), faults), retry_policy=FAST)
    saver.setup()
    faults.fail_next("XXA00", match="insert into checkpoints (")
    with pytest.raises(psycopg.Error) as info:
        saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert get_sqlstate(info.value) == "XXA00"
    assert faults.count("insert into checkpoints (") == 1


def test_sync_saver_with_a_pool_recovers_from_killed_connections(sync_pool, libpq_url) -> None:
    app = app_name()
    saver = CockroachDBSaver(sync_pool(app), retry_policy=FAST)
    saver.setup()
    saved = saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert kill_sessions(libpq_url, app) >= 1
    time.sleep(0.3)
    assert saver.get_tuple(saved) is not None


def test_single_connection_saver_does_not_retry_a_dead_connection(libpq_url, caplog) -> None:
    app = app_name()
    with CockroachDBSaver.from_conn_string(
        libpq_url, retry_policy=FAST, connect_kwargs={"application_name": app}
    ) as saver:
        saver.setup()
        assert kill_sessions(libpq_url, app) == 1
        time.sleep(0.3)
        with (
            caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.checkpointer.saver"),
            pytest.raises(psycopg.OperationalError),
        ):
            saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
        assert "ConnectionPool" in caplog.text


def test_parallel_setup_converges(libpq_url) -> None:
    database = f"setup_race_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    url = urlsplit(libpq_url)._replace(path=f"/{database}").geturl()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with CockroachDBSaver.from_conn_string(
                url, retry_policy=RetryPolicy(initial_backoff=0.05, max_attempts=10)
            ) as saver:
                saver.setup()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180)
    try:
        assert not any(thread.is_alive() for thread in threads), "a setup thread did not finish"
        assert errors == []
        with psycopg.connect(url, autocommit=True) as conn:
            versions = [
                row[0] for row in conn.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
            ]
        assert versions == list(range(len(CockroachDBSaver.MIGRATIONS)))
    finally:
        with psycopg.connect(libpq_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {database} CASCADE")


@pytest.mark.timeout(60)
def test_list_does_not_hold_the_lock_while_the_caller_iterates(sync_pool) -> None:
    saver = CockroachDBSaver(sync_pool(app_name()), retry_policy=FAST)
    saver.setup()
    thread = str(uuid.uuid4())
    saver.put(config(thread), empty_checkpoint(), {}, {})
    saver.put(config(thread), empty_checkpoint(), {}, {})
    for item in saver.list(config(thread)):
        assert saver.get_tuple(item.config) is not None


def test_sync_saver_attempt_timeout_rides_out_a_stuck_write(sync_pool, libpq_url, caplog) -> None:
    saver = CockroachDBSaver(
        sync_pool(app_name()),
        retry_policy=RetryPolicy(attempt_timeout=0.5, initial_backoff=0.05, max_attempts=8),
    )
    saver.setup()
    thread = str(uuid.uuid4())
    checkpoint = empty_checkpoint()
    saver.put(config(thread), checkpoint, {"step": 1}, {})
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        hold_row_lock(libpq_url, LOCK_SQL, (thread,), release_after=1.2),
    ):
        saver.put(config(thread), checkpoint, {"step": 2}, {})
    assert "attempt timeout" in caplog.text
    assert saver.get_tuple(config(thread)).metadata["step"] == 2


def test_from_conn_string_rejects_managed_options(libpq_url) -> None:
    with (
        pytest.raises(ValueError, match="autocommit"),
        CockroachDBSaver.from_conn_string(libpq_url, connect_kwargs={"autocommit": False}),
    ):
        pass


DDL_OFF_OPTIONS = "-c autocommit_before_ddl=off -c create_table_with_schema_locked=on"


@pytest.mark.parametrize("pipeline", [False, True])
def test_sync_setup_upgrade_and_ttl_work_with_autocommit_before_ddl_off(
    libpq_url, pipeline: bool
) -> None:
    # A dedicated database keeps the shared checkpoint tables out of this
    database = f"ddl_off_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    url = urlsplit(libpq_url)._replace(path=f"/{database}").geturl()
    try:
        with CockroachDBSaver.from_conn_string(
            url, pipeline=pipeline, retry_policy=FAST, connect_kwargs={"options": DDL_OFF_OPTIONS}
        ) as saver:
            saver.setup()
            # Forget the last three versions, as if a package upgrade had added them:
            # setup then alters tables that already exist and are schema_locked
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute(
                    "DELETE FROM checkpoint_migrations WHERE v >= %s",
                    (len(CockroachDBSaver.MIGRATIONS) - 3,),
                )
            saver.setup()
            saver.enable_ttl()
            saver.disable_ttl()
        with psycopg.connect(url, autocommit=True) as conn:
            versions = [
                row[0] for row in conn.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
            ]
        assert versions == list(range(len(CockroachDBSaver.MIGRATIONS)))
    finally:
        with psycopg.connect(libpq_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {database} CASCADE")
