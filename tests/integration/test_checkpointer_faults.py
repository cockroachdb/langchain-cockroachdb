"""Checkpointers survive real CockroachDB faults."""

import asyncio
import logging
import threading
import time
import uuid
from urllib.parse import urlsplit

import psycopg
import pytest
import pytest_asyncio
from langgraph.checkpoint.base import empty_checkpoint
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver, RetryPolicy, get_sqlstate
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
    writes = saver.get_tuple(saved).pending_writes
    assert len(writes) == 1
    assert writes[0][0] == "t1"
    assert writes[0][1] == "channel"
    assert writes[0][2] == "value"
    faults.fail_next(code, match="delete from checkpoints")
    saver.delete_thread(thread)
    assert saver.get_tuple(config(thread)) is None
    assert faults.injected == [code] * 5


def test_pipeline_mode_saver_recovers_from_retryable_errors(libpq_url, unsafe_internals) -> None:
    faults = PsycopgFaults(unsafe_internals)
    app = app_name()
    with CockroachDBSaver.from_conn_string(
        libpq_url,
        pipeline=True,
        retry_policy=FAST,
        connect_kwargs={"cursor_factory": faults.cursor_class(), "application_name": app},
    ) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        faults.fail_next("40001", match="insert into checkpoints (")
        saved = saver.put(config(thread), empty_checkpoint(), {"step": 1}, {})
        faults.fail_next("40001", match="from checkpoints ")
        assert saver.get_tuple(saved) is not None
        faults.fail_next("40001", match="from checkpoints ")
        assert len(list(saver.list(config(thread)))) == 1
        faults.fail_next("40001", match="insert into checkpoint_writes")
        saver.put_writes(saved, [("channel", "value")], task_id="t1")
        writes = saver.get_tuple(saved).pending_writes
        assert len(writes) == 1
        assert writes[0][0] == "t1"
        assert writes[0][1] == "channel"
        assert writes[0][2] == "value"
        faults.fail_next("40001", match="delete from checkpoints")
        saver.delete_thread(thread)
        assert saver.get_tuple(config(thread)) is None
        assert faults.injected == ["40001"] * 5


def test_sync_saver_raises_non_retryable_errors_once(sync_pool, unsafe_internals) -> None:
    faults = PsycopgFaults(unsafe_internals)
    saver = CockroachDBSaver(sync_pool(app_name(), faults), retry_policy=FAST)
    saver.setup()
    faults.fail_next("XXA00", match="insert into checkpoints (")
    with pytest.raises(psycopg.Error) as info:
        saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert get_sqlstate(info.value) == "XXA00"
    assert faults.count("insert into checkpoints (") == 1


@pytest.mark.parametrize("delay", [0.0, 0.3], ids=["immediate", "after_300ms"])
def test_sync_saver_with_a_pool_recovers_from_killed_connections(
    sync_pool, libpq_url, delay: float, caplog
) -> None:
    app = app_name()
    saver = CockroachDBSaver(sync_pool(app), retry_policy=FAST)
    saver.setup()
    saved = saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert kill_sessions(libpq_url, app) >= 1
    time.sleep(delay)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        assert saver.get_tuple(saved) is not None
    # Assert at least one retry for the get after the kill
    assert any("Retrying get checkpoint after" in record.message for record in caplog.records)


def test_single_connection_saver_does_not_retry_a_dead_connection(libpq_url, caplog) -> None:
    app = app_name()
    with CockroachDBSaver.from_conn_string(
        libpq_url, retry_policy=FAST, connect_kwargs={"application_name": app}
    ) as saver:
        saver.setup()
        assert kill_sessions(libpq_url, app) == 1
        time.sleep(0.3)
        caplog.clear()
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(psycopg.OperationalError),
        ):
            saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
        # Assert exactly one "ConnectionPool" warning and no retry
        assert sum("ConnectionPool" in record.message for record in caplog.records) == 1
        assert not any("Retrying" in record.message for record in caplog.records)


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
    count = 0
    for item in saver.list(config(thread)):
        assert saver.get_tuple(item.config) is not None
        count += 1
    assert count == 2


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


@pytest.mark.parametrize("key", ["autocommit", "row_factory", "prepare_threshold"])
def test_from_conn_string_rejects_managed_options(libpq_url, key: str) -> None:
    with (
        pytest.raises(ValueError, match=key),
        CockroachDBSaver.from_conn_string(libpq_url, connect_kwargs={key: "dummy"}),
    ):
        pass


def test_from_conn_string_retries_connect(caplog) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        pytest.raises(psycopg.OperationalError),
        CockroachDBSaver.from_conn_string(
            "postgresql://root@localhost:1/defaultdb?sslmode=disable",
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff=0.01),
        ),
    ):
        pass
    # Assert exactly one retry warning for the connect label
    retry_warnings = [
        r for r in caplog.records if "Retrying connect checkpointer after" in r.message
    ]
    assert len(retry_warnings) == 1


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


@pytest_asyncio.fixture
async def async_pool(libpq_url):
    pools: list[AsyncConnectionPool] = []

    async def make(
        application_name: str, faults: PsycopgFaults | None = None
    ) -> AsyncConnectionPool:
        options = pool_options(application_name)
        if faults is not None:
            options["cursor_factory"] = faults.async_cursor_class()
        pool = AsyncConnectionPool(libpq_url, min_size=1, max_size=4, kwargs=options, open=False)
        await pool.open()
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        await pool.close()


@pytest.mark.parametrize("code", RETRYABLE_CODES)
async def test_async_saver_recovers_from_retryable_errors(
    async_pool, unsafe_internals, code: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    saver = AsyncCockroachDBSaver(await async_pool(app_name(), faults), retry_policy=FAST)
    await saver.setup()
    thread = str(uuid.uuid4())
    faults.fail_next(code, match="insert into checkpoints (")
    saved = await saver.aput(config(thread), empty_checkpoint(), {"step": 1}, {})
    faults.fail_next(code, match="from checkpoints ")
    assert await saver.aget_tuple(saved) is not None
    faults.fail_next(code, match="from checkpoints ")
    assert len([item async for item in saver.alist(config(thread))]) == 1
    faults.fail_next(code, match="insert into checkpoint_writes")
    await saver.aput_writes(saved, [("channel", "value")], task_id="t1")
    writes = (await saver.aget_tuple(saved)).pending_writes
    assert len(writes) == 1
    assert writes[0][0] == "t1"
    assert writes[0][1] == "channel"
    assert writes[0][2] == "value"
    faults.fail_next(code, match="delete from checkpoints")
    await saver.adelete_thread(thread)
    assert await saver.aget_tuple(config(thread)) is None
    assert faults.injected == [code] * 5


async def test_pipeline_mode_async_saver_recovers_from_retryable_errors(
    libpq_url, unsafe_internals
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    app = app_name()
    async with AsyncCockroachDBSaver.from_conn_string(
        libpq_url,
        pipeline=True,
        retry_policy=FAST,
        connect_kwargs={"cursor_factory": faults.async_cursor_class(), "application_name": app},
    ) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        faults.fail_next("40001", match="insert into checkpoints (")
        saved = await saver.aput(config(thread), empty_checkpoint(), {"step": 1}, {})
        faults.fail_next("40001", match="from checkpoints ")
        assert await saver.aget_tuple(saved) is not None
        faults.fail_next("40001", match="from checkpoints ")
        assert len([item async for item in saver.alist(config(thread))]) == 1
        faults.fail_next("40001", match="insert into checkpoint_writes")
        await saver.aput_writes(saved, [("channel", "value")], task_id="t1")
        writes = (await saver.aget_tuple(saved)).pending_writes
        assert len(writes) == 1
        assert writes[0][0] == "t1"
        assert writes[0][1] == "channel"
        assert writes[0][2] == "value"
        faults.fail_next("40001", match="delete from checkpoints")
        await saver.adelete_thread(thread)
        assert await saver.aget_tuple(config(thread)) is None
        assert faults.injected == ["40001"] * 5


@pytest.mark.parametrize("delay", [0.0, 0.3], ids=["immediate", "after_300ms"])
async def test_async_saver_with_a_pool_recovers_from_killed_connections(
    async_pool, libpq_url, delay: float, caplog
) -> None:
    app = app_name()
    saver = AsyncCockroachDBSaver(await async_pool(app), retry_policy=FAST)
    await saver.setup()
    saved = await saver.aput(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert kill_sessions(libpq_url, app) >= 1
    await asyncio.sleep(delay)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        assert await saver.aget_tuple(saved) is not None
    # Assert at least one retry for the get after the kill
    assert any("Retrying get checkpoint after" in record.message for record in caplog.records)


async def test_async_single_connection_does_not_retry_a_dead_connection(libpq_url, caplog) -> None:
    app = app_name()
    async with AsyncCockroachDBSaver.from_conn_string(
        libpq_url, retry_policy=FAST, connect_kwargs={"application_name": app}
    ) as saver:
        await saver.setup()
        assert kill_sessions(libpq_url, app) == 1
        await asyncio.sleep(0.3)
        caplog.clear()
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(psycopg.OperationalError),
        ):
            await saver.aput(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
        # Assert exactly one "ConnectionPool" warning and no retry
        assert sum("ConnectionPool" in record.message for record in caplog.records) == 1
        assert not any("Retrying" in record.message for record in caplog.records)


@pytest.mark.timeout(60)
async def test_alist_does_not_hold_the_lock_while_the_caller_iterates(async_pool) -> None:
    saver = AsyncCockroachDBSaver(await async_pool(app_name()), retry_policy=FAST)
    await saver.setup()
    thread = str(uuid.uuid4())
    await saver.aput(config(thread), empty_checkpoint(), {}, {})
    await saver.aput(config(thread), empty_checkpoint(), {}, {})
    count = 0
    async for item in saver.alist(config(thread)):
        assert await saver.aget_tuple(item.config) is not None
        count += 1
    assert count == 2


async def test_async_saver_attempt_timeout_rides_out_a_stuck_write(
    async_pool, libpq_url, caplog
) -> None:
    saver = AsyncCockroachDBSaver(
        await async_pool(app_name()),
        retry_policy=RetryPolicy(attempt_timeout=0.5, initial_backoff=0.05, max_attempts=8),
    )
    await saver.setup()
    thread = str(uuid.uuid4())
    checkpoint = empty_checkpoint()
    await saver.aput(config(thread), checkpoint, {"step": 1}, {})
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        hold_row_lock(libpq_url, LOCK_SQL, (thread,), release_after=1.2),
    ):
        await saver.aput(config(thread), checkpoint, {"step": 2}, {})
    assert "attempt timeout" in caplog.text
    assert (await saver.aget_tuple(config(thread))).metadata["step"] == 2


async def test_async_saver_raises_non_retryable_errors_once(async_pool, unsafe_internals) -> None:
    faults = PsycopgFaults(unsafe_internals)
    saver = AsyncCockroachDBSaver(await async_pool(app_name(), faults), retry_policy=FAST)
    await saver.setup()
    faults.fail_next("XXA00", match="insert into checkpoints (")
    with pytest.raises(psycopg.Error) as info:
        await saver.aput(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert get_sqlstate(info.value) == "XXA00"
    assert faults.count("insert into checkpoints (") == 1


async def test_async_parallel_setup_converges(libpq_url) -> None:
    database = f"setup_race_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    url = urlsplit(libpq_url)._replace(path=f"/{database}").geturl()

    async def setup_task() -> None:
        async with AsyncCockroachDBSaver.from_conn_string(
            url, retry_policy=RetryPolicy(initial_backoff=0.05, max_attempts=10)
        ) as saver:
            await saver.setup()

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[setup_task() for _ in range(6)], return_exceptions=True), timeout=180
        )
        exceptions = [r for r in results if isinstance(r, BaseException)]
        assert exceptions == []
        with psycopg.connect(url, autocommit=True) as conn:
            versions = [
                row[0] for row in conn.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
            ]
        assert versions == list(range(len(AsyncCockroachDBSaver.MIGRATIONS)))
    finally:
        with psycopg.connect(libpq_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {database} CASCADE")


@pytest.mark.parametrize("pipeline", [False, True])
async def test_async_setup_upgrade_and_ttl_work_with_autocommit_before_ddl_off(
    libpq_url, pipeline: bool
) -> None:
    # A dedicated database keeps the shared checkpoint tables out of this
    database = f"ddl_off_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    url = urlsplit(libpq_url)._replace(path=f"/{database}").geturl()
    try:
        async with AsyncCockroachDBSaver.from_conn_string(
            url, pipeline=pipeline, retry_policy=FAST, connect_kwargs={"options": DDL_OFF_OPTIONS}
        ) as saver:
            await saver.setup()
            # Forget the last three versions, as if a package upgrade had added them:
            # setup then alters tables that already exist and are schema_locked
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute(
                    "DELETE FROM checkpoint_migrations WHERE v >= %s",
                    (len(AsyncCockroachDBSaver.MIGRATIONS) - 3,),
                )
            await saver.setup()
            await saver.aenable_ttl()
            await saver.adisable_ttl()
        with psycopg.connect(url, autocommit=True) as conn:
            versions = [
                row[0] for row in conn.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
            ]
        assert versions == list(range(len(AsyncCockroachDBSaver.MIGRATIONS)))
    finally:
        with psycopg.connect(libpq_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {database} CASCADE")


@pytest.mark.parametrize("key", ["autocommit", "row_factory", "prepare_threshold"])
async def test_async_from_conn_string_rejects_managed_options(libpq_url, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        async with AsyncCockroachDBSaver.from_conn_string(libpq_url, connect_kwargs={key: "dummy"}):
            pass


async def test_async_from_conn_string_retries_connect(caplog) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        pytest.raises(psycopg.OperationalError),
    ):
        async with AsyncCockroachDBSaver.from_conn_string(
            "postgresql://root@localhost:1/defaultdb?sslmode=disable",
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff=0.01),
        ):
            pass
    # Assert exactly one retry warning for the connect label
    retry_warnings = [
        r for r in caplog.records if "Retrying connect checkpointer after" in r.message
    ]
    assert len(retry_warnings) == 1
