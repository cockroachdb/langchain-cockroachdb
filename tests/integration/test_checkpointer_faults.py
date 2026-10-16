"""Checkpointers survive real CockroachDB faults."""

import asyncio
import contextlib
import dataclasses
import logging
import re
import threading
import time
import uuid
from urllib.parse import urlsplit

import psycopg
import pytest
import pytest_asyncio
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import (
    UNCATEGORIZED_SQLSTATES,
    AsyncCockroachDBSaver,
    CockroachDBSaver,
    RetryPolicy,
    get_sqlstate,
)
from tests.integration.faults import (
    Interruption,
    PsycopgFaults,
    hold_row_lock,
    kill_sessions,
    wait_for_statement,
)

FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)
STUCK_WRITE = RetryPolicy(attempt_timeout=0.5, initial_backoff=0.05, max_attempts=8)
RETRYABLE_CODES = ["40001", "40003", "57P01", "58C01", "08006"]
# A real change, so the open transaction holds a write intent that blocks the saver's UPSERT
LOCK_SQL = "UPDATE checkpoints SET metadata = '{}' WHERE thread_id = %s"


def with_phrases(policy: RetryPolicy, phrases: str) -> RetryPolicy:
    """``policy`` as it is, or with no transient phrases when ``phrases`` is "none".

    Without phrases, an error with no usable SQLSTATE is retried only because
    psycopg reports its pooled connection broken.
    """
    return dataclasses.replace(policy, transient_messages=()) if phrases == "none" else policy


# The error part of a retry warning: its class, then its SQLSTATE if it has one
ERROR_DESCRIPTION = re.compile(r"\w+(?: \(SQLSTATE (?P<code>\w{5})\))?: ")


def assert_retried_for_the_broken_flag(caplog: pytest.LogCaptureFixture, label: str) -> None:
    """Assert that every retry of ``label`` was for an error with no usable SQLSTATE.

    Under a policy without transient phrases, only psycopg's broken flag makes
    such an error retryable. A retryable SQLSTATE, such as 57P01 or an 08 code,
    would be retried by the SQLSTATE rule instead and leave the flag untested.
    """
    prefix = f"Retrying {label} after "
    retries = [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]
    assert retries, f"no retry of {label}"
    for message in retries:
        description = ERROR_DESCRIPTION.match(message, len(prefix))
        assert description is not None, message
        code = description.group("code")
        assert code is None or code in UNCATEGORIZED_SQLSTATES, message


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

    def make(
        application_name: str,
        faults: PsycopgFaults | None = None,
        *,
        cursor_factory: type[psycopg.Cursor] | None = None,
        max_size: int = 4,
    ) -> ConnectionPool:
        options = pool_options(application_name)
        if faults is not None:
            options["cursor_factory"] = faults.cursor_class()
        if cursor_factory is not None:
            options["cursor_factory"] = cursor_factory
        pool = ConnectionPool(libpq_url, min_size=1, max_size=max_size, kwargs=options, open=True)
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


@pytest.mark.parametrize("phrases", ["default", "none"])
@pytest.mark.parametrize("delay", [0.0, 0.3], ids=["immediate", "after_300ms"])
def test_sync_saver_with_a_pool_recovers_from_killed_connections(
    sync_pool, libpq_url, delay: float, phrases: str, caplog
) -> None:
    app = app_name()
    saver = CockroachDBSaver(sync_pool(app), retry_policy=with_phrases(FAST, phrases))
    saver.setup()
    saved = saver.put(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert kill_sessions(libpq_url, app) >= 1
    time.sleep(delay)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        assert saver.get_tuple(saved) is not None
    # Assert at least one retry for the get after the kill
    assert any("Retrying get checkpoint after" in record.message for record in caplog.records)
    if phrases == "none":
        assert_retried_for_the_broken_flag(caplog, "get checkpoint")


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


@pytest.mark.parametrize("phrases", ["default", "none"])
def test_sync_saver_attempt_timeout_rides_out_a_stuck_write(
    sync_pool, libpq_url, caplog, phrases: str
) -> None:
    saver = CockroachDBSaver(sync_pool(app_name()), retry_policy=with_phrases(STUCK_WRITE, phrases))
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
    if phrases == "none":
        assert_retried_for_the_broken_flag(caplog, "put checkpoint")
    assert saver.get_tuple(config(thread)).metadata["step"] == 2


@pytest.mark.timeout(60)
def test_pooled_sync_saver_writes_other_threads_while_one_write_waits(sync_pool, libpq_url) -> None:
    # Each operation runs on its own pooled connection, so a write that waits on
    # a locked row holds up nothing else
    app = app_name()
    saver = CockroachDBSaver(sync_pool(app), retry_policy=FAST)
    saver.setup()
    waiting, other = str(uuid.uuid4()), str(uuid.uuid4())
    checkpoint = empty_checkpoint()
    saver.put(config(waiting), checkpoint, {"step": 1}, {})
    with hold_row_lock(libpq_url, LOCK_SQL, (waiting,), release_after=30):
        writer = threading.Thread(
            target=saver.put, args=(config(waiting), checkpoint, {"step": 2}, {}), daemon=True
        )
        writer.start()
        wait_for_statement(libpq_url, app, "insert into checkpoints")
        second = threading.Thread(
            target=saver.put, args=(config(other), empty_checkpoint(), {}, {}), daemon=True
        )
        second.start()
        second.join(10)
        assert not second.is_alive(), "the second write waited for the blocked one"
        assert writer.is_alive()
    writer.join(30)
    assert not writer.is_alive()
    assert saver.get_tuple(config(waiting)).metadata["step"] == 2
    assert saver.get_tuple(config(other)) is not None


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
        application_name: str,
        faults: PsycopgFaults | None = None,
        *,
        cursor_factory: type[psycopg.AsyncCursor] | None = None,
        max_size: int = 4,
    ) -> AsyncConnectionPool:
        options = pool_options(application_name)
        if faults is not None:
            options["cursor_factory"] = faults.async_cursor_class()
        if cursor_factory is not None:
            options["cursor_factory"] = cursor_factory
        pool = AsyncConnectionPool(
            libpq_url, min_size=1, max_size=max_size, kwargs=options, open=False
        )
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


@pytest.mark.parametrize("phrases", ["default", "none"])
@pytest.mark.parametrize("delay", [0.0, 0.3], ids=["immediate", "after_300ms"])
async def test_async_saver_with_a_pool_recovers_from_killed_connections(
    async_pool, libpq_url, delay: float, phrases: str, caplog
) -> None:
    app = app_name()
    saver = AsyncCockroachDBSaver(await async_pool(app), retry_policy=with_phrases(FAST, phrases))
    await saver.setup()
    saved = await saver.aput(config(str(uuid.uuid4())), empty_checkpoint(), {}, {})
    assert kill_sessions(libpq_url, app) >= 1
    await asyncio.sleep(delay)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        assert await saver.aget_tuple(saved) is not None
    # Assert at least one retry for the get after the kill
    assert any("Retrying get checkpoint after" in record.message for record in caplog.records)
    if phrases == "none":
        assert_retried_for_the_broken_flag(caplog, "get checkpoint")


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


@pytest.mark.parametrize("phrases", ["default", "none"])
async def test_async_saver_attempt_timeout_rides_out_a_stuck_write(
    async_pool, libpq_url, caplog, phrases: str
) -> None:
    saver = AsyncCockroachDBSaver(
        await async_pool(app_name()), retry_policy=with_phrases(STUCK_WRITE, phrases)
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
    if phrases == "none":
        assert_retried_for_the_broken_flag(caplog, "put checkpoint")
    assert (await saver.aget_tuple(config(thread))).metadata["step"] == 2


@pytest.mark.timeout(60)
async def test_pooled_async_saver_writes_other_threads_while_one_write_waits(
    async_pool, libpq_url
) -> None:
    app = app_name()
    saver = AsyncCockroachDBSaver(await async_pool(app), retry_policy=FAST)
    await saver.setup()
    waiting, other = str(uuid.uuid4()), str(uuid.uuid4())
    checkpoint = empty_checkpoint()
    await saver.aput(config(waiting), checkpoint, {"step": 1}, {})
    with hold_row_lock(libpq_url, LOCK_SQL, (waiting,), release_after=30):
        writer = asyncio.create_task(saver.aput(config(waiting), checkpoint, {"step": 2}, {}))
        await asyncio.to_thread(wait_for_statement, libpq_url, app, "insert into checkpoints")
        await asyncio.wait_for(saver.aput(config(other), empty_checkpoint(), {}, {}), 10)
        assert not writer.done()
    await asyncio.wait_for(writer, 30)
    assert (await saver.aget_tuple(config(waiting))).metadata["step"] == 2
    assert await saver.aget_tuple(config(other)) is not None


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


# One transaction per write. A write with several statements (put with
# channel blobs, put_writes with several writes, delete_thread) sends BEGIN,
# its statements and COMMIT as one batch. It commits all of them or none, even
# when the caller stops partway through, and a failed transaction is rolled
# back before the connection is used again.

WRITE_MODES = pytest.mark.parametrize("mode", ["pool", "connection", "pipeline"])
# Where each write is stopped: a statement sent after BEGIN and at least one
# other statement of the write, or for put_writes its only one
STOP_AT = {
    "put": "insert into checkpoints (",
    "put_writes": "insert into checkpoint_writes",
    "delete_thread": "delete from checkpoint_blobs",
}
# Rows in checkpoints, checkpoint_blobs and checkpoint_writes after each write
ROWS_AFTER = {"put": (1, 1, 0), "put_writes": (1, 1, 3), "delete_thread": (0, 0, 0)}
CHANNEL_VALUES = {"messages": ["hello"], "count": 1}


def checkpoint_with_blobs() -> tuple[dict, dict[str, str]]:
    """A checkpoint whose put writes a blob row and then the checkpoint row."""
    checkpoint = empty_checkpoint()
    versions = {"messages": "1", "count": "1"}
    checkpoint["channel_values"] = dict(CHANNEL_VALUES)
    checkpoint["channel_versions"] = dict(versions)
    return checkpoint, versions


def rows(libpq_url: str, thread: str) -> tuple[int, ...]:
    """How many rows ``thread`` has in each checkpoint table."""
    tables = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        return tuple(
            conn.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE thread_id = %s").format(
                    sql.Identifier(table)
                ),
                (thread,),
            ).fetchone()[0]
            for table in tables
        )


@contextlib.contextmanager
def sync_saver_for(mode: str, libpq_url: str, sync_pool, cursor_factory):
    """A sync saver on a pool of one connection, a single connection, or a pipeline.

    A pool of one makes every retry and every later call reuse the connection.
    """
    app = app_name()
    if mode == "pool":
        pool = sync_pool(app, cursor_factory=cursor_factory, max_size=1)
        yield CockroachDBSaver(pool, retry_policy=FAST)
        return
    with CockroachDBSaver.from_conn_string(
        libpq_url,
        pipeline=mode == "pipeline",
        retry_policy=FAST,
        connect_kwargs={"cursor_factory": cursor_factory, "application_name": app},
    ) as saver:
        yield saver


@contextlib.asynccontextmanager
async def async_saver_for(mode: str, libpq_url: str, async_pool, cursor_factory):
    """The async version of sync_saver_for."""
    app = app_name()
    if mode == "pool":
        pool = await async_pool(app, cursor_factory=cursor_factory, max_size=1)
        yield AsyncCockroachDBSaver(pool, retry_policy=FAST)
        return
    async with AsyncCockroachDBSaver.from_conn_string(
        libpq_url,
        pipeline=mode == "pipeline",
        retry_policy=FAST,
        connect_kwargs={"cursor_factory": cursor_factory, "application_name": app},
    ) as saver:
        yield saver


def sync_write(saver: CockroachDBSaver, operation: str, thread: str):
    """Set up what ``operation`` needs on ``thread`` and return the write itself."""
    checkpoint, versions = checkpoint_with_blobs()
    if operation == "put":
        return lambda: saver.put(config(thread), checkpoint, {}, versions)
    saved = saver.put(config(thread), checkpoint, {}, versions)
    saver.put_writes(saved, [("a", 1)], task_id="t0")
    if operation == "put_writes":
        return lambda: saver.put_writes(saved, [("b", 2), ("c", 3)], task_id="t1")
    return lambda: saver.delete_thread(thread)


async def async_write(saver: AsyncCockroachDBSaver, operation: str, thread: str):
    """The async version of sync_write; the write it returns is a coroutine function."""
    checkpoint, versions = checkpoint_with_blobs()
    if operation == "put":
        return lambda: saver.aput(config(thread), checkpoint, {}, versions)
    saved = await saver.aput(config(thread), checkpoint, {}, versions)
    await saver.aput_writes(saved, [("a", 1)], task_id="t0")
    if operation == "put_writes":
        return lambda: saver.aput_writes(saved, [("b", 2), ("c", 3)], task_id="t1")
    return lambda: saver.adelete_thread(thread)


def assert_left_clean(caplog: pytest.LogCaptureFixture, saver) -> None:
    """The saver rolled back the stopped transaction itself.

    psycopg_pool warns "rolling back returned connection" when it gets back a
    connection still inside a transaction, and counts the broken ones it drops.
    """
    assert not any("rolling back returned connection" in r.getMessage() for r in caplog.records)
    if isinstance(saver.conn, (ConnectionPool, AsyncConnectionPool)):
        assert saver.conn.get_stats().get("returns_bad", 0) == 0
    else:
        assert saver.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


@WRITE_MODES
@pytest.mark.parametrize("operation", list(STOP_AT))
def test_sync_saver_leaves_nothing_behind_when_a_write_is_interrupted(
    sync_pool, libpq_url, caplog, mode: str, operation: str
) -> None:
    interruption = Interruption(STOP_AT[operation])
    with sync_saver_for(mode, libpq_url, sync_pool, interruption.cursor_class()) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        write = sync_write(saver, operation, thread)
        before = rows(libpq_url, thread)
        interruption.armed = True
        with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="interrupted"):
            write()
        assert rows(libpq_url, thread) == before
        assert_left_clean(caplog, saver)
        # The rollback left the connection clean, so the same write now commits
        write()
        assert rows(libpq_url, thread) == ROWS_AFTER[operation]


@WRITE_MODES
@pytest.mark.parametrize("operation", list(STOP_AT))
async def test_async_saver_leaves_nothing_behind_when_a_write_is_cancelled(
    async_pool, libpq_url, caplog, mode: str, operation: str
) -> None:
    # The cancellation lands between BEGIN and COMMIT, with earlier statements sent
    interruption = Interruption(STOP_AT[operation])
    async with async_saver_for(
        mode, libpq_url, async_pool, interruption.async_cursor_class()
    ) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        write = await async_write(saver, operation, thread)
        before = rows(libpq_url, thread)
        interruption.reached = asyncio.Event()
        interruption.armed = True
        with caplog.at_level(logging.WARNING):
            task = asyncio.create_task(write())
            await asyncio.wait_for(interruption.reached.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert rows(libpq_url, thread) == before
        assert_left_clean(caplog, saver)
        await write()
        assert rows(libpq_url, thread) == ROWS_AFTER[operation]


@WRITE_MODES
@pytest.mark.parametrize("operation", list(STOP_AT))
def test_sync_saver_leaves_nothing_behind_when_a_statement_in_the_batch_fails(
    sync_pool, libpq_url, unsafe_internals, caplog, mode: str, operation: str
) -> None:
    # A server error after BEGIN that the retry policy does not retry
    faults = PsycopgFaults(unsafe_internals)
    with sync_saver_for(mode, libpq_url, sync_pool, faults.cursor_class()) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        write = sync_write(saver, operation, thread)
        before = rows(libpq_url, thread)
        faults.fail_next("XXA00", match=STOP_AT[operation])
        with caplog.at_level(logging.WARNING), pytest.raises(psycopg.Error) as info:
            write()
        assert get_sqlstate(info.value) == "XXA00"
        assert rows(libpq_url, thread) == before
        assert_left_clean(caplog, saver)
        write()
        assert rows(libpq_url, thread) == ROWS_AFTER[operation]


@WRITE_MODES
@pytest.mark.parametrize("operation", list(STOP_AT))
async def test_async_saver_leaves_nothing_behind_when_a_statement_in_the_batch_fails(
    async_pool, libpq_url, unsafe_internals, caplog, mode: str, operation: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with async_saver_for(mode, libpq_url, async_pool, faults.async_cursor_class()) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        write = await async_write(saver, operation, thread)
        before = rows(libpq_url, thread)
        faults.fail_next("XXA00", match=STOP_AT[operation])
        with caplog.at_level(logging.WARNING), pytest.raises(psycopg.Error) as info:
            await write()
        assert get_sqlstate(info.value) == "XXA00"
        assert rows(libpq_url, thread) == before
        assert_left_clean(caplog, saver)
        await write()
        assert rows(libpq_url, thread) == ROWS_AFTER[operation]


@WRITE_MODES
def test_sync_saver_write_batch_is_retried_by_cockroachdb(
    sync_pool, libpq_url, unsafe_internals, caplog, mode: str
) -> None:
    # BEGIN, the statements and COMMIT reach the server as one batch, so
    # CockroachDB retries the batch itself and the saver sees no error.
    # force_retry fails the transaction until it is 500 ms old, so the write
    # cannot finish sooner; had the server not retried, the saver would have.
    faults = PsycopgFaults(unsafe_internals)
    with sync_saver_for(mode, libpq_url, sync_pool, faults.cursor_class()) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        faults.server_retry_next("insert into checkpoints (", interval="500ms")
        start = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
            saver.put(config(thread), checkpoint, {}, versions)
        assert time.monotonic() - start >= 0.5
        assert not any("Retrying" in record.getMessage() for record in caplog.records)
        assert rows(libpq_url, thread) == ROWS_AFTER["put"]


@WRITE_MODES
async def test_async_saver_write_batch_is_retried_by_cockroachdb(
    async_pool, libpq_url, unsafe_internals, caplog, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with async_saver_for(mode, libpq_url, async_pool, faults.async_cursor_class()) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        faults.server_retry_next("insert into checkpoints (", interval="500ms")
        start = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
            await saver.aput(config(thread), checkpoint, {}, versions)
        assert time.monotonic() - start >= 0.5
        assert not any("Retrying" in record.getMessage() for record in caplog.records)
        assert rows(libpq_url, thread) == ROWS_AFTER["put"]


def test_sync_pipeline_saver_retries_a_write_that_follows_a_read(
    libpq_url, unsafe_internals, caplog
) -> None:
    # On a saver-wide pipeline a read leaves its implicit transaction open until
    # the next sync, and the next write's BEGIN joins it. CockroachDB cannot
    # retry that batch itself, because the read's results already reached the
    # client, so the retry error reaches the saver, which rolls back and retries.
    faults = PsycopgFaults(unsafe_internals)
    with sync_saver_for("pipeline", libpq_url, None, faults.cursor_class()) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        assert saver.get_tuple(config(thread)) is None
        # The transaction started at the read, so a pause after it must not outlast this
        faults.server_retry_next("insert into checkpoints (", interval="5s")
        with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
            saver.put(config(thread), checkpoint, {}, versions)
        retries = [
            r.getMessage() for r in caplog.records if "Retrying put checkpoint" in r.getMessage()
        ]
        assert len(retries) == 1
        assert "SQLSTATE 40001" in retries[0]
        assert rows(libpq_url, thread) == ROWS_AFTER["put"]


async def test_async_pipeline_saver_retries_a_write_that_follows_a_read(
    libpq_url, unsafe_internals, caplog
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with async_saver_for("pipeline", libpq_url, None, faults.async_cursor_class()) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        assert await saver.aget_tuple(config(thread)) is None
        faults.server_retry_next("insert into checkpoints (", interval="5s")
        with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
            await saver.aput(config(thread), checkpoint, {}, versions)
        retries = [
            r.getMessage() for r in caplog.records if "Retrying put checkpoint" in r.getMessage()
        ]
        assert len(retries) == 1
        assert "SQLSTATE 40001" in retries[0]
        assert rows(libpq_url, thread) == ROWS_AFTER["put"]


# A caller that shares the saver's single connection. Rows are counted only
# after the caller's transaction ends: another connection reading them earlier
# would wait for its intents.
CALLER_ROW_SQL = (
    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
    "VALUES (%s, '', 'caller', '{}')"
)


def delete_caller_row(libpq_url: str, caller: str) -> None:
    """Remove the caller's stand-in row, which is not a real checkpoint and would
    break anything that lists every checkpoint in the table afterwards."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (caller,))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("outcome", ["commit", "rollback"])
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
def test_sync_saver_write_joins_the_callers_transaction(libpq_url, mode: str, outcome: str) -> None:
    # The write nests a savepoint in the caller's transaction instead of a BEGIN,
    # which would fail and abort it, so the caller's commit or rollback decides
    with sync_saver_for(mode, libpq_url, None, psycopg.Cursor) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        with saver.conn.transaction():
            saver.put(config(thread), checkpoint, {}, versions)
            if outcome == "rollback":
                raise psycopg.Rollback()
        assert rows(libpq_url, thread) == (ROWS_AFTER["put"] if outcome == "commit" else (0, 0, 0))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("outcome", ["commit", "rollback"])
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
async def test_async_saver_write_joins_the_callers_transaction(
    libpq_url, mode: str, outcome: str
) -> None:
    async with async_saver_for(mode, libpq_url, None, psycopg.AsyncCursor) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        async with saver.conn.transaction():
            await saver.aput(config(thread), checkpoint, {}, versions)
            if outcome == "rollback":
                raise psycopg.Rollback()
        assert rows(libpq_url, thread) == (ROWS_AFTER["put"] if outcome == "commit" else (0, 0, 0))


@pytest.mark.timeout(60)
def test_sync_saver_leaves_the_callers_open_transaction_alone(libpq_url) -> None:
    # The caller opened a transaction with a plain BEGIN and wrote in it. Neither
    # a read nor a write of the saver may roll that back.
    with sync_saver_for("connection", libpq_url, None, psycopg.Cursor) as saver:
        saver.setup()
        caller, thread = str(uuid.uuid4()), str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        try:
            saver.conn.execute("BEGIN")
            saver.conn.execute(CALLER_ROW_SQL, (caller,))
            assert saver.get_tuple(config(thread)) is None
            saver.put(config(thread), checkpoint, {}, versions)
            assert saver.conn.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            saver.conn.execute("COMMIT")
            assert rows(libpq_url, caller) == (1, 0, 0)
            assert rows(libpq_url, thread) == ROWS_AFTER["put"]
        finally:
            # Release the caller's locks first if an assertion stopped the test early
            with contextlib.suppress(psycopg.Error):
                saver.conn.rollback()
            delete_caller_row(libpq_url, caller)


@pytest.mark.timeout(60)
async def test_async_saver_leaves_the_callers_open_transaction_alone(libpq_url) -> None:
    async with async_saver_for("connection", libpq_url, None, psycopg.AsyncCursor) as saver:
        await saver.setup()
        caller, thread = str(uuid.uuid4()), str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        try:
            await saver.conn.execute("BEGIN")
            await saver.conn.execute(CALLER_ROW_SQL, (caller,))
            assert await saver.aget_tuple(config(thread)) is None
            await saver.aput(config(thread), checkpoint, {}, versions)
            assert saver.conn.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            await saver.conn.execute("COMMIT")
            assert rows(libpq_url, caller) == (1, 0, 0)
            assert rows(libpq_url, thread) == ROWS_AFTER["put"]
        finally:
            with contextlib.suppress(psycopg.Error):
                await saver.conn.rollback()
            delete_caller_row(libpq_url, caller)


def force_40001(faults: PsycopgFaults, error: str) -> None:
    """Make the next checkpoint row fail with 40001.

    "forced_error" is a plain statement error carrying the code.
    "retry_error" is CockroachDB's own retry error, from force_retry.
    """
    if error == "forced_error":
        faults.fail_next("40001", match="insert into checkpoints (")
    else:
        faults.server_retry_next("insert into checkpoints (", interval="10s")


@pytest.mark.timeout(60)
@pytest.mark.parametrize("error", ["forced_error", "retry_error"])
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
def test_sync_saver_raises_an_error_inside_the_callers_transaction_at_once(
    libpq_url, unsafe_internals, caplog, mode: str, error: str
) -> None:
    # The caller owns the transaction. A retry inside it would keep it open
    # through the backoff and hide the 40001 from the caller's retry logic, so
    # the saver raises it at once, and the caller retries the whole transaction.
    faults = PsycopgFaults(unsafe_internals)
    with sync_saver_for(mode, libpq_url, None, faults.cursor_class()) as saver:
        saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        with (
            caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
            pytest.raises(psycopg.Error) as info,
            saver.conn.transaction(),
        ):
            # The caller's own read reaches it, so CockroachDB cannot retry the
            # transaction by itself
            assert saver.conn.execute("SELECT 1").fetchone() is not None
            force_40001(faults, error)
            saver.put(config(thread), checkpoint, {}, versions)
        assert get_sqlstate(info.value) == "40001"
        # One attempt, so no backoff either
        assert faults.count("insert into checkpoints (") == 1
        assert not any("Retrying" in record.getMessage() for record in caplog.records)
        with saver.conn.transaction():
            saver.put(config(thread), checkpoint, {}, versions)
        assert rows(libpq_url, thread) == ROWS_AFTER["put"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("error", ["forced_error", "retry_error"])
@pytest.mark.parametrize("mode", ["connection", "pipeline"])
async def test_async_saver_raises_an_error_inside_the_callers_transaction_at_once(
    libpq_url, unsafe_internals, caplog, mode: str, error: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with async_saver_for(mode, libpq_url, None, faults.async_cursor_class()) as saver:
        await saver.setup()
        thread = str(uuid.uuid4())
        checkpoint, versions = checkpoint_with_blobs()
        with (
            caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
            pytest.raises(psycopg.Error) as info,
        ):
            async with saver.conn.transaction():
                cur = await saver.conn.execute("SELECT 1")
                assert await cur.fetchone() is not None
                force_40001(faults, error)
                await saver.aput(config(thread), checkpoint, {}, versions)
        assert get_sqlstate(info.value) == "40001"
        assert faults.count("insert into checkpoints (") == 1
        assert not any("Retrying" in record.getMessage() for record in caplog.records)
        async with saver.conn.transaction():
            await saver.aput(config(thread), checkpoint, {}, versions)
        assert rows(libpq_url, thread) == ROWS_AFTER["put"]


def assert_retried_on_a_clean_connection(caplog: pytest.LogCaptureFixture, saver) -> None:
    """The failed transaction was rolled back before its connection was used again.

    psycopg_pool warns "rolling back returned connection" when it gets back a
    connection still inside a transaction, and counts the broken ones it drops.
    """
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("Retrying put checkpoint after SerializationFailure") for m in messages)
    assert not any("rolling back returned connection" in m for m in messages)
    if isinstance(saver.conn, (ConnectionPool, AsyncConnectionPool)):
        assert saver.conn.get_stats().get("returns_bad", 0) == 0


@WRITE_MODES
def test_sync_saver_retries_a_failed_transaction_on_a_clean_connection(
    sync_pool, libpq_url, unsafe_internals, caplog, mode: str
) -> None:
    # 40001 on the checkpoint row, after the blob row of the same transaction
    faults = PsycopgFaults(unsafe_internals)
    with sync_saver_for(mode, libpq_url, sync_pool, faults.cursor_class()) as saver:
        saver.setup()
        checkpoint, versions = checkpoint_with_blobs()
        faults.fail_next("40001", match="insert into checkpoints (")
        with caplog.at_level(logging.WARNING):
            saved = saver.put(config(str(uuid.uuid4())), checkpoint, {}, versions)
        assert faults.injected == ["40001"]
        # The retry ran the whole transaction again
        assert faults.count("insert into checkpoint_blobs") == 2
        assert_retried_on_a_clean_connection(caplog, saver)
        assert saver.get_tuple(saved).checkpoint["channel_values"] == CHANNEL_VALUES


@WRITE_MODES
async def test_async_saver_retries_a_failed_transaction_on_a_clean_connection(
    async_pool, libpq_url, unsafe_internals, caplog, mode: str
) -> None:
    faults = PsycopgFaults(unsafe_internals)
    async with async_saver_for(mode, libpq_url, async_pool, faults.async_cursor_class()) as saver:
        await saver.setup()
        checkpoint, versions = checkpoint_with_blobs()
        faults.fail_next("40001", match="insert into checkpoints (")
        with caplog.at_level(logging.WARNING):
            saved = await saver.aput(config(str(uuid.uuid4())), checkpoint, {}, versions)
        assert faults.injected == ["40001"]
        assert faults.count("insert into checkpoint_blobs") == 2
        assert_retried_on_a_clean_connection(caplog, saver)
        assert (await saver.aget_tuple(saved)).checkpoint["channel_values"] == CHANNEL_VALUES


@pytest.mark.parametrize("phrases", ["default", "none"])
def test_sync_saver_attempt_timeout_rides_out_a_stuck_transaction(
    sync_pool, libpq_url, caplog, phrases: str
) -> None:
    # The put also writes a blob row, so the guard shuts the socket inside an
    # open transaction. The pool drops the broken connection, and the retry
    # runs the whole transaction on a new one.
    saver = CockroachDBSaver(sync_pool(app_name()), retry_policy=with_phrases(STUCK_WRITE, phrases))
    saver.setup()
    thread = str(uuid.uuid4())
    checkpoint, versions = checkpoint_with_blobs()
    saver.put(config(thread), checkpoint, {"step": 1}, versions)
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        hold_row_lock(libpq_url, LOCK_SQL, (thread,), release_after=1.2),
    ):
        saver.put(config(thread), checkpoint, {"step": 2}, versions)
    assert "attempt timeout" in caplog.text
    assert saver.conn.get_stats().get("returns_bad", 0) >= 1
    if phrases == "none":
        assert_retried_for_the_broken_flag(caplog, "put checkpoint")
    saved = saver.get_tuple(config(thread))
    assert saved.metadata["step"] == 2
    assert saved.checkpoint["channel_values"] == CHANNEL_VALUES


@pytest.mark.parametrize("phrases", ["default", "none"])
async def test_async_saver_attempt_timeout_rides_out_a_stuck_transaction(
    async_pool, libpq_url, caplog, phrases: str
) -> None:
    saver = AsyncCockroachDBSaver(
        await async_pool(app_name()), retry_policy=with_phrases(STUCK_WRITE, phrases)
    )
    await saver.setup()
    thread = str(uuid.uuid4())
    checkpoint, versions = checkpoint_with_blobs()
    await saver.aput(config(thread), checkpoint, {"step": 1}, versions)
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        hold_row_lock(libpq_url, LOCK_SQL, (thread,), release_after=1.2),
    ):
        await saver.aput(config(thread), checkpoint, {"step": 2}, versions)
    assert "attempt timeout" in caplog.text
    assert saver.conn.get_stats().get("returns_bad", 0) >= 1
    if phrases == "none":
        assert_retried_for_the_broken_flag(caplog, "put checkpoint")
    saved = await saver.aget_tuple(config(thread))
    assert saved.metadata["step"] == 2
    assert saved.checkpoint["channel_values"] == CHANNEL_VALUES
