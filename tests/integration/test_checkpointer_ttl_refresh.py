"""Checkpointer row-level TTL keeps the blob rows a live thread still uses.

TTL expires each row by its own created_at. A blob row holds one channel
version and is written once, so a channel that has not changed since an
early step keeps an old blob row that every newer checkpoint uses. These
tests age rows with SQL, as time would, and check that a put refreshes the
blob rows its checkpoint uses, that enable_ttl gives blob rows a grace,
stores months and years as days, pauses TTL while every saver learns the
new setting, and refreshes once the blob rows of threads idle when it runs,
and that the saver reads the TTL setting from the catalog. They delete expired rows the way the TTL job would, so they
need not wait for it. Each test has a database of its own, since TTL is a
table setting.

One slow test, skipped with -k "not ttl_expiration", keeps a thread active
past a short interval while CockroachDB's TTL job runs.
"""

from __future__ import annotations

import logging
import operator
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from typing import Annotated, Any, TypedDict
from urllib.parse import urlsplit

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver
from langchain_cockroachdb.checkpointer import base

MODES = pytest.mark.parametrize("mode", ["connection", "pipeline", "pool"])
THREAD = "thread-1"
LATEST: Any = {"configurable": {"thread_id": THREAD, "checkpoint_ns": ""}}
FACTS = ["the sky is blue"]
NOTES = {"tone": "brief"}


@contextmanager
def fresh_database(libpq_url: str) -> Iterator[str]:
    """A new database with the checkpoint tables set up, dropped afterwards."""
    database = f"ttl_refresh_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    url = urlsplit(libpq_url)._replace(path=f"/{database}").geturl()
    try:
        with CockroachDBSaver.from_conn_string(url) as saver:
            saver.setup()
        yield url
    finally:
        with psycopg.connect(libpq_url, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


@contextmanager
def sync_saver(url: str, mode: str = "connection") -> Iterator[CockroachDBSaver]:
    """A saver on one connection, on one in pipeline mode, or on a pool."""
    if mode == "pool":
        with ConnectionPool(
            url, max_size=4, kwargs={"autocommit": True, "row_factory": dict_row}
        ) as pool:
            yield CockroachDBSaver(pool)
    else:
        with CockroachDBSaver.from_conn_string(url, pipeline=mode == "pipeline") as saver:
            yield saver


@asynccontextmanager
async def async_saver(url: str, mode: str = "connection") -> AsyncIterator[AsyncCockroachDBSaver]:
    """An async saver on one connection, on one in pipeline mode, or on a pool."""
    if mode == "pool":
        async with AsyncConnectionPool(
            url, max_size=4, kwargs={"autocommit": True, "row_factory": dict_row}
        ) as pool:
            yield AsyncCockroachDBSaver(pool)
    else:
        async with AsyncCockroachDBSaver.from_conn_string(
            url, pipeline=mode == "pipeline"
        ) as saver:
            yield saver


class Thread:
    """One thread's checkpoints, each holding every channel's value, as LangGraph writes them."""

    def __init__(self, saver: Any, thread_id: str = THREAD) -> None:
        self.saver = saver
        self.config: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        self.values: dict[str, Any] = {}
        self.versions: dict[str, str] = {}

    def _next(self, changed: dict[str, Any]) -> tuple[Any, Any, dict[str, str]]:
        """The config to put with, the next checkpoint, and its new versions."""
        new_versions = {
            channel: self.saver.get_next_version(self.versions.get(channel), None)
            for channel in changed
        }
        self.values.update(changed)
        self.versions.update(new_versions)
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = dict(self.values)
        checkpoint["channel_versions"] = dict(self.versions)
        return self.config, checkpoint, new_versions

    def put(self, **changed: Any) -> None:
        """Put the next checkpoint, with ``changed`` channels at new versions."""
        config, checkpoint, new_versions = self._next(changed)
        self.config = self.saver.put(config, checkpoint, {}, new_versions)

    async def aput(self, **changed: Any) -> None:
        config, checkpoint, new_versions = self._next(changed)
        self.config = await self.saver.aput(config, checkpoint, {}, new_versions)


def age(url: str, channel: str, interval: str, thread_id: str = THREAD) -> None:
    """Make a thread's blob rows for ``channel`` look ``interval`` old, as time would."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "UPDATE checkpoint_blobs SET created_at = now() - CAST(%s AS INTERVAL) "
            "WHERE thread_id = %s AND channel = %s",
            [interval, thread_id, channel],
        )


def age_thread(url: str, thread_id: str, interval: str) -> None:
    """Make all of a thread's rows look ``interval`` old: a thread idle that long."""
    with psycopg.connect(url, autocommit=True) as conn:
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            conn.execute(
                f"UPDATE {table} SET created_at = now() - CAST(%s AS INTERVAL) "
                "WHERE thread_id = %s",
                [interval, thread_id],
            )


def blob_age(url: str, channel: str, thread_id: str = THREAD) -> timedelta:
    """How old a thread's one blob row for ``channel`` is."""
    with psycopg.connect(url, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT now() - created_at FROM checkpoint_blobs WHERE thread_id = %s AND channel = %s",
            [thread_id, channel],
        ).fetchall()
    [(found,)] = rows
    return found


def expire_like_the_ttl_job(url: str, interval: str) -> None:
    """Delete the blob rows older than ``interval``, as the TTL job would once it runs."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM checkpoint_blobs WHERE created_at < now() - CAST(%s AS INTERVAL)",
            [interval],
        )


def ttl_options(url: str, table: str) -> list[str]:
    """The TTL storage parameters of ``table``, as pg_class.reloptions renders them."""
    with psycopg.connect(url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT reloptions FROM pg_catalog.pg_class WHERE oid = to_regclass(%s)", [table]
        ).fetchone()
    assert row is not None
    return [option for option in row[0] or [] if option.startswith("ttl")]


def assert_refreshed_and_kept(url: str, values: dict[str, Any]) -> None:
    """After a put that changed only messages, with facts aged two hours and notes one minute."""
    # Without the refresh, the TTL job deleted facts from the latest state
    assert values.get("facts") == FACTS
    assert values.get("notes") == NOTES
    assert values["messages"] == ["hello", "hi"]
    # facts was older than the refresh window of six minutes, so the put
    # refreshed it. notes was not, and it still expires after the checkpoint
    assert blob_age(url, "facts") < timedelta(minutes=1)
    assert timedelta(minutes=1) <= blob_age(url, "notes") < timedelta(minutes=6)


@MODES
def test_a_put_refreshes_old_blob_rows_its_checkpoint_still_uses(libpq_url: str, mode: str) -> None:
    with fresh_database(libpq_url) as url, sync_saver(url, mode) as saver:
        saver.enable_ttl("1 hour")
        thread = Thread(saver)
        thread.put(facts=FACTS, notes=NOTES, messages=["hello"])
        age(url, "facts", "2 hours")
        age(url, "notes", "1 minute")
        # The next turn changes only messages
        thread.put(messages=["hello", "hi"])
        expire_like_the_ttl_job(url, "1 hour")
        latest = saver.get_tuple(LATEST)
        assert latest is not None
        assert_refreshed_and_kept(url, latest.checkpoint["channel_values"])


@MODES
async def test_async_a_put_refreshes_old_blob_rows_its_checkpoint_still_uses(
    libpq_url: str, mode: str
) -> None:
    with fresh_database(libpq_url) as url:
        async with async_saver(url, mode) as saver:
            await saver.aenable_ttl("1 hour")
            thread = Thread(saver)
            await thread.aput(facts=FACTS, notes=NOTES, messages=["hello"])
            age(url, "facts", "2 hours")
            age(url, "notes", "1 minute")
            await thread.aput(messages=["hello", "hi"])
            expire_like_the_ttl_job(url, "1 hour")
            latest = await saver.aget_tuple(LATEST)
            assert latest is not None
            assert_refreshed_and_kept(url, latest.checkpoint["channel_values"])


def test_with_ttl_off_a_put_refreshes_nothing(libpq_url: str) -> None:
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        thread = Thread(saver)
        thread.put(facts=FACTS, messages=["hello"])
        age(url, "facts", "2 hours")
        thread.put(messages=["hello", "hi"])
        assert blob_age(url, "facts") >= timedelta(hours=2)
        # With TTL on, the next put refreshes facts
        saver.enable_ttl("1 hour")
        thread.put(messages=["hello", "hi", "on"])
        assert blob_age(url, "facts") < timedelta(minutes=1)
        # Turned off through this saver, the next put knows at once
        saver.disable_ttl()
        age(url, "facts", "2 hours")
        thread.put(messages=["hello", "hi", "on", "off"])
        assert blob_age(url, "facts") >= timedelta(hours=2)


async def test_async_with_ttl_off_a_put_refreshes_nothing(libpq_url: str) -> None:
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver:
            thread = Thread(saver)
            await thread.aput(facts=FACTS, messages=["hello"])
            age(url, "facts", "2 hours")
            await thread.aput(messages=["hello", "hi"])
            assert blob_age(url, "facts") >= timedelta(hours=2)
            await saver.aenable_ttl("1 hour")
            await thread.aput(messages=["hello", "hi", "on"])
            assert blob_age(url, "facts") < timedelta(minutes=1)
            await saver.adisable_ttl()
            age(url, "facts", "2 hours")
            await thread.aput(messages=["hello", "hi", "on", "off"])
            assert blob_age(url, "facts") >= timedelta(hours=2)


# What enable_ttl("1 hour", "@daily") leaves in pg_class.reloptions: the same
# on CockroachDB v25.4 and v26
INTERVAL_ONLY = [
    "ttl='on'",
    "ttl_expiration_expression=e'(created_at + \\'1 hour\\')'",
    "ttl_job_cron='@daily'",
]
WITH_GRACE = [
    "ttl='on'",
    "ttl_expiration_expression="
    "e'(created_at + \\'1 hour\\' + CAST(\\'1 hour\\' AS INTERVAL) / 10)'",
    "ttl_job_cron='@daily'",
]


def assert_grace_written(url: str) -> None:
    assert ttl_options(url, "checkpoints") == INTERVAL_ONLY
    assert ttl_options(url, "checkpoint_writes") == INTERVAL_ONLY
    assert ttl_options(url, "checkpoint_blobs") == WITH_GRACE
    # The window the savers parse from it
    with psycopg.connect(url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT reloptions FROM pg_catalog.pg_class WHERE oid = to_regclass(%s)",
            ["checkpoint_blobs"],
        ).fetchone()
    assert row is not None
    assert base._parse_blob_ttl(row[0], 10) == (timedelta(minutes=6), None)


def test_enable_ttl_writes_a_grace_for_blob_rows(libpq_url: str) -> None:
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        saver.enable_ttl("1 hour", "@daily")
        assert_grace_written(url)


async def test_async_enable_ttl_writes_a_grace_for_blob_rows(libpq_url: str) -> None:
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver:
            await saver.aenable_ttl("1 hour", "@daily")
            assert_grace_written(url)


def test_another_processs_enable_ttl_is_seen_after_the_cache_interval(
    libpq_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(base, "monotonic", lambda: clock[0])
    with fresh_database(libpq_url) as url, sync_saver(url) as saver, sync_saver(url) as other:
        thread = Thread(saver)
        thread.put(facts=FACTS, messages=["hello"])
        # This put uses facts without writing it, so the saver reads the TTL setting: off
        thread.put(messages=["hello", "hi"])
        other.enable_ttl("1 hour")
        age(url, "facts", "2 hours")
        thread.put(messages=["hello", "hi", "again"])
        # Within the cache interval the saver still takes TTL to be off
        assert blob_age(url, "facts") >= timedelta(hours=2)
        clock[0] += saver.TTL_STATE_CACHE_SECONDS
        thread.put(messages=["hello", "hi", "again", "more"])
        assert blob_age(url, "facts") < timedelta(minutes=1)


async def test_async_another_processs_enable_ttl_is_seen_after_the_cache_interval(
    libpq_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(base, "monotonic", lambda: clock[0])
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver, async_saver(url) as other:
            thread = Thread(saver)
            await thread.aput(facts=FACTS, messages=["hello"])
            await thread.aput(messages=["hello", "hi"])
            await other.aenable_ttl("1 hour")
            age(url, "facts", "2 hours")
            await thread.aput(messages=["hello", "hi", "again"])
            assert blob_age(url, "facts") >= timedelta(hours=2)
            clock[0] += saver.TTL_STATE_CACHE_SECONDS
            await thread.aput(messages=["hello", "hi", "again", "more"])
            assert blob_age(url, "facts") < timedelta(minutes=1)


def write_idle_threads(url: str) -> None:
    """Three threads written while TTL is off and idle since, as time has aged them.

    "live" wrote its checkpoints just now, but its facts two hours ago.
    "fresh" wrote its facts a minute ago. "gone" wrote everything three
    hours ago, longer ago than the interval of one hour.
    """
    with CockroachDBSaver.from_conn_string(url) as writer:
        for thread_id in ("gone", "live", "fresh"):
            thread = Thread(writer, thread_id)
            thread.put(facts=FACTS, messages=["hello"])
            thread.put(messages=["hello", "hi"])
    age(url, "facts", "2 hours", thread_id="live")
    age(url, "facts", "1 minute", thread_id="fresh")
    age_thread(url, "gone", "3 hours")


def refresh_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == base.__name__ and "orward" in r.getMessage()
    ]


def assert_idle_threads_kept(url: str, caplog: pytest.LogCaptureFixture) -> None:
    """enable_ttl("1 hour") refreshed what the checkpoints inside the hour use, and only that."""
    # Older than the refresh window of six minutes, and used by a checkpoint
    # inside the interval: brought forward
    assert blob_age(url, "facts", "live") < timedelta(minutes=1)
    # Younger than the window: left alone, and still expires after its checkpoint
    assert timedelta(minutes=1) <= blob_age(url, "facts", "fresh") < timedelta(minutes=6)
    # A thread idle longer than the interval expires as a whole
    assert blob_age(url, "facts", "gone") >= timedelta(hours=3)
    started, finished = refresh_logs(caplog)
    assert "1 hour" in started
    assert finished.endswith("Blob rows brought forward: 1.")
    # Without the refresh, the TTL job's first run deleted facts from the live thread
    expire_like_the_ttl_job(url, "1 hour")
    with CockroachDBSaver.from_conn_string(url) as reader:
        live = reader.get_tuple({"configurable": {"thread_id": "live", "checkpoint_ns": ""}})
    assert live is not None
    assert live.checkpoint["channel_values"].get("facts") == FACTS


@MODES
def test_enable_ttl_keeps_the_blob_rows_of_threads_idle_when_it_runs(
    libpq_url: str, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        write_idle_threads(url)
        with sync_saver(url, mode) as saver:
            # Two checkpoints a page, so the six take several pages
            saver.TTL_REFRESH_BATCH_SIZE = 2
            saver.enable_ttl("1 hour")
        assert_idle_threads_kept(url, caplog)


@MODES
async def test_async_enable_ttl_keeps_the_blob_rows_of_threads_idle_when_it_runs(
    libpq_url: str, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        write_idle_threads(url)
        async with async_saver(url, mode) as saver:
            saver.TTL_REFRESH_BATCH_SIZE = 2
            await saver.aenable_ttl("1 hour")
        assert_idle_threads_kept(url, caplog)


def test_enable_ttl_refreshes_once_per_setting(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        # An empty table
        saver.enable_ttl("1 hour")
        started, finished = refresh_logs(caplog)
        assert finished.endswith("Blob rows brought forward: 0.")
        # The same setting, with another cron: nothing to refresh
        caplog.clear()
        saver.enable_ttl("1 hour", "@weekly")
        assert refresh_logs(caplog) == []
        # Any other interval refreshes again, a longer one too
        saver.enable_ttl("2 hours")
        assert len(refresh_logs(caplog)) == 2
        saver.enable_ttl("30 minutes")
        assert len(refresh_logs(caplog)) == 4


async def test_async_enable_ttl_refreshes_once_per_setting(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver:
            await saver.aenable_ttl("1 hour")
            started, finished = refresh_logs(caplog)
            assert finished.endswith("Blob rows brought forward: 0.")
            caplog.clear()
            await saver.aenable_ttl("1 hour", "@weekly")
            assert refresh_logs(caplog) == []
            await saver.aenable_ttl("2 hours")
            assert len(refresh_logs(caplog)) == 2
            await saver.aenable_ttl("30 minutes")
            assert len(refresh_logs(caplog)) == 4


PROFILE = {"name": "Ada", "plan": "pro"}


def write_six_steps(url: str) -> None:
    """One thread of six steps, the newest five minutes ago and one each ten minutes before.

    Its messages change every step, so each step writes a blob row for
    them. Its profile is set on the first step and never changes, so every
    checkpoint uses that one row.
    """
    steps = []
    with CockroachDBSaver.from_conn_string(url) as writer:
        thread = Thread(writer, "six")
        for step in range(1, 7):
            changed: dict[str, Any] = {"messages": [f"m{i}" for i in range(1, step + 1)]}
            if step == 1:
                changed["profile"] = PROFILE
            thread.put(**changed)
            steps.append((thread.config["configurable"]["checkpoint_id"], dict(thread.versions)))
    with psycopg.connect(url, autocommit=True) as conn:
        for step, (checkpoint_id, versions) in enumerate(steps, start=1):
            ago = f"{65 - 10 * step} minutes"
            conn.execute(
                "UPDATE checkpoints SET created_at = now() - CAST(%s AS INTERVAL) "
                "WHERE thread_id = 'six' AND checkpoint_id = %s",
                [ago, checkpoint_id],
            )
            conn.execute(
                "UPDATE checkpoint_blobs SET created_at = now() - CAST(%s AS INTERVAL) "
                "WHERE thread_id = 'six' AND channel = 'messages' AND version = %s",
                [ago, versions["messages"]],
            )
        conn.execute(
            "UPDATE checkpoint_blobs SET created_at = now() - INTERVAL '55 minutes' "
            "WHERE thread_id = 'six' AND channel = 'profile'"
        )


def blob_times(url: str) -> dict[tuple[str, str], Any]:
    with psycopg.connect(url, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT channel, version, created_at FROM checkpoint_blobs WHERE thread_id = 'six'"
        ).fetchall()
    return {(channel, version): written for channel, version, written in rows}


def database_now(url: str) -> Any:
    with psycopg.connect(url, autocommit=True) as conn:
        row = conn.execute("SELECT now()").fetchone()
    assert row is not None
    return row[0]


def assert_only_the_profile_was_refreshed(
    url: str,
    caplog: pytest.LogCaptureFixture,
    before: dict[tuple[str, str], Any],
    started: Any,
    finished: Any,
) -> dict[tuple[str, str], Any]:
    """After enable_ttl("1 hour") on the six steps, with two checkpoints a page."""
    after = blob_times(url)
    # Only the profile: each messages row is used by the one checkpoint that
    # wrote it, so it is within the window of its newest checkpoint
    changed = {key for key in before if after[key] != before[key]}
    assert {channel for channel, _ in changed} == {"profile"}
    [profile] = changed
    # Brought forward to the time of the pass, never back, and once though
    # every page holds a checkpoint that uses it
    assert started <= after[profile] <= finished
    assert after[profile] > before[profile]
    assert refresh_logs(caplog)[-1].endswith("Blob rows brought forward: 1.")
    return after


@MODES
def test_enable_ttl_rewrites_only_blob_rows_left_unchanged_too_long(
    libpq_url: str, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        write_six_steps(url)
        before = blob_times(url)
        assert len(before) == 7
        with sync_saver(url, mode) as saver:
            saver.TTL_REFRESH_BATCH_SIZE = 2
            started = database_now(url)
            saver.enable_ttl("1 hour")
            after = assert_only_the_profile_was_refreshed(
                url, caplog, before, started, database_now(url)
            )
            # Run again, the pass changes nothing more
            saver._refresh_used_blobs(saver._ttl_plan("1 hour", "@daily"))
            assert refresh_logs(caplog)[-1].endswith("Blob rows brought forward: 0.")
            assert blob_times(url) == after


async def test_async_enable_ttl_rewrites_only_blob_rows_left_unchanged_too_long(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        write_six_steps(url)
        before = blob_times(url)
        async with async_saver(url) as saver:
            saver.TTL_REFRESH_BATCH_SIZE = 2
            started = database_now(url)
            await saver.aenable_ttl("1 hour")
            after = assert_only_the_profile_was_refreshed(
                url, caplog, before, started, database_now(url)
            )
            await saver._refresh_used_blobs(saver._ttl_plan("1 hour", "@daily"))
            assert refresh_logs(caplog)[-1].endswith("Blob rows brought forward: 0.")
            assert blob_times(url) == after


def test_a_longer_interval_refreshes_rows_the_shorter_one_left_stale(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Lengthening runs the pass too.

    With a one-day interval the pass leaves alone a thread idle for three
    days: its rows have expired, but the TTL job, which runs on its cron,
    has not deleted them yet. A thirty-day interval brings the thread back
    inside, and the pass brings forward its profile, set sixty days ago.
    """
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        with CockroachDBSaver.from_conn_string(url) as writer:
            Thread(writer).put(profile=PROFILE, messages=["hello"])
        age_thread(url, THREAD, "3 days")
        age(url, "profile", "60 days")
        with sync_saver(url) as saver:
            saver.enable_ttl("1 day")
            assert blob_age(url, "profile") >= timedelta(days=60)
            saver.enable_ttl("30 days")
        assert blob_age(url, "profile") < timedelta(minutes=1)
        # Written with the checkpoint, so within the window of it
        assert blob_age(url, "messages") >= timedelta(days=3)
        assert refresh_logs(caplog)[-1].endswith("Blob rows brought forward: 1.")


def test_enable_ttl_refreshes_more_rows_than_one_update_takes(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    # 150 channels set once and left unchanged: two UPDATEs, of 100 and 50
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        with CockroachDBSaver.from_conn_string(url) as writer:
            Thread(writer).put(**{f"ch{i:03}": [i] for i in range(150)})
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(
                "UPDATE checkpoint_blobs SET created_at = now() - INTERVAL '2 hours' "
                "WHERE thread_id = %s",
                [THREAD],
            )
        with sync_saver(url) as saver:
            saver.enable_ttl("1 hour")
        assert refresh_logs(caplog)[-1].endswith("Blob rows brought forward: 150.")
        with psycopg.connect(url, autocommit=True) as conn:
            row = conn.execute(
                "SELECT count(*) FROM checkpoint_blobs "
                "WHERE thread_id = %s AND created_at > now() - INTERVAL '1 minute'",
                [THREAD],
            ).fetchone()
        assert row is not None and row[0] == 150


def test_enable_ttl_stores_a_month_as_30_days(libpq_url: str) -> None:
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        saver.enable_ttl("1 month", "@daily")
        days = "e'(created_at + \\'30 days\\')'"
        assert ttl_options(url, "checkpoints")[1] == f"ttl_expiration_expression={days}"
        assert ttl_options(url, "checkpoint_writes")[1] == f"ttl_expiration_expression={days}"
        assert ttl_options(url, "checkpoint_blobs")[1] == (
            "ttl_expiration_expression="
            "e'(created_at + \\'30 days\\' + CAST(\\'30 days\\' AS INTERVAL) / 10)'"
        )
        # The refresh window is three days
        thread = Thread(saver)
        thread.put(facts=FACTS, notes=NOTES, messages=["hello"])
        age(url, "facts", "40 days")
        age(url, "notes", "2 days")
        thread.put(messages=["hello", "hi"])
        assert blob_age(url, "facts") < timedelta(minutes=1)
        assert timedelta(days=2) <= blob_age(url, "notes") < timedelta(days=3)


def test_a_ttl_setting_written_by_0_5_still_refreshes(libpq_url: str) -> None:
    # What 0.5's enable_ttl("1 hour") wrote: no grace on blob rows
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        with psycopg.connect(url, autocommit=True) as conn:
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                conn.execute(
                    f"ALTER TABLE {table} SET (ttl_expiration_expression = "
                    "$$(created_at + '1 hour')$$, ttl_job_cron = '@daily')"
                )
        thread = Thread(saver)
        thread.put(facts=FACTS, notes=NOTES, messages=["hello"])
        age(url, "facts", "2 hours")
        age(url, "notes", "1 minute")
        thread.put(messages=["hello", "hi"])
        # The window parsed from it is six minutes
        assert blob_age(url, "facts") < timedelta(minutes=1)
        assert timedelta(minutes=1) <= blob_age(url, "notes") < timedelta(minutes=6)


def make_idle(url: str) -> None:
    """Age the thread "idle" as if idle for 20 hours, inside an interval of one day.

    Its profile is ten days old; its messages and turn rows were written
    with its newest checkpoint.
    """
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "UPDATE checkpoints SET created_at = now() - INTERVAL '20 hours' "
            "WHERE thread_id = 'idle'"
        )
        conn.execute(
            "UPDATE checkpoint_blobs SET created_at = now() - INTERVAL '20 hours' "
            "WHERE thread_id = 'idle'"
        )
        conn.execute(
            "UPDATE checkpoint_blobs SET created_at = now() - INTERVAL '10 days' "
            "WHERE thread_id = 'idle' AND channel = 'profile'"
        )


def expiry_margins(url: str, thread_id: str) -> dict[str, timedelta]:
    """How long after the thread's latest checkpoint each blob row it uses expires, with TTL 1 day."""
    with psycopg.connect(url, autocommit=True) as conn:
        rows = conn.execute(
            "WITH latest AS (SELECT thread_id, checkpoint_ns, checkpoint, created_at "
            "FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id DESC LIMIT 1) "
            "SELECT b.channel, (b.created_at + INTERVAL '1 day' + INTERVAL '1 day' / 10) "
            "- (l.created_at + INTERVAL '1 day') "
            "FROM latest AS l "
            "CROSS JOIN LATERAL jsonb_each_text(l.checkpoint -> 'channel_versions') "
            "AS v (channel, version) "
            "JOIN checkpoint_blobs AS b ON b.thread_id = l.thread_id "
            "AND b.checkpoint_ns = l.checkpoint_ns AND b.channel = v.channel "
            "AND b.version = v.version",
            [thread_id],
        ).fetchall()
    return {channel: margin for channel, margin in rows}


def test_a_thread_written_while_the_pass_runs_keeps_its_rows(libpq_url: str) -> None:
    """The case the pause closes.

    Another saver read "TTL off" before enable_ttl. Once the pass has read
    the idle thread's checkpoints, it writes a step on that thread that
    changes neither its profile nor its messages. By then it has read the
    paused setting, so it refreshes them. Without the pause it refreshed
    nothing, and the messages row, which the pass left alone, expired 17
    hours 36 minutes before the new checkpoint.
    """
    with fresh_database(libpq_url) as url, sync_saver(url) as saver, sync_saver(url) as other:
        thread = Thread(other, "idle")
        thread.put(profile=PROFILE, messages=["m1"], turn=["t1"])
        make_idle(url)
        assert other._ttl_refresh_window() is None
        read_page = saver._read_checkpoint_page

        def put_after_the_page_is_read(*args: Any) -> Any:
            page = read_page(*args)
            if args[0] is None:
                thread.put(turn=["t1", "t2"])
            return page

        saver._read_checkpoint_page = put_after_the_page_is_read  # type: ignore[method-assign]
        saver.enable_ttl("1 day")
        margins = expiry_margins(url, "idle")
        assert set(margins) == {"profile", "messages", "turn"}
        assert all(margin >= timedelta(0) for margin in margins.values()), margins


async def test_async_a_thread_written_while_the_pass_runs_keeps_its_rows(libpq_url: str) -> None:
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver, async_saver(url) as other:
            thread = Thread(other, "idle")
            await thread.aput(profile=PROFILE, messages=["m1"], turn=["t1"])
            make_idle(url)
            assert await other._ttl_refresh_window() is None
            read_page = saver._read_checkpoint_page

            async def put_after_the_page_is_read(*args: Any) -> Any:
                page = await read_page(*args)
                if args[0] is None:
                    await thread.aput(turn=["t1", "t2"])
                return page

            saver._read_checkpoint_page = put_after_the_page_is_read  # type: ignore[method-assign]
            await saver.aenable_ttl("1 day")
            margins = expiry_margins(url, "idle")
            assert set(margins) == {"profile", "messages", "turn"}
            assert all(margin >= timedelta(0) for margin in margins.values()), margins


def pause_as_a_stopped_run_leaves_it(url: str, saver: Any, interval: str, cron: str) -> None:
    """Write the settings with TTL paused, as enable_ttl does before it waits."""
    with psycopg.connect(url, autocommit=True) as conn:
        for statement in saver._ttl_plan(interval, cron).paused:
            conn.execute(statement)


def paused_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "is paused by enable_ttl" in r.getMessage()
    ]


def test_enable_ttl_finishes_a_run_that_stopped_while_ttl_was_paused(
    libpq_url: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    # As in a fresh process, which has logged no pause yet
    monkeypatch.setattr(base, "_LOGGED_TTL_PAUSED", set())
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        pause_as_a_stopped_run_leaves_it(url, saver, "1 hour", "@daily")
        assert "ttl_pause=true" in ttl_options(url, "checkpoint_blobs")
        thread = Thread(saver)
        thread.put(facts=FACTS, messages=["hello"])
        age(url, "facts", "2 hours")
        # Puts refresh while TTL is paused, and the saver says TTL is paused
        thread.put(messages=["hello", "hi"])
        assert blob_age(url, "facts") < timedelta(minutes=1)
        [warning] = paused_warnings(caplog)
        assert "run enable_ttl again" in warning
        # The same interval again: the paused setting is a run that stopped,
        # so this run refreshes and resumes TTL
        saver.enable_ttl("1 hour")
        assert len(refresh_logs(caplog)) == 2
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert "ttl_pause=true" not in ttl_options(url, table)
        assert_grace_written(url)


async def test_async_enable_ttl_finishes_a_run_that_stopped_while_ttl_was_paused(
    libpq_url: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    monkeypatch.setattr(base, "_LOGGED_TTL_PAUSED", set())
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver:
            pause_as_a_stopped_run_leaves_it(url, saver, "1 hour", "@daily")
            thread = Thread(saver)
            await thread.aput(facts=FACTS, messages=["hello"])
            age(url, "facts", "2 hours")
            await thread.aput(messages=["hello", "hi"])
            assert blob_age(url, "facts") < timedelta(minutes=1)
            [warning] = paused_warnings(caplog)
            assert "run enable_ttl again" in warning
            await saver.aenable_ttl("1 hour")
            assert len(refresh_logs(caplog)) == 2
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert "ttl_pause=true" not in ttl_options(url, table)


def resume_left(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "changed while this run ran" in r.getMessage()
    ]


def test_enable_ttl_leaves_the_resume_to_a_run_that_wrote_another_setting(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        refresh = saver._refresh_used_blobs

        def another_run_pauses_meanwhile(plan: Any) -> None:
            refresh(plan)
            # Another run, with another interval, writes its own setting paused
            pause_as_a_stopped_run_leaves_it(url, saver, "2 hours", "@daily")

        saver._refresh_used_blobs = another_run_pauses_meanwhile  # type: ignore[method-assign]
        saver.enable_ttl("1 hour")
        # This run left TTL paused for the other run, whose setting is in place
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            options = ttl_options(url, table)
            assert "ttl_pause=true" in options
            assert "2 hours" in " ".join(options)
        [message] = resume_left(caplog)
        assert "'2 hours'" in message
        # The other run finishes and resumes TTL
        del saver._refresh_used_blobs
        saver.enable_ttl("2 hours")
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert "ttl_pause=true" not in ttl_options(url, table)
        assert len(resume_left(caplog)) == 1


async def test_async_enable_ttl_leaves_the_resume_to_a_run_that_wrote_another_setting(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=base.__name__)
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver:
            refresh = saver._refresh_used_blobs

            async def another_run_pauses_meanwhile(plan: Any) -> None:
                await refresh(plan)
                pause_as_a_stopped_run_leaves_it(url, saver, "2 hours", "@daily")

            saver._refresh_used_blobs = another_run_pauses_meanwhile  # type: ignore[method-assign]
            await saver.aenable_ttl("1 hour")
            assert "ttl_pause=true" in ttl_options(url, "checkpoint_blobs")
            [message] = resume_left(caplog)
            assert "'2 hours'" in message
            del saver._refresh_used_blobs
            await saver.aenable_ttl("2 hours")
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                assert "ttl_pause=true" not in ttl_options(url, table)


def canary_rows(url: str) -> int:
    with psycopg.connect(url, autocommit=True) as conn:
        row = conn.execute("SELECT count(*) FROM canary").fetchone()
    assert row is not None
    return int(row[0])


def test_ttl_expiration_deletes_nothing_while_ttl_is_paused(libpq_url: str) -> None:
    """CockroachDB's TTL job keeps expired rows while enable_ttl has TTL paused.

    A canary table with TTL running on the same cron shows when the job has
    run: twice over, once its expired row is gone, and again for a second.
    """
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        Thread(saver).put(facts=FACTS, messages=["hello"])
        age_thread(url, THREAD, "2 hours")
        pause_as_a_stopped_run_leaves_it(url, saver, "1 second", "* * * * *")
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(
                "CREATE TABLE canary (id INT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL)"
            )
            conn.execute(
                "ALTER TABLE canary SET (ttl_expiration_expression = "
                "$$(created_at + '1 second')$$, ttl_job_cron = '* * * * *')"
            )
        deadline = time.monotonic() + 300
        for round_ in range(2):
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute("INSERT INTO canary VALUES (%s, now() - INTERVAL '2 hours')", [round_])
            while canary_rows(url):
                assert time.monotonic() < deadline, "the TTL job did not run within 5 minutes"
                time.sleep(5)
        with psycopg.connect(url, autocommit=True) as conn:
            for table in ("checkpoints", "checkpoint_blobs"):
                row = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE thread_id = %s", [THREAD]
                ).fetchone()
                assert row is not None and row[0] > 0, f"the TTL job deleted from {table}"
        # The next enable_ttl finishes the run and resumes TTL
        saver.enable_ttl("1 second", "* * * * *")
        assert "ttl_pause=true" not in ttl_options(url, "checkpoint_blobs")


def delta_chain() -> list[tuple[Any, dict[str, str], list[tuple[str, Any]]]]:
    """Three checkpoints, their new versions and their pending writes.

    The first holds a snapshot of "items"; the others hold only its writes,
    as LangGraph's DeltaChannel stores them between snapshots.
    """
    root = empty_checkpoint()
    root["channel_values"] = {"items": ["seed"]}
    root["channel_versions"] = {"items": "1"}
    return [
        (root, {"items": "1"}, [("items", ["a"])]),
        (empty_checkpoint(), {}, [("items", ["b"])]),
        (empty_checkpoint(), {}, []),
    ]


def delete_checkpoint(url: str, checkpoint_id: str) -> None:
    """Delete one checkpoint row, as TTL deletes an old one."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
            [THREAD, checkpoint_id],
        )


def warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_delta_history_walk_cut_by_a_missing_ancestor_logs_a_warning(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    with fresh_database(libpq_url) as url, sync_saver(url) as saver:
        config: Any = LATEST
        ids = []
        for checkpoint, new_versions, writes in delta_chain():
            config = saver.put(config, checkpoint, {}, new_versions)
            if writes:
                saver.put_writes(config, writes, task_id="task-1")
            ids.append(checkpoint["id"])
        history = saver.get_delta_channel_history(config=config, channels=["items"])
        assert history["items"]["seed"] == ["seed"]
        assert warnings(caplog) == []
        delete_checkpoint(url, ids[0])
        history = saver.get_delta_channel_history(config=config, channels=["items"])
        # The result is what the walk found before the gap, as before
        assert [write[2] for write in history["items"]["writes"]] == [["b"]]
        assert "seed" not in history["items"]
        # Each load of the state walks again; the saver logs the gap once
        saver.get_delta_channel_history(config=config, channels=["items"])
        [message] = warnings(caplog)
        assert THREAD in message and ids[0] in message and "items" in message
        assert "checkpoint_ns ''" in message


async def test_async_a_delta_history_walk_cut_by_a_missing_ancestor_logs_a_warning(
    libpq_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    with fresh_database(libpq_url) as url:
        async with async_saver(url) as saver:
            config: Any = LATEST
            ids = []
            for checkpoint, new_versions, writes in delta_chain():
                config = await saver.aput(config, checkpoint, {}, new_versions)
                if writes:
                    await saver.aput_writes(config, writes, task_id="task-1")
                ids.append(checkpoint["id"])
            history = await saver.aget_delta_channel_history(config=config, channels=["items"])
            assert history["items"]["seed"] == ["seed"]
            assert warnings(caplog) == []
            delete_checkpoint(url, ids[0])
            history = await saver.aget_delta_channel_history(config=config, channels=["items"])
            assert [write[2] for write in history["items"]["writes"]] == [["b"]]
            assert "seed" not in history["items"]
            await saver.aget_delta_channel_history(config=config, channels=["items"])
            [message] = warnings(caplog)
            assert THREAD in message and ids[0] in message and "items" in message
            assert "checkpoint_ns ''" in message


class Profile(TypedDict, total=False):
    profile: dict[str, str]
    turns: Annotated[list[int], operator.add]


def blob_rows(url: str, thread_id: str) -> int:
    with psycopg.connect(url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT count(*) FROM checkpoint_blobs WHERE thread_id = %s", [thread_id]
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_ttl_expiration_keeps_every_channel_of_a_thread_active_past_the_interval(
    libpq_url: str,
) -> None:
    """A graph keeps running past the TTL interval while CockroachDB's TTL job runs.

    Its profile is set on the first turn and never changes, so every later
    checkpoint uses that first blob row. An idle thread written just after
    it shows when the TTL job has run: its rows are gone. The live thread's
    id sorts first, so the job reaches its rows before the idle thread's.
    """
    pytest.importorskip("langgraph.graph")
    from langgraph.graph import START, StateGraph

    def respond(state: Profile) -> dict[str, Any]:
        return {"turns": [len(state.get("turns", []))]}

    builder = StateGraph(Profile)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    profile = {"name": "Ada", "plan": "pro"}
    live: Any = {"configurable": {"thread_id": f"a-live-{uuid.uuid4()}"}}
    idle: Any = {"configurable": {"thread_id": f"z-idle-{uuid.uuid4()}"}}
    with fresh_database(libpq_url) as url, sync_saver(url, "pool") as saver:
        graph = builder.compile(checkpointer=saver)
        # A refresh window of three seconds; the job runs every minute
        saver.enable_ttl("30 seconds", cron="* * * * *")
        graph.invoke({"profile": profile, "turns": []}, live)
        graph.invoke({"profile": profile, "turns": []}, idle)
        turns = 1
        # The job can take a few minutes to start
        deadline = time.monotonic() + 300
        while blob_rows(url, idle["configurable"]["thread_id"]):
            assert time.monotonic() < deadline, (
                "CockroachDB's TTL job did not delete the idle thread's rows within 5 minutes"
            )
            time.sleep(2)
            graph.invoke({"turns": []}, live)
            turns += 1
        graph.invoke({"turns": []}, live)
        turns += 1
        values = graph.get_state(live).values
        assert values.get("profile") == profile
        assert values["turns"] == list(range(turns))
        saver.disable_ttl()
