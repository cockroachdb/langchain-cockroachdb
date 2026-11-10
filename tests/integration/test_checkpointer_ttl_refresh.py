"""Checkpointer row-level TTL keeps the blob rows a live thread still uses.

TTL expires each row by its own created_at. A blob row holds one channel
version and is written once, so a channel that has not changed since an
early step keeps an old blob row that every newer checkpoint uses. These
tests age blob rows with SQL, as time would, and check that a put refreshes
the ones its checkpoint uses, that enable_ttl gives blob rows a grace, and
that the saver reads the TTL setting from the catalog. They delete expired
rows the way the TTL job would, so they need not wait for it. Each test has
a database of its own, since TTL is a table setting.

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


def age(url: str, channel: str, interval: str) -> None:
    """Make the thread's blob rows for ``channel`` look ``interval`` old, as time would."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "UPDATE checkpoint_blobs SET created_at = now() - CAST(%s AS INTERVAL) "
            "WHERE thread_id = %s AND channel = %s",
            [interval, THREAD, channel],
        )


def blob_age(url: str, channel: str) -> timedelta:
    """How old the thread's one blob row for ``channel`` is."""
    with psycopg.connect(url, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT now() - created_at FROM checkpoint_blobs WHERE thread_id = %s AND channel = %s",
            [THREAD, channel],
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
        [message] = warnings(caplog)
        assert THREAD in message and ids[0] in message and "items" in message


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
            [message] = warnings(caplog)
            assert THREAD in message and ids[0] in message and "items" in message


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
