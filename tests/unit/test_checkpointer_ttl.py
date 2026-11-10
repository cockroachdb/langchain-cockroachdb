"""Checkpointer row-level TTL keeps the blob rows a live thread still uses.

A blob row holds one channel version and is written once, so its created_at
is when that version was first written. TTL expires each row by its own
created_at, so a channel that has not changed since an early step keeps an
old blob row that every newer checkpoint uses, and TTL could delete it from
a live thread. With TTL on, a put refreshes created_at on the blob rows its
checkpoint uses but does not write, when they are older than the refresh
window, a tenth of the interval. enable_ttl gives blob rows that much grace
past the interval. The saver reads the TTL setting from the catalog and
keeps it for TTL_STATE_CACHE_SECONDS.

A walk for DeltaChannel history that stops at a missing ancestor, as when
TTL deleted an old checkpoint, logs a WARNING.

The fakes record the statements; the integration tests check the same
against CockroachDB.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from psycopg import errors as pg_errors
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver, RetryPolicy
from langchain_cockroachdb._psycopg_conn import mark_in_caller_transaction
from langchain_cockroachdb.checkpointer import async_saver as async_saver_module
from langchain_cockroachdb.checkpointer import base
from langchain_cockroachdb.checkpointer import saver as saver_module
from tests.unit.checkpointer_fakes import (
    FakeAsyncConnection,
    FakeAsyncPool,
    FakeConnection,
    FakePool,
    close_sockets,
)

KINDS = pytest.mark.parametrize("kind", ["sync", "async"])
POLICY = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)
CONFIG: Any = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
BLOBS = CockroachDBSaver.UPSERT_CHECKPOINT_BLOBS_SQL
CHECKPOINT = CockroachDBSaver.UPSERT_CHECKPOINTS_SQL
SIX_MINUTES = timedelta(minutes=6)
# The same as SQL gets it: a duration in hours, minutes and seconds, never
# days, which CockroachDB adds by the calendar of the session's time zone
SIX_MINUTES_SQL = "0:06:00.000000"

# checkpoint_blobs' pg_class.reloptions after enable_ttl("1 hour"), as
# CockroachDB v25.4.20 renders them: 0.5 wrote the expression without the
# grace, 0.6 writes it with the grace. v26.3.5 renders them the same and adds
# schema_locked, which it turns on by default.
OLD_V25 = [
    "ttl='on'",
    "ttl_expiration_expression=e'(created_at + \\'1 hour\\')'",
    "ttl_job_cron='@daily'",
]
NEW_V25 = [
    "ttl='on'",
    "ttl_expiration_expression="
    "e'(created_at + \\'1 hour\\' + CAST(\\'1 hour\\' AS INTERVAL) / 10)'",
    "ttl_job_cron='@daily'",
]
OLD_V26 = [*OLD_V25, "schema_locked=true"]
NEW_V26 = [*NEW_V25, "schema_locked=true"]
# What TTL off reads as: no row options on v25.4, schema_locked alone on v26
TTL_OFF = pytest.mark.parametrize(
    "reloptions", [None, [], ["schema_locked=true"]], ids=["v25.4", "empty", "v26"]
)


@pytest.fixture(autouse=True)
def _close_sockets() -> Iterator[None]:
    yield
    close_sockets()


@pytest.fixture(autouse=True)
def _forget_what_was_logged() -> Iterator[None]:
    """Each test starts as a fresh process would, with nothing logged yet."""
    base._LOGGED_TTL_SETTINGS.clear()
    base._LOGGED_TTL_READ_ERRORS.clear()
    yield
    base._LOGGED_TTL_SETTINGS.clear()
    base._LOGGED_TTL_READ_ERRORS.clear()


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The seconds each enable_ttl waited while TTL was paused, without waiting."""
    seconds: list[float] = []

    def wait(duration: float) -> None:
        seconds.append(duration)

    async def await_wait(duration: float) -> None:
        seconds.append(duration)

    monkeypatch.setattr(saver_module, "_sleep", wait)
    monkeypatch.setattr(async_saver_module, "_sleep", await_wait)
    return seconds


def rendered(expression: str) -> str:
    """A ttl_expiration_expression as pg_class.reloptions renders it."""
    return "ttl_expiration_expression=e'" + expression.replace("'", "\\'") + "'"


def old(interval: str) -> str:
    return f"(created_at + '{interval}')"


def new(interval: str) -> str:
    return f"(created_at + '{interval}' + CAST('{interval}' AS INTERVAL) / 10)"


def saver_for(kind: str) -> tuple[Any, list[Any]]:
    """A saver on a fake pool, and the fake connections it has used so far.

    The fake libpq has pipeline mode, so a write with several statements
    sends BEGIN, its statements and COMMIT as one batch.
    """
    saver: Any
    if kind == "sync":
        pool: Any = FakePool()
        saver = CockroachDBSaver(pool, retry_policy=POLICY)
    else:
        pool = FakeAsyncPool()
        saver = AsyncCockroachDBSaver(pool, retry_policy=POLICY)
    saver.supports_pipeline = True
    return saver, pool.handed_out


class Catalog:
    """Stands in for the saver's read of checkpoint_blobs' reloptions, and counts the reads."""

    def __init__(
        self, kind: str, saver: Any, reloptions: list[str] | None, error: Exception | None = None
    ) -> None:
        self.reloptions = reloptions
        # Raised by every read while set
        self.error = error
        self.reads = 0
        saver._read_blob_ttl = self.read if kind == "sync" else self.aread

    def read(self) -> list[str] | None:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return self.reloptions

    async def aread(self) -> list[str] | None:
        return self.read()


def executed(used: list[Any]) -> list[tuple[str, Any]]:
    """Every statement the fake connections ran, with its parameters, in order."""
    return [call for conn in used for call in conn.executed]


def statements(used: list[Any]) -> list[str]:
    return [statement for statement, _ in executed(used)]


def refresh_params(used: list[Any]) -> list[Any]:
    """The parameters of each refresh statement sent."""
    refresh = base.REFRESH_CHECKPOINT_BLOBS_SQL
    return [list(params) for statement, params in executed(used) if statement == refresh]


def checkpoint_using_old_blobs() -> tuple[Any, dict[str, str]]:
    """A checkpoint and its new versions: only "messages" and "count" changed in this put.

    "facts" and "notes" are stored as blobs and kept their versions, so the
    checkpoint uses their old blob rows. "count", "flag" and "name" are
    stored inline in the checkpoint. "cleared" has a version and no value.
    """
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {
        "facts": ["the sky is blue"],
        "notes": {"tone": "brief"},
        "messages": ["hello"],
        "count": 3,
        "flag": None,
        "name": "ada",
    }
    checkpoint["channel_versions"] = {
        "facts": "1",
        "notes": "2",
        "messages": "3",
        "count": "3",
        "flag": "1",
        "name": "1",
        "cleared": "2",
    }
    return checkpoint, {"messages": "3", "count": "3"}


async def put(kind: str, saver: Any, checkpoint: Any, new_versions: dict[str, str]) -> None:
    if kind == "sync":
        saver.put(CONFIG, checkpoint, {}, new_versions)
    else:
        await saver.aput(CONFIG, checkpoint, {}, new_versions)


def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make the savers' monotonic clock read the list's one value."""
    clock = [1000.0]
    monkeypatch.setattr(base, "monotonic", lambda: clock[0])
    return clock


def test_the_refresh_statement_binds_every_value() -> None:
    assert " ".join(base.REFRESH_CHECKPOINT_BLOBS_SQL.split()) == (
        "UPDATE checkpoint_blobs SET created_at = now() "
        "WHERE thread_id = %s AND checkpoint_ns = %s "
        "AND (channel, version) = ANY(%s) "
        "AND created_at < now() - CAST(%s AS INTERVAL)"
    )
    # The catalog read binds the table name too
    assert " ".join(base.SELECT_BLOB_TTL_SQL.split()) == (
        "SELECT reloptions FROM pg_catalog.pg_class WHERE oid = to_regclass(%s)"
    )
    assert CockroachDBSaver.REFRESH_CHECKPOINT_BLOBS_SQL == base.REFRESH_CHECKPOINT_BLOBS_SQL
    assert AsyncCockroachDBSaver.SELECT_BLOB_TTL_SQL == base.SELECT_BLOB_TTL_SQL


def test_the_named_defaults() -> None:
    assert base.TTL_REFRESH_FRACTION == CockroachDBSaver.TTL_REFRESH_FRACTION == 10
    assert base.TTL_STATE_CACHE_SECONDS == AsyncCockroachDBSaver.TTL_STATE_CACHE_SECONDS == 60


@KINDS
async def test_a_put_refreshes_the_blob_rows_it_uses_but_does_not_write(kind: str) -> None:
    saver, used = saver_for(kind)
    Catalog(kind, saver, NEW_V26)
    checkpoint, new_versions = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, new_versions)
    # One transaction: the new blob, the refresh and the checkpoint row
    assert statements(used) == [
        "BEGIN",
        BLOBS,
        base.REFRESH_CHECKPOINT_BLOBS_SQL,
        CHECKPOINT,
        "COMMIT",
    ]
    # Used and not written: not "messages", which this put writes, not the
    # channels stored inline, and not "cleared", which has no value
    assert refresh_params(used) == [
        ["thread-1", "", [("facts", "1"), ("notes", "2")], SIX_MINUTES_SQL]
    ]


@KINDS
async def test_a_refresh_alone_commits_with_the_checkpoint_row(kind: str) -> None:
    saver, used = saver_for(kind)
    Catalog(kind, saver, OLD_V25)
    checkpoint, _ = checkpoint_using_old_blobs()
    # Nothing new: every channel kept its version
    await put(kind, saver, checkpoint, {})
    assert statements(used) == ["BEGIN", base.REFRESH_CHECKPOINT_BLOBS_SQL, CHECKPOINT, "COMMIT"]
    assert refresh_params(used) == [
        ["thread-1", "", [("facts", "1"), ("notes", "2"), ("messages", "3")], SIX_MINUTES_SQL]
    ]


@KINDS
@TTL_OFF
async def test_with_ttl_off_a_put_refreshes_nothing(kind: str, reloptions: Any) -> None:
    saver, used = saver_for(kind)
    catalog = Catalog(kind, saver, reloptions)
    checkpoint, _ = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, {})
    assert catalog.reads == 1
    # One statement, so no transaction, as before puts refreshed blob rows
    assert statements(used) == [CHECKPOINT]


@KINDS
async def test_a_put_that_writes_every_blob_it_uses_does_not_read_the_catalog(kind: str) -> None:
    saver, used = saver_for(kind)
    catalog = Catalog(kind, saver, NEW_V26)
    checkpoint, _ = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, dict(checkpoint["channel_versions"]))
    await put(kind, saver, empty_checkpoint(), {})
    assert catalog.reads == 0
    assert base.REFRESH_CHECKPOINT_BLOBS_SQL not in statements(used)


@KINDS
async def test_the_saver_reads_the_ttl_setting_with_a_bound_table_name(kind: str) -> None:
    saver, used = saver_for(kind)
    checkpoint, new_versions = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, new_versions)
    # Its own operation, before the write. The fakes find no row: no TTL
    assert executed(used)[0] == (base.SELECT_BLOB_TTL_SQL, ("checkpoint_blobs",))
    assert base.REFRESH_CHECKPOINT_BLOBS_SQL not in statements(used)


@pytest.mark.parametrize(
    "reloptions",
    [
        OLD_V25,
        OLD_V26,
        NEW_V25,
        NEW_V26,
        # A literal with no character to escape is plain, with quotes doubled
        ["ttl='on'", "ttl_expiration_expression='(created_at + ''1 hour'')'"],
    ],
    ids=["old-v25.4", "old-v26", "new-v25.4", "new-v26", "plain-literal"],
)
def test_parses_both_expressions_as_both_versions_render_them(reloptions: list[str]) -> None:
    assert base._parse_blob_ttl(reloptions, 10) == (SIX_MINUTES, None)


@pytest.mark.parametrize(
    ("interval", "window"),
    [
        ("1 hour", timedelta(minutes=6)),
        ("2 hours", timedelta(minutes=12)),
        ("90 minutes", timedelta(minutes=9)),
        ("30 days", timedelta(days=3)),
        ("7 days", timedelta(hours=16, minutes=48)),
        ("1 week", timedelta(hours=16, minutes=48)),
        ("1 month", timedelta(days=3)),
        ("1 year", timedelta(days=36)),
        ("3 seconds", timedelta(milliseconds=300)),
        ("1 millisecond", timedelta(microseconds=100)),
        ("1 microsecond", timedelta(0)),
        ("5  days", timedelta(hours=12)),
    ],
)
def test_the_window_is_a_tenth_of_the_interval(interval: str, window: timedelta) -> None:
    # As CockroachDB divides an interval for the grace: 30 days to a month,
    # 12 months to a year
    for expression in (old(interval), new(interval)):
        assert base._parse_blob_ttl([rendered(expression)], 10) == (window, None)


def test_the_window_follows_the_grace_in_the_catalog() -> None:
    grace_of_a_twentieth = "(created_at + '1 hour' + CAST('1 hour' AS INTERVAL) / 20)"
    assert base._parse_blob_ttl([rendered(grace_of_a_twentieth)], 10) == (
        timedelta(minutes=3),
        None,
    )
    # Without a grace in the catalog, the saver's fraction sets the window
    assert base._parse_blob_ttl([rendered(old("1 hour"))], 4) == (timedelta(minutes=15), None)


@TTL_OFF
def test_no_ttl_setting_means_no_refresh(reloptions: Any) -> None:
    assert base._parse_blob_ttl(reloptions, 10) == (None, None)


@pytest.mark.parametrize(
    "setting",
    [
        # An interval enable_ttl refuses
        rendered(old("1 hour 30 minutes")),
        # Two intervals
        rendered("(created_at + '1 hour' + CAST('2 hours' AS INTERVAL) / 10)"),
        rendered("(created_at + '1 hour' + '2 hours'::INTERVAL / 10)"),
        rendered("(created_at + '1 hour' + CAST('1 hour' AS INTERVAL) / 0)"),
        rendered("(updated_at + '1 hour')"),
        "ttl_expiration_expression='expires_at'",
        # Whitespace CockroachDB keeps as written
        rendered("(created_at+'1 hour')"),
    ],
)
def test_a_setting_enable_ttl_did_not_write_means_no_refresh(setting: str) -> None:
    window, foreign = base._parse_blob_ttl(["ttl='on'", setting, "ttl_job_cron='@daily'"], 10)
    assert window is None
    assert foreign == setting


@pytest.mark.parametrize("interval", ["1 hour", "7 days", "30 days"])
def test_parses_the_grace_written_with_a_double_colon_cast(interval: str) -> None:
    # Hand-written, or copied from DDL that writes casts that way
    colons = f"(created_at + '{interval}' + '{interval}'::INTERVAL / 10)"
    cast_form = new(interval)
    assert base._parse_blob_ttl([rendered(colons)], 10) == base._parse_blob_ttl(
        [rendered(cast_form)], 10
    )
    assert base._parse_blob_ttl([rendered(colons)], 10)[0] is not None


def test_ttl_expire_after_is_a_setting_enable_ttl_did_not_write() -> None:
    reloptions = ["ttl='on'", "ttl_expire_after='01:00:00'", "ttl_job_cron='@hourly'"]
    window, foreign = base._parse_blob_ttl(reloptions, 10)
    assert window is None
    assert foreign is not None
    assert "ttl_expire_after='01:00:00'" in foreign


def test_a_window_python_cannot_hold_means_no_refresh() -> None:
    # enable_ttl takes it, but no timedelta can hold a tenth of it
    huge = rendered(old("999999999999 years"))
    assert base._parse_blob_ttl([huge], 10) == (None, huge)
    # Nor can a saver divide by a fraction of 0
    assert base._parse_blob_ttl([rendered(old("1 hour"))], 0) == (None, rendered(old("1 hour")))


@KINDS
async def test_the_saver_keeps_what_it_read_for_the_cache_interval(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = fake_clock(monkeypatch)
    saver, used = saver_for(kind)
    catalog = Catalog(kind, saver, NEW_V26)
    checkpoint, new_versions = checkpoint_using_old_blobs()

    async def refreshes() -> bool:
        before = len(refresh_params(used))
        await put(kind, saver, checkpoint, new_versions)
        return len(refresh_params(used)) > before

    assert await refreshes()
    assert catalog.reads == 1
    # Another process turns TTL off. This saver goes on with what it read
    catalog.reloptions = ["schema_locked=true"]
    clock[0] += saver.TTL_STATE_CACHE_SECONDS - 1
    assert await refreshes()
    assert catalog.reads == 1
    # until the cache interval has passed
    clock[0] += 1
    assert not await refreshes()
    assert catalog.reads == 2


@KINDS
@pytest.mark.parametrize("change", ["enable_ttl", "disable_ttl"])
async def test_enable_and_disable_ttl_make_the_next_put_read_the_catalog(
    kind: str, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_clock(monkeypatch)
    saver, _ = saver_for(kind)
    catalog = Catalog(kind, saver, NEW_V26)
    checkpoint, new_versions = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == 1
    if kind == "sync":
        getattr(saver, change)()
    else:
        await getattr(saver, f"a{change}")()
    # enable_ttl reads the setting itself, to tell whether it changed
    before = catalog.reads
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == before + 1


@KINDS
async def test_a_failed_enable_ttl_also_makes_the_next_put_read_the_catalog(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The statements before the failure took effect, and the failing one may
    # have too, so what the saver read before may be out of date
    fake_clock(monkeypatch)
    saver, _ = saver_for(kind)
    catalog = Catalog(kind, saver, None)
    checkpoint, new_versions = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, new_versions)

    def refuse(statement: str) -> None:
        if "checkpoint_blobs" in statement:
            raise pg_errors.InsufficientPrivilege("permission denied")

    async def arefuse(statement: str) -> None:
        refuse(statement)

    # The TTL change on checkpoint_blobs fails, and is not retried
    pool = saver.conn
    factory = pool.factory
    if kind == "sync":
        pool.factory = lambda: FakeConnection(on_statement=refuse)
    else:
        pool.factory = lambda: FakeAsyncConnection(on_statement=arefuse)
    with pytest.raises(pg_errors.InsufficientPrivilege):
        if kind == "sync":
            saver.enable_ttl("1 hour")
        else:
            await saver.aenable_ttl("1 hour")
    pool.factory = factory
    before = catalog.reads
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == before + 1


@KINDS
async def test_a_ttl_setting_enable_ttl_did_not_write_is_logged_once_at_info(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = fake_clock(monkeypatch)
    saver, used = saver_for(kind)
    foreign = rendered(old("1 hour 30 minutes"))
    catalog = Catalog(kind, saver, ["ttl='on'", foreign])
    checkpoint, new_versions = checkpoint_using_old_blobs()
    caplog.set_level(logging.INFO, logger=base.__name__)

    def logged() -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == base.__name__]

    for _ in range(3):
        await put(kind, saver, checkpoint, new_versions)
        clock[0] += saver.TTL_STATE_CACHE_SECONDS
    assert catalog.reads == 3
    assert refresh_params(used) == []
    [record] = logged()
    assert record.levelno == logging.INFO
    assert "1 hour 30 minutes" in record.getMessage()
    # Once per process: another saver that reads the same setting stays quiet
    other, _ = saver_for(kind)
    Catalog(kind, other, ["ttl='on'", foreign])
    await put(kind, other, checkpoint, new_versions)
    assert len(logged()) == 1
    # Another setting the savers cannot use is logged too
    catalog.reloptions = ["ttl='on'", "ttl_expire_after='01:00:00'"]
    await put(kind, saver, checkpoint, new_versions)
    assert len(logged()) == 2
    assert "ttl_expire_after" in logged()[1].getMessage()


def ddl(used: list[Any]) -> list[str]:
    """The DDL statements sent, with their whitespace made single spaces."""
    return [" ".join(s.split()) for s in statements(used) if s.lstrip().startswith("ALTER")]


async def enable_ttl(kind: str, saver: Any, interval: str, cron: str = "@daily") -> None:
    if kind == "sync":
        saver.enable_ttl(interval, cron)
    else:
        await saver.aenable_ttl(interval, cron)


PAUSED_1_HOUR = [
    "ALTER TABLE checkpoint_blobs SET ( ttl_expiration_expression = "
    "$$(created_at + '1 hour' + CAST('1 hour' AS INTERVAL) / 10)$$, "
    "ttl_job_cron = '@daily', ttl_pause = true );",
    "ALTER TABLE checkpoints SET ( ttl_expiration_expression = "
    "$$(created_at + '1 hour')$$, ttl_job_cron = '@daily', ttl_pause = true );",
    "ALTER TABLE checkpoint_writes SET ( ttl_expiration_expression = "
    "$$(created_at + '1 hour')$$, ttl_job_cron = '@daily', ttl_pause = true );",
]
RESUME = [
    "ALTER TABLE checkpoints RESET (ttl_pause);",
    "ALTER TABLE checkpoint_writes RESET (ttl_pause);",
    "ALTER TABLE checkpoint_blobs RESET (ttl_pause);",
]


@KINDS
async def test_enable_ttl_pauses_waits_refreshes_and_resumes(kind: str, waits: list[float]) -> None:
    saver, used = saver_for(kind)
    await enable_ttl(kind, saver, "1 hour")
    # The settings with TTL paused, checkpoint_blobs first, so blob rows stop
    # expiring before anything else changes; checkpoint_blobs resumes last,
    # so an unpaused checkpoint_blobs means a finished run. The blob rows get
    # a grace of a tenth of the interval
    assert ddl(used) == PAUSED_1_HOUR + RESUME
    calls = executed(used)
    # DDL takes no parameters, so the validated interval and cron are in the text
    assert [params for s, params in calls if s.lstrip().startswith("ALTER")] == [None] * 6
    # It reads the setting first; the fakes find none and no checkpoint, so
    # after the wait it reads one empty page for its refresh
    assert calls[0] == (base.SELECT_BLOB_TTL_SQL, ("checkpoint_blobs",))
    assert calls[4] == (base.SELECT_CHECKPOINT_PAGE_SQL, ["1:00:00.000000", 500])
    assert len(calls) == 8
    # One wait for every saver to read the new setting: 60 seconds, plus 5
    assert waits == [65.0]
    # What enable_ttl writes is what the saver parses
    blob_expression = PAUSED_1_HOUR[0].split("$$")[1]
    assert base._parse_blob_ttl([rendered(blob_expression)], 10) == (SIX_MINUTES, None)


@KINDS
async def test_the_wait_follows_the_saver_cache_interval_and_margin(
    kind: str, waits: list[float]
) -> None:
    saver, _ = saver_for(kind)
    saver.TTL_STATE_CACHE_SECONDS = 7.0
    saver.TTL_PAUSE_MARGIN_SECONDS = 2.0
    await enable_ttl(kind, saver, "1 hour")
    assert waits == [9.0]
    assert base.TTL_PAUSE_MARGIN_SECONDS == CockroachDBSaver.TTL_PAUSE_MARGIN_SECONDS == 5.0


@KINDS
async def test_with_the_same_setting_in_place_enable_ttl_neither_waits_nor_pauses(
    kind: str, waits: list[float]
) -> None:
    saver, used = saver_for(kind)
    refresh = Pass(kind, saver, [[]])
    Catalog(kind, saver, NEW_V26)
    # The same expression, unpaused; only the cron changes
    await enable_ttl(kind, saver, "1 hour", "@weekly")
    assert waits == []
    assert refresh.reads == []
    assert ddl(used) == [
        "ALTER TABLE checkpoints SET ( ttl_expiration_expression = "
        "$$(created_at + '1 hour')$$, ttl_job_cron = '@weekly' );",
        "ALTER TABLE checkpoint_blobs SET ( ttl_expiration_expression = "
        "$$(created_at + '1 hour' + CAST('1 hour' AS INTERVAL) / 10)$$, "
        "ttl_job_cron = '@weekly' );",
        "ALTER TABLE checkpoint_writes SET ( ttl_expiration_expression = "
        "$$(created_at + '1 hour')$$, ttl_job_cron = '@weekly' );",
    ]


@KINDS
async def test_a_paused_setting_is_a_run_that_stopped_partway(
    kind: str, waits: list[float]
) -> None:
    # As enable_ttl leaves it when it stops after pausing: the next run, with
    # the same interval, waits, refreshes and resumes
    saver, used = saver_for(kind)
    refresh = Pass(kind, saver, [[]])
    Catalog(kind, saver, [*NEW_V26, "ttl_pause=true"])
    await enable_ttl(kind, saver, "1 hour")
    assert waits == [65.0]
    assert len(refresh.reads) == 1
    assert ddl(used)[-3:] == RESUME


def test_a_paused_setting_reads_as_paused() -> None:
    assert base._ttl_paused([*NEW_V25, "ttl_pause=true"])
    assert base._ttl_paused([*NEW_V26[:3], "ttl_pause=true", "schema_locked=true"])
    assert not base._ttl_paused(NEW_V26)
    assert not base._ttl_paused(None)
    # The savers refresh while TTL is paused: the expression is the library's
    assert base._parse_blob_ttl([*NEW_V26, "ttl_pause=true"], 10) == (SIX_MINUTES, None)


@KINDS
async def test_puts_refresh_while_ttl_is_paused(kind: str) -> None:
    saver, used = saver_for(kind)
    Catalog(kind, saver, [*NEW_V26, "ttl_pause=true"])
    checkpoint, new_versions = checkpoint_using_old_blobs()
    await put(kind, saver, checkpoint, new_versions)
    assert refresh_params(used) == [
        ["thread-1", "", [("facts", "1"), ("notes", "2")], SIX_MINUTES_SQL]
    ]


@KINDS
@pytest.mark.parametrize(
    ("interval", "stored"),
    [
        ("1 month", "30 days"),
        ("3 months", "90 days"),
        ("1 year", "365 days"),
        ("2 years", "730 days"),
    ],
)
async def test_enable_ttl_stores_months_and_years_as_days(
    kind: str, interval: str, stored: str
) -> None:
    # The calendar adds months by different numbers of days, so a grace
    # could not cover them; days have a fixed length
    saver, used = saver_for(kind)
    await enable_ttl(kind, saver, interval)
    blobs, checkpoints, writes = ddl(used)[:3]
    assert f"$$(created_at + '{stored}' + CAST('{stored}' AS INTERVAL) / 10)$$" in blobs
    assert f"$$(created_at + '{stored}')$$" in checkpoints
    assert f"$$(created_at + '{stored}')$$" in writes


@pytest.mark.parametrize(
    ("interval", "stored", "length", "window"),
    [
        ("1 hour", "1 hour", "1:00:00.000000", "0:06:00.000000"),
        ("90 minutes", "90 minutes", "1:30:00.000000", "0:09:00.000000"),
        ("7 days", "7 days", "168:00:00.000000", "16:48:00.000000"),
        ("1 week", "1 week", "168:00:00.000000", "16:48:00.000000"),
        ("30 days", "30 days", "720:00:00.000000", "72:00:00.000000"),
        ("1 month", "30 days", "720:00:00.000000", "72:00:00.000000"),
        ("1 year", "365 days", "8760:00:00.000000", "876:00:00.000000"),
        ("3 seconds", "3 seconds", "0:00:03.000000", "0:00:00.300000"),
    ],
)
def test_the_plan_binds_the_interval_and_window_as_durations(
    interval: str, stored: str, length: str, window: str
) -> None:
    plan = CockroachDBSaver(FakePool())._ttl_plan(interval, "@daily")
    assert plan.stored == stored
    # As SQL gets them: hours, minutes and seconds, never days
    assert (base._duration(plan.interval), base._duration(plan.window)) == (length, window)
    assert plan.blob_expression == f"(created_at + '{stored}' + CAST('{stored}' AS INTERVAL) / 10)"


@KINDS
@pytest.mark.parametrize("interval", ["1 fortnight", "1 hour; DROP TABLE x"])
async def test_enable_ttl_refuses_an_interval_before_any_statement(
    kind: str, interval: str
) -> None:
    saver, used = saver_for(kind)
    with pytest.raises(ValueError, match="Invalid ttl_interval"):
        await enable_ttl(kind, saver, interval)
    assert used == []


@KINDS
@pytest.mark.parametrize(
    "interval", ["300 years", "106752 days", "999999999999 years", "99999999999999999999 hours"]
)
async def test_enable_ttl_refuses_an_interval_longer_than_cockroachdb_keeps(
    kind: str, interval: str
) -> None:
    # The refresh binds it in hours, which CockroachDB keeps as int64
    # nanoseconds and wraps to a negative interval past about 106,751 days
    saver, used = saver_for(kind)
    with pytest.raises(ValueError) as raised:
        await enable_ttl(kind, saver, interval)
    # The message names what the caller passed, not the days it would store
    assert repr(interval) in str(raised.value)
    assert "106751 days" in str(raised.value)
    assert used == []


def test_the_longest_interval_enable_ttl_takes() -> None:
    plan = CockroachDBSaver(FakePool())._ttl_plan("106751 days", "@daily")
    assert plan.interval == timedelta(days=106751)
    # The longest duration CockroachDB keeps
    assert base._duration(base._LONGEST_DURATION) == "2562047:47:16.854775"


@KINDS
@pytest.mark.parametrize("fraction", [0, -1, 2.5, True, "10"])
async def test_enable_ttl_refuses_a_fraction_it_cannot_format(kind: str, fraction: Any) -> None:
    # It goes into the DDL text, so it is checked before any statement runs
    saver, used = saver_for(kind)
    saver.TTL_REFRESH_FRACTION = fraction
    with pytest.raises(ValueError, match="TTL_REFRESH_FRACTION"):
        await enable_ttl(kind, saver, "1 hour")
    assert used == []


def test_the_refresh_statements_bind_every_value() -> None:
    first = " ".join(base.SELECT_CHECKPOINT_PAGE_SQL.split())
    after = " ".join(base.SELECT_CHECKPOINT_PAGE_AFTER_SQL.split())
    select = (
        "SELECT thread_id, checkpoint_ns, checkpoint_id, created_at, CASE WHEN created_at > "
        "now() - CAST(%s AS INTERVAL) THEN checkpoint -> 'channel_versions' END AS "
        "channel_versions FROM checkpoints "
    )
    order = "ORDER BY thread_id, checkpoint_ns, checkpoint_id LIMIT %s"
    assert first == select + order
    assert after == (
        select + "WHERE (thread_id, checkpoint_ns, checkpoint_id) > (%s, %s, %s) " + order
    )
    assert " ".join(base.REFRESH_USED_BLOBS_SQL.split()) == (
        "UPDATE checkpoint_blobs AS b SET created_at = now() FROM unnest( "
        "CAST(%s AS STRING[]), CAST(%s AS STRING[]), CAST(%s AS STRING[]), "
        "CAST(%s AS STRING[]), CAST(%s AS TIMESTAMPTZ[]) ) "
        "AS u (thread_id, checkpoint_ns, channel, version, newest) "
        "WHERE b.thread_id = u.thread_id AND b.checkpoint_ns = u.checkpoint_ns "
        "AND b.channel = u.channel AND b.version = u.version "
        "AND b.created_at < u.newest - CAST(%s AS INTERVAL)"
    )
    saver = CockroachDBSaver(FakePool())
    plan = saver._ttl_plan("1 hour", "@daily")
    assert saver._checkpoint_page(None, plan, 500) == (
        base.SELECT_CHECKPOINT_PAGE_SQL,
        ["1:00:00.000000", 500],
    )
    assert saver._checkpoint_page(("t", "", "c9"), plan, 500) == (
        base.SELECT_CHECKPOINT_PAGE_AFTER_SQL,
        ["1:00:00.000000", "t", "", "c9", 500],
    )


T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def page_row(
    thread_id: str, checkpoint_id: str, written: datetime, versions: dict[str, str] | None
) -> dict[str, Any]:
    """A row of a refresh page. ``versions`` is None for a checkpoint outside the interval."""
    return {
        "thread_id": thread_id,
        "checkpoint_ns": "",
        "checkpoint_id": checkpoint_id,
        "created_at": written,
        "channel_versions": versions,
    }


def test_a_page_yields_each_used_blob_row_with_its_newest_checkpoint() -> None:
    page = [
        page_row("t1", "c1", at(0), None),
        # The page's order is the key's, not always the time's
        page_row("t1", "c3", at(20), {"facts": "1", "messages": "3"}),
        page_row("t1", "c2", at(10), {"facts": "1", "messages": "2"}),
        page_row("t2", "c1", at(5), {"facts": "1"}),
    ]
    assert CockroachDBSaver._used_blobs(page) == [
        # Used by c2 and c3, so it takes the newest of them; c1 is outside
        ("t1", "", "facts", "1", at(20)),
        ("t1", "", "messages", "2", at(10)),
        ("t1", "", "messages", "3", at(20)),
        ("t2", "", "facts", "1", at(5)),
    ]
    assert CockroachDBSaver._page_end(page, None) == ("t2", "", "c1")
    assert CockroachDBSaver._page_end([], ("t", "", "c")) == ("t", "", "c")


@pytest.mark.parametrize(("size", "sizes"), [(100, [100, 100, 50]), (0, [1] * 250)])
def test_each_update_of_the_refresh_changes_at_most_100_rows(size: int, sizes: list[int]) -> None:
    saver = CockroachDBSaver(FakePool())
    saver.TTL_REFRESH_UPDATE_SIZE = size
    plan = saver._ttl_plan("1 hour", "@daily")
    used = [("t", "", f"ch{i:03}", "1", at(i)) for i in range(250)]
    batches = saver._refresh_batches(used, plan)
    # Sizes below 1 count as 1
    assert [len(params[0]) for params in batches] == sizes
    # Five arrays, one per column of the rows, then the window as a duration
    first = batches[0]
    assert first[0][0] == "t" and first[1][0] == ""
    assert first[2][0] == "ch000" and first[3][0] == "1" and first[4][0] == at(0)
    assert first[5] == "0:06:00.000000"
    assert base.TTL_REFRESH_UPDATE_SIZE == CockroachDBSaver.TTL_REFRESH_UPDATE_SIZE == 100


def numbered_rows(count: int, start: int = 0) -> list[dict[str, Any]]:
    """``count`` checkpoints inside the interval, each using a blob row of its own."""
    return [
        page_row("t", f"c{i:04}", at(i), {f"ch{i:04}": "1"}) for i in range(start, start + count)
    ]


class Pass:
    """Stands in for the reads and UPDATEs of the refresh in enable_ttl.

    Each UPDATE reports that it changed every row it was given.
    """

    def __init__(self, kind: str, saver: Any, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = list(pages)
        self.reads: list[Any] = []
        self.updates: list[list[Any]] = []
        if kind == "sync":
            saver._read_checkpoint_page = self.read
            saver._refresh_blob_rows = self.update
        else:
            saver._read_checkpoint_page = self.aread
            saver._refresh_blob_rows = self.aupdate

    def read(self, after: Any, plan: Any, limit: int) -> list[dict[str, Any]]:
        self.reads.append((after, limit))
        return self.pages.pop(0)

    async def aread(self, after: Any, plan: Any, limit: int) -> list[dict[str, Any]]:
        return self.read(after, plan, limit)

    def update(self, params: list[Any]) -> int:
        self.updates.append(params)
        return len(params[0])

    async def aupdate(self, params: list[Any]) -> int:
        return self.update(params)


@KINDS
async def test_enable_ttl_refreshes_page_by_page_until_a_page_is_short(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    saver, _ = saver_for(kind)
    saver.TTL_REFRESH_BATCH_SIZE = 3
    refresh = Pass(kind, saver, [numbered_rows(3), numbered_rows(3, 3), numbered_rows(1, 6)])
    caplog.set_level(logging.INFO, logger=base.__name__)
    await enable_ttl(kind, saver, "1 hour")
    # Each page starts after the last key of the one before
    assert refresh.reads == [(None, 3), (("t", "", "c0002"), 3), (("t", "", "c0005"), 3)]
    assert [len(params[0]) for params in refresh.updates] == [3, 3, 1]
    started, finished = [
        r.getMessage()
        for r in caplog.records
        if r.name == base.__name__ and "orward" in r.getMessage()
    ]
    assert "1 hour" in started and "3 at a time" in started
    assert "Blob rows brought forward: 7." in finished


@KINDS
async def test_a_page_whose_rows_exceed_an_update_takes_several(kind: str) -> None:
    saver, _ = saver_for(kind)
    refresh = Pass(kind, saver, [numbered_rows(250)])
    await enable_ttl(kind, saver, "1 hour")
    assert [len(params[0]) for params in refresh.updates] == [100, 100, 50]


@KINDS
@pytest.mark.parametrize("size", [0, -1])
async def test_refresh_batch_sizes_below_one_count_as_one(kind: str, size: int) -> None:
    saver, _ = saver_for(kind)
    saver.TTL_REFRESH_BATCH_SIZE = size
    refresh = Pass(kind, saver, [numbered_rows(1), []])
    await enable_ttl(kind, saver, "1 hour")
    assert [limit for _, limit in refresh.reads] == [1, 1]


@KINDS
@pytest.mark.parametrize(
    ("found", "interval", "refreshes"),
    [
        # The same expression: puts have refreshed blob rows all along
        (NEW_V26, "1 hour", False),
        # Lengthened: checkpoints the old setting expired but its job has not
        # deleted yet come back inside the interval, with stale blob rows
        (NEW_V26, "2 hours", True),
        (NEW_V26, "1 day", True),
        # Shortened: rows within the old window may be outside the new one
        (NEW_V26, "30 minutes", True),
        # A setting from 0.5 had no grace; one not written by enable_ttl, or none
        (OLD_V26, "1 hour", True),
        (OLD_V26, "2 hours", True),
        (["ttl='on'", rendered(old("1 hour 30 minutes"))], "2 hours", True),
        (["schema_locked=true"], "1 hour", True),
        (None, "1 hour", True),
    ],
)
async def test_enable_ttl_refreshes_unless_the_same_setting_is_in_place(
    kind: str, found: Any, interval: str, refreshes: bool
) -> None:
    saver, _ = saver_for(kind)
    refresh = Pass(kind, saver, [[]])
    Catalog(kind, saver, found)
    await enable_ttl(kind, saver, interval, "@weekly")
    assert len(refresh.reads) == int(refreshes)


@KINDS
async def test_the_refresh_reads_and_updates_with_bound_values_under_the_ddl_timeout(
    kind: str,
) -> None:
    # Maintenance, so a short attempt_timeout for puts does not cut it short
    saver, used = saver_for(kind)
    saver.retry_policy = RetryPolicy(
        initial_backoff=0.01, max_backoff=0.05, attempt_timeout=1.5, ddl_attempt_timeout=90.0
    )
    timeouts: list[Any] = []
    cursor = saver._cursor

    def recording_cursor(**kwargs: Any) -> Any:
        timeouts.append(kwargs.get("timeout"))
        return cursor(**kwargs)

    saver._cursor = recording_cursor
    plan = saver._ttl_plan("1 hour", "@daily")
    params = saver._refresh_batches([("t", "", "facts", "1", at(0))], plan)[0]
    if kind == "sync":
        page = saver._read_checkpoint_page(("t", "", "c"), plan, 50)
        changed = saver._refresh_blob_rows(params)
    else:
        page = await saver._read_checkpoint_page(("t", "", "c"), plan, 50)
        changed = await saver._refresh_blob_rows(params)
    # The fakes find no checkpoint and change no row
    assert (page, changed) == ([], 0)
    assert executed(used) == [
        (base.SELECT_CHECKPOINT_PAGE_AFTER_SQL, ["1:00:00.000000", "t", "", "c", 50]),
        (base.REFRESH_USED_BLOBS_SQL, params),
    ]
    assert timeouts == [90.0, 90.0]


class Stops:
    """A refresh page read that fails, as when the run is stopped partway."""

    def __init__(self, kind: str, saver: Any) -> None:
        saver._read_checkpoint_page = self.read if kind == "sync" else self.aread

    def read(self, *args: Any) -> Any:
        raise RuntimeError("stopped")

    async def aread(self, *args: Any) -> Any:
        return self.read()


@KINDS
async def test_a_refresh_that_stops_leaves_ttl_paused_for_the_next_run_to_finish(
    kind: str,
) -> None:
    saver, used = saver_for(kind)
    Stops(kind, saver)
    with pytest.raises(RuntimeError, match="stopped"):
        await enable_ttl(kind, saver, "1 hour")
    # TTL stays paused, which deletes nothing: no table was unpaused
    assert ddl(used) == PAUSED_1_HOUR
    # The next run finds the paused setting, so it refreshes and resumes
    Catalog(kind, saver, [*NEW_V26[:3], "ttl_pause=true"])
    refresh = Pass(kind, saver, [[]])
    await enable_ttl(kind, saver, "1 hour")
    assert len(refresh.reads) == 1
    assert ddl(used) == PAUSED_1_HOUR * 2 + RESUME


@KINDS
async def test_a_read_refused_by_the_database_leaves_puts_working_without_a_refresh(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A missing privilege, for example: the policy does not retry it
    clock = fake_clock(monkeypatch)
    saver, used = saver_for(kind)
    catalog = Catalog(kind, saver, NEW_V26, pg_errors.InsufficientPrivilege("permission denied"))
    checkpoint, new_versions = checkpoint_using_old_blobs()
    caplog.set_level(logging.WARNING, logger=base.__name__)
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == 1
    assert refresh_params(used) == []
    [record] = [r for r in caplog.records if r.name == base.__name__]
    assert record.levelno == logging.WARNING
    assert "42501" in record.getMessage()
    # "No refresh" is kept for the cache interval, like any answer
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == 1
    # Then the saver reads again; the WARNING is logged once per process
    clock[0] += saver.TTL_STATE_CACHE_SECONDS
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == 2
    other, _ = saver_for(kind)
    Catalog(kind, other, NEW_V26, pg_errors.InsufficientPrivilege("permission denied"))
    await put(kind, other, checkpoint, new_versions)
    assert len([r for r in caplog.records if r.name == base.__name__]) == 1
    # Once a read works, puts refresh again
    catalog.error = None
    clock[0] += saver.TTL_STATE_CACHE_SECONDS
    await put(kind, saver, checkpoint, new_versions)
    assert len(refresh_params(used)) == 1


def caller_transaction_error() -> Exception:
    error = pg_errors.InsufficientPrivilege("permission denied")
    mark_in_caller_transaction(error)
    return error


@KINDS
@pytest.mark.parametrize(
    "error",
    [
        # Retried by the policy, and raised once it gives up
        pg_errors.SerializationFailure("restart transaction"),
        # No SQLSTATE: the connection, not the database
        psycopg.OperationalError("connection lost"),
        # Inside a transaction the caller opened, which the error aborted
        caller_transaction_error(),
    ],
    ids=["retryable", "no-sqlstate", "caller-transaction"],
)
async def test_other_read_errors_fail_the_put(kind: str, error: Exception) -> None:
    saver, used = saver_for(kind)
    Catalog(kind, saver, NEW_V26, error)
    checkpoint, new_versions = checkpoint_using_old_blobs()
    with pytest.raises(type(error)):
        await put(kind, saver, checkpoint, new_versions)
    assert statements(used) == []


def _config(checkpoint_id: str) -> Any:
    return {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }


def _id(config: Any) -> str:
    return config["configurable"]["checkpoint_id"]


def _chain(length: int, seed_at: int | None = None) -> dict[str, CheckpointTuple]:
    """``length`` checkpoints, each the parent of the next, each with pending writes.

    The checkpoint at ``seed_at`` stores a value for both channels.
    """
    chain: dict[str, CheckpointTuple] = {}
    for step in range(length):
        checkpoint = empty_checkpoint()
        if step == seed_at:
            checkpoint["channel_values"] = {"items": ["seed"], "tags": ["seed"]}
        chain[f"{step:04}"] = CheckpointTuple(
            config=_config(f"{step:04}"),
            checkpoint=checkpoint,
            metadata={"step": step},
            parent_config=_config(f"{step - 1:04}") if step else None,
            pending_writes=[("task-a", "items", [f"i{step}"]), ("task-a", "tags", [f"t{step}"])],
        )
    return chain


def _chain_page(chain: dict[str, CheckpointTuple], start: Any, limit: int) -> list[Any]:
    """What the chain query returns: the start row, then parents up to ``limit`` rows.

    It stops at an ancestor that is missing, and has no rows if the start is.
    """
    if _id(start) not in chain:
        return []
    page = [chain[_id(start)]]
    while len(page) < limit:
        parent = page[-1].parent_config
        if parent is None or _id(parent) not in chain:
            break
        page.append(chain[_id(parent)])
    return page


def walker(
    kind: str, chain: dict[str, CheckpointTuple], monkeypatch: pytest.MonkeyPatch, page_size: int
) -> Any:
    """A saver whose history walk reads ``chain`` in pages of ``page_size``."""
    saver: Any
    if kind == "sync":
        saver = CockroachDBSaver(ConnectionPool("", open=False))
        monkeypatch.setattr(saver, "get_tuple", lambda config: chain.get(_id(config)))
        monkeypatch.setattr(
            saver, "_fetch_chain_page", lambda start, limit: _chain_page(chain, start, limit)
        )
    else:
        saver = AsyncCockroachDBSaver(AsyncConnectionPool("", open=False))

        async def aget_tuple(config: Any) -> Any:
            return chain.get(_id(config))

        async def fetch_chain_page(start: Any, limit: int) -> list[Any]:
            return _chain_page(chain, start, limit)

        monkeypatch.setattr(saver, "aget_tuple", aget_tuple)
        monkeypatch.setattr(saver, "_fetch_chain_page", fetch_chain_page)
    saver.DELTA_HISTORY_FIRST_PAGE = page_size
    saver.DELTA_HISTORY_PAGE_SIZE = page_size
    return saver


async def history(kind: str, saver: Any, checkpoint_id: str) -> Any:
    channels = ["items", "tags"]
    if kind == "sync":
        return saver.get_delta_channel_history(config=_config(checkpoint_id), channels=channels)
    return await saver.aget_delta_channel_history(config=_config(checkpoint_id), channels=channels)


def warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@KINDS
@pytest.mark.parametrize("page_size", [256, 2])
async def test_a_walk_cut_by_a_missing_ancestor_logs_a_warning(
    kind: str, page_size: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chain = _chain(8)
    # As row-level TTL deletes an old checkpoint
    del chain["0003"]
    saver = walker(kind, chain, monkeypatch, page_size)
    caplog.set_level(logging.WARNING)
    result = await history(kind, saver, "0007")
    # The result is what the walk found before the gap, as before
    assert [write[2] for write in result["tags"]["writes"]] == [["t4"], ["t5"], ["t6"]]
    assert "seed" not in result["tags"]
    [message] = warnings(caplog)
    assert "thread-1" in message
    assert "checkpoint_ns ''" in message
    assert "0003" in message
    assert "items" in message and "tags" in message
    assert "incomplete" in message


@KINDS
async def test_a_cut_chain_is_logged_once_per_saver_and_missing_checkpoint(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chain = _chain(8)
    del chain["0003"]
    saver = walker(kind, chain, monkeypatch, 256)
    caplog.set_level(logging.WARNING)
    # LangGraph rebuilds the state on every get_state and resumed run
    for _ in range(3):
        await history(kind, saver, "0007")
    assert len(warnings(caplog)) == 1
    # Another missing checkpoint is logged too
    del chain["0005"]
    await history(kind, saver, "0007")
    assert len(warnings(caplog)) == 2
    assert "0005" in warnings(caplog)[1]
    # Another saver logs what it finds
    await history(kind, walker(kind, chain, monkeypatch, 256), "0007")
    assert len(warnings(caplog)) == 3


@KINDS
async def test_a_saver_makes_its_record_of_cut_chains_on_first_use(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # So a subclass whose __init__ never calls the base one still works
    chain = _chain(4)
    del chain["0001"]
    saver = walker(kind, chain, monkeypatch, 256)
    assert "_cut_chains_logged" not in vars(saver)
    caplog.set_level(logging.WARNING)
    await history(kind, saver, "0003")
    await history(kind, saver, "0003")
    assert len(warnings(caplog)) == 1


@KINDS
async def test_a_saver_forgets_the_cut_chains_it_logged_once_it_holds_too_many(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(base, "_CUT_CHAINS_REMEMBERED", 1)
    first, second = _chain(4), _chain(4)
    del first["0001"]
    del second["0002"]
    chains = {"first": first, "second": second}
    current = ["first"]

    def chain_now() -> dict[str, CheckpointTuple]:
        return chains[current[0]]

    saver = walker(kind, first, monkeypatch, 256)
    if kind == "sync":
        monkeypatch.setattr(saver, "get_tuple", lambda config: chain_now().get(_id(config)))
        monkeypatch.setattr(
            saver, "_fetch_chain_page", lambda start, limit: _chain_page(chain_now(), start, limit)
        )
    else:

        async def aget_tuple(config: Any) -> Any:
            return chain_now().get(_id(config))

        async def fetch_chain_page(start: Any, limit: int) -> list[Any]:
            return _chain_page(chain_now(), start, limit)

        monkeypatch.setattr(saver, "aget_tuple", aget_tuple)
        monkeypatch.setattr(saver, "_fetch_chain_page", fetch_chain_page)
    caplog.set_level(logging.WARNING)
    for name in ["first", "second", "first"]:
        current[0] = name
        await history(kind, saver, "0003")
    # It remembers one cut at a time, so the first is logged again
    assert len(warnings(caplog)) == 3


@KINDS
async def test_a_missing_parent_of_the_target_logs_a_warning(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chain = _chain(4)
    del chain["0002"]
    saver = walker(kind, chain, monkeypatch, 256)
    caplog.set_level(logging.WARNING)
    result = await history(kind, saver, "0003")
    assert result["tags"]["writes"] == []
    [message] = warnings(caplog)
    assert "thread-1" in message and "0002" in message


@KINDS
async def test_no_warning_at_the_root_or_once_every_channel_has_its_seed(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    # The walk reaches the root
    result = await history(kind, walker(kind, _chain(6), monkeypatch, 2), "0005")
    assert len(result["tags"]["writes"]) == 5
    # Every channel finds its seed before the gap
    chain = _chain(6, seed_at=3)
    del chain["0001"]
    result = await history(kind, walker(kind, chain, monkeypatch, 2), "0005")
    assert result["items"]["seed"] == ["seed"]
    assert warnings(caplog) == []
