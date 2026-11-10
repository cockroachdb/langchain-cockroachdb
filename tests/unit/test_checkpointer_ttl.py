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
from datetime import timedelta
from typing import Any

import pytest
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from psycopg import errors as pg_errors
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver, RetryPolicy
from langchain_cockroachdb.checkpointer import base
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

    def __init__(self, kind: str, saver: Any, reloptions: list[str] | None) -> None:
        self.reloptions = reloptions
        self.reads = 0
        saver._read_blob_ttl = self.read if kind == "sync" else self.aread

    def read(self) -> list[str] | None:
        self.reads += 1
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
    assert refresh_params(used) == [["thread-1", "", [("facts", "1"), ("notes", "2")], SIX_MINUTES]]


@KINDS
async def test_a_refresh_alone_commits_with_the_checkpoint_row(kind: str) -> None:
    saver, used = saver_for(kind)
    Catalog(kind, saver, OLD_V25)
    checkpoint, _ = checkpoint_using_old_blobs()
    # Nothing new: every channel kept its version
    await put(kind, saver, checkpoint, {})
    assert statements(used) == ["BEGIN", base.REFRESH_CHECKPOINT_BLOBS_SQL, CHECKPOINT, "COMMIT"]
    assert refresh_params(used) == [
        ["thread-1", "", [("facts", "1"), ("notes", "2"), ("messages", "3")], SIX_MINUTES]
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
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == 2


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
    await put(kind, saver, checkpoint, new_versions)
    assert catalog.reads == 2


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
    # Another setting the saver cannot use is logged too
    catalog.reloptions = ["ttl='on'", "ttl_expire_after='01:00:00'"]
    await put(kind, saver, checkpoint, new_versions)
    assert len(logged()) == 2
    assert "ttl_expire_after" in logged()[1].getMessage()


@KINDS
async def test_enable_ttl_gives_blob_rows_a_grace_of_a_tenth_of_the_interval(kind: str) -> None:
    saver, used = saver_for(kind)
    if kind == "sync":
        saver.enable_ttl("1 hour", "@daily")
    else:
        await saver.aenable_ttl("1 hour", "@daily")
    sent = [" ".join(statement.split()) for statement in statements(used)]
    assert sent == [
        "ALTER TABLE checkpoints SET ( ttl_expiration_expression = "
        "$$(created_at + '1 hour')$$, ttl_job_cron = '@daily' );",
        "ALTER TABLE checkpoint_blobs SET ( ttl_expiration_expression = "
        "$$(created_at + '1 hour' + CAST('1 hour' AS INTERVAL) / 10)$$, "
        "ttl_job_cron = '@daily' );",
        "ALTER TABLE checkpoint_writes SET ( ttl_expiration_expression = "
        "$$(created_at + '1 hour')$$, ttl_job_cron = '@daily' );",
    ]
    # DDL takes no parameters, so the validated interval and cron are in the text
    assert all(params is None for _, params in executed(used))
    # What enable_ttl writes is what the saver parses
    blob_expression = sent[1].split("$$")[1]
    assert base._parse_blob_ttl([rendered(blob_expression)], 10) == (SIX_MINUTES, None)


@KINDS
@pytest.mark.parametrize("fraction", [0, -1, 2.5, True, "10"])
async def test_enable_ttl_refuses_a_fraction_it_cannot_format(kind: str, fraction: Any) -> None:
    # It goes into the DDL text, so it is checked before any statement runs
    saver, used = saver_for(kind)
    saver.TTL_REFRESH_FRACTION = fraction
    with pytest.raises(ValueError, match="TTL_REFRESH_FRACTION"):
        if kind == "sync":
            saver.enable_ttl("1 hour")
        else:
            await saver.aenable_ttl("1 hour")
    assert used == []


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
    assert "0003" in message
    assert "items" in message and "tags" in message
    assert "incomplete" in message


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
