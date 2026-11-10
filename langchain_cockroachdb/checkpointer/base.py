"""Base class for CockroachDB checkpoint savers.

Contains SQL constants, migrations, and shared serialization logic.
Adapted from langgraph-checkpoint-postgres for CockroachDB compatibility:
- Replaces CREATE INDEX CONCURRENTLY with CREATE INDEX (CockroachDB runs
  index creation as non-blocking background jobs by default)
- Uses separate queries instead of correlated subqueries for performance
- Same table schema as PostgresSaver for potential migration compatibility
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import monotonic
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg.types.json import Jsonb

# Imported here too, where the savers have always found it
from langchain_cockroachdb._connection import connection_options  # noqa: F401
from langchain_cockroachdb._psycopg_conn import in_caller_transaction
from langchain_cockroachdb.retry import RetryPolicy, _describe, get_sqlstate

logger = logging.getLogger(__name__)

MetadataInput = dict[str, Any] | None

# Row-level TTL. A blob row holds one channel version and is written once,
# so its created_at is when that version was first written, and every later
# checkpoint whose channel kept that version uses it. With TTL on, a put
# brings created_at forward on the blob rows its checkpoint uses but does
# not write, once they are older than the refresh window: the TTL interval
# divided by TTL_REFRESH_FRACTION. enable_ttl gives blob rows that window as
# a grace past the interval, so a blob row expires no earlier than the
# newest checkpoint that uses it. Before it changes the setting, enable_ttl
# also brings forward, once, the blob rows that checkpoints inside the
# interval use, so threads idle when TTL is turned on keep theirs too.
TTL_REFRESH_FRACTION = 10
# Seconds a saver uses what it read about TTL from the catalog. After that,
# the next put that needs it reads it again, so a change another process
# makes is seen within this time. The saver's own enable_ttl and disable_ttl
# make it read again at once.
TTL_STATE_CACHE_SECONDS = 60.0
# Checkpoints that one statement of the refresh in enable_ttl reads
TTL_REFRESH_BATCH_SIZE = 500
# Blob rows that one UPDATE of the refresh in enable_ttl changes at most
TTL_REFRESH_UPDATE_SIZE = 100

# Schema migrations. Position in list = version number.
# CockroachDB differences from PostgresSaver:
# - No CONCURRENTLY keyword (CockroachDB CREATE INDEX is non-blocking by default)
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);""",
    "ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP not null;",
    # No-op migration to keep version numbers in sync with PostgresSaver
    "SELECT 1;",
    # CockroachDB: regular CREATE INDEX (non-blocking by default via online schema changes)
    """
    CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
    """,
    """ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';""",
    # Add created_at columns for TTL support (always added, TTL enabled separately)
    """ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();""",
    """ALTER TABLE checkpoint_blobs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();""",
    """ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();""",
]

# The TTL expression enable_ttl writes on checkpoint_blobs: blob rows expire
# the refresh window after the interval. CockroachDB cannot type an untyped
# literal divided by a number, hence the CAST. It keeps the expression as
# written, and the savers parse it back from the catalog, so it stays on one
# line.
BLOB_TTL_EXPRESSION = "(created_at + '{interval}' + CAST('{interval}' AS INTERVAL) / {fraction})"

# SQL templates for enabling TTL on checkpoint tables.
# Uses ttl_expiration_expression (recommended) to avoid full table rewrites.
# DDL takes no bound parameters, so the validated interval and cron are
# formatted in, and TTL_REFRESH_FRACTION as the fraction.
ENABLE_TTL_SQL = [
    """ALTER TABLE checkpoints SET (
        ttl_expiration_expression = $$(created_at + '{interval}')$$,
        ttl_job_cron = '{cron}'
    );""",
    """ALTER TABLE checkpoint_blobs SET (
        ttl_expiration_expression = $$"""
    + BLOB_TTL_EXPRESSION
    + """$$,
        ttl_job_cron = '{cron}'
    );""",
    """ALTER TABLE checkpoint_writes SET (
        ttl_expiration_expression = $$(created_at + '{interval}')$$,
        ttl_job_cron = '{cron}'
    );""",
]

DISABLE_TTL_SQL = [
    "ALTER TABLE checkpoints RESET (ttl);",
    "ALTER TABLE checkpoint_blobs RESET (ttl);",
    "ALTER TABLE checkpoint_writes RESET (ttl);",
]

# Lightweight checkpoint query -- no correlated subqueries.
# Blobs and writes are fetched separately for performance.
SELECT_SQL = """
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata
from checkpoints """

# Fetch blobs for a single checkpoint using its channel_versions keys/values.
# Params: thread_id, checkpoint_ns, then pairs of (channel, version) via ANY.
SELECT_BLOBS_SQL = """
select bl.channel, bl.type, bl.blob
from checkpoint_blobs bl
where bl.thread_id = %s
    and bl.checkpoint_ns = %s
    and (bl.channel, bl.version) = ANY(%s)
"""

# Fetch pending writes for a single checkpoint. LangGraph applies a step's
# writes in writes_sort_key order (task_path, task_id, idx) and replays stored
# writes in the order they are returned, so they must come back in that order.
SELECT_WRITES_SQL = """
select cw.task_id, cw.channel, cw.type, cw.blob
from checkpoint_writes cw
where cw.thread_id = %s
    and cw.checkpoint_ns = %s
    and cw.checkpoint_id = %s
order by cw.task_path, cw.task_id, cw.idx
"""

# Batch fetch: blobs for multiple checkpoints in one query.
# Uses a subquery to expand each checkpoint's channel_versions JSONB into
# (checkpoint_id, channel, version) rows, then joins against checkpoint_blobs.
# Returns checkpoint_id so results can be grouped back per checkpoint.
SELECT_BLOBS_BATCH_SQL = """
select c.checkpoint_id, bl.channel, bl.type, bl.blob
from checkpoints c
cross join lateral jsonb_each_text(c.checkpoint -> 'channel_versions') AS jt(key, value)
inner join checkpoint_blobs bl
    on bl.thread_id = c.thread_id
    and bl.checkpoint_ns = c.checkpoint_ns
    and bl.channel = jt.key
    and bl.version = jt.value
where c.thread_id = %s
    and c.checkpoint_ns = %s
    and c.checkpoint_id = ANY(%s)
"""

# Batch fetch: writes for multiple checkpoints in one query.
SELECT_WRITES_BATCH_SQL = """
select cw.checkpoint_id, cw.task_id, cw.channel, cw.type, cw.blob
from checkpoint_writes cw
where cw.thread_id = %s
    and cw.checkpoint_ns = %s
    and cw.checkpoint_id = ANY(%s)
order by cw.checkpoint_id, cw.task_path, cw.task_id, cw.idx
"""

SELECT_PENDING_SENDS_SQL = f"""
select
    checkpoint_id,
    jsonb_agg(jsonb_build_object(
        'type', type,
        'blob', encode(blob, 'hex')
    ) order by task_path, task_id, idx) as sends
from checkpoint_writes
where thread_id = %s
    and checkpoint_id = any(%s)
    and channel = '{TASKS}'
group by checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"""

# Brings created_at forward on the blob rows a new checkpoint uses but does
# not write, when they are older than the refresh window, so TTL keeps them
# as long as the checkpoint. A put sends it in the transaction that writes
# the checkpoint row. Run again, it changes nothing more.
# Params: thread_id, checkpoint_ns, the (channel, version) pairs, the window.
REFRESH_CHECKPOINT_BLOBS_SQL = """
    UPDATE checkpoint_blobs SET created_at = now()
    WHERE thread_id = %s
        AND checkpoint_ns = %s
        AND (channel, version) = ANY(%s)
        AND created_at < now() - CAST(%s AS INTERVAL)
"""

# A table's storage parameters, its TTL settings among them, as CockroachDB
# renders them: key='value'. No row when the table does not exist.
SELECT_BLOB_TTL_SQL = """
    SELECT reloptions FROM pg_catalog.pg_class WHERE oid = to_regclass(%s)
"""

# One page of checkpoints in primary key order, for the refresh enable_ttl
# runs: when each was written, and the channel versions of each checkpoint
# still inside the TTL interval, NULL for the others. Paging by key keeps
# every statement small, whatever the table holds. Params: the interval, the
# last key of the page before (in the second form only), the page size.
_CHECKPOINT_PAGE_SQL = """
    SELECT thread_id, checkpoint_ns, checkpoint_id, created_at,
        CASE WHEN created_at > now() - CAST(%s AS INTERVAL)
            THEN checkpoint -> 'channel_versions' END AS channel_versions
    FROM checkpoints
    {after}
    ORDER BY thread_id, checkpoint_ns, checkpoint_id
    LIMIT %s
"""
SELECT_CHECKPOINT_PAGE_SQL = _CHECKPOINT_PAGE_SQL.format(after="")
SELECT_CHECKPOINT_PAGE_AFTER_SQL = _CHECKPOINT_PAGE_SQL.format(
    after="WHERE (thread_id, checkpoint_ns, checkpoint_id) > (%s, %s, %s)"
)

# Brings each blob row forward to when the newest checkpoint inside the TTL
# interval that uses it was written, when it is older than that by more than
# the refresh window: what a put of that checkpoint would have done. Only
# rows left unchanged that long are rewritten, and none ends up newer than
# its newest checkpoint. Run again, it changes nothing more. CockroachDB
# gives every array of a multi-array unnest one type unless each is cast.
# Params: arrays of thread_id, checkpoint_ns, channel, version and the
# newest checkpoint's created_at, then the window.
REFRESH_USED_BLOBS_SQL = """
    UPDATE checkpoint_blobs AS b SET created_at = u.newest
    FROM unnest(
        CAST(%s AS STRING[]), CAST(%s AS STRING[]), CAST(%s AS STRING[]),
        CAST(%s AS STRING[]), CAST(%s AS TIMESTAMPTZ[])
    ) AS u (thread_id, checkpoint_ns, channel, version, newest)
    WHERE b.thread_id = u.thread_id
        AND b.checkpoint_ns = u.checkpoint_ns
        AND b.channel = u.channel
        AND b.version = u.version
        AND b.created_at < u.newest - CAST(%s AS INTERVAL)
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"""

# Concurrent setup() calls can race to record the same version, so this must not fail on a duplicate
INSERT_MIGRATION_SQL = (
    "INSERT INTO checkpoint_migrations (v) VALUES (%s) ON CONFLICT (v) DO NOTHING"
)

# One page of a checkpoint's parent chain, starting at the checkpoint itself
# and newest first. Follows parent_checkpoint_id, so a forked thread yields
# only the ancestors on its own path.
SELECT_PARENT_CHAIN_SQL = """
WITH RECURSIVE chain (checkpoint_id, parent_checkpoint_id, depth) AS (
    SELECT checkpoint_id, parent_checkpoint_id, 1
    FROM checkpoints
    WHERE thread_id = %(thread_id)s
        AND checkpoint_ns = %(checkpoint_ns)s
        AND checkpoint_id = %(checkpoint_id)s
    UNION ALL
    SELECT c.checkpoint_id, c.parent_checkpoint_id, chain.depth + 1
    FROM chain
    JOIN checkpoints c
        ON c.thread_id = %(thread_id)s
        AND c.checkpoint_ns = %(checkpoint_ns)s
        AND c.checkpoint_id = chain.parent_checkpoint_id
    WHERE chain.depth < %(limit)s
)
SELECT checkpoint_id FROM chain ORDER BY depth
"""

# An interval enable_ttl takes, such as '7 days'
_TTL_INTERVAL = re.compile(
    r"^(?P<amount>\d+)\s+"
    r"(?P<unit>microsecond|millisecond|second|minute|hour|day|week|month|year)s?$"
)
# The length of each unit, to divide an interval as CockroachDB divides it
# for the grace: a month counts 30 days and a year 12 months. Since enable_ttl
# stores months and years as days, only a setting from 0.5 still holds them.
_TTL_UNITS = {
    "microsecond": timedelta(microseconds=1),
    "millisecond": timedelta(milliseconds=1),
    "second": timedelta(seconds=1),
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=360),
}
# The days enable_ttl stores for a month and a year, so the interval it
# stores has a fixed length. CockroachDB adds months and years by the
# calendar, and the same months added a few days apart can differ by up to 3
# days, so a grace could not cover them.
_TTL_FIXED_DAYS = {"month": 30, "year": 365}
# The expressions enable_ttl writes on checkpoint_blobs: with the grace since
# 0.6, and without it before. The grace may also read '<interval>'::INTERVAL.
_BLOB_TTL_WITH_GRACE = re.compile(
    r"\(created_at \+ '(?P<interval>[^']*)' \+ "
    r"(?:CAST\('(?P<cast>[^']*)' AS INTERVAL\)|'(?P<colons>[^']*)'::INTERVAL)"
    r" / (?P<fraction>[1-9][0-9]*)\)"
)
_BLOB_TTL_WITHOUT_GRACE = re.compile(r"\(created_at \+ '(?P<interval>[^']*)'\)")
_TTL_EXPRESSION_KEY = "ttl_expiration_expression="
# The backslash escapes of an escape string, other than \xHH
_SQL_ESCAPES = {"a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}
# What every saver in the process has logged, so each is logged once: a TTL
# setting enable_ttl did not write, by (table, setting), and a failed read of
# the setting, by (table, SQLSTATE)
_LOGGED_TTL_SETTINGS: set[tuple[str, str]] = set()
_LOGGED_TTL_READ_ERRORS: set[tuple[str, str]] = set()
# The most cut DeltaChannel chains a saver remembers having logged. When it
# has logged this many, it forgets them and starts again.
_CUT_CHAINS_REMEMBERED = 4096
# The longest duration CockroachDB keeps. It holds the hours, minutes and
# seconds of an interval as int64 nanoseconds, and longer duration text
# wraps to a negative interval with no error: about 106,751 days.
_LONGEST_DURATION = timedelta(microseconds=(2**63 - 1) // 1000)


def _fixed_interval(ttl_interval: str) -> str:
    """``ttl_interval`` with months and years written as days, so its length is fixed."""
    found = _TTL_INTERVAL.match(ttl_interval)
    days = None if found is None else _TTL_FIXED_DAYS.get(found["unit"])
    if found is None or days is None:
        return ttl_interval
    return f"{int(found['amount']) * days} days"


def _interval_length(interval: str) -> timedelta | None:
    """How long an interval that enable_ttl takes is, or None if it is not one.

    Raises:
        OverflowError: If it is too long for a timedelta.
    """
    found = _TTL_INTERVAL.match(interval)
    if found is None:
        return None
    return _TTL_UNITS[found["unit"]] * int(found["amount"])


def _duration(length: timedelta) -> str:
    """``length`` as interval text in hours, minutes and seconds, never days.

    CockroachDB adds days by the calendar of the session's time zone, where a
    day can last 23 or 25 hours, and adds hours as hours.
    """
    micros = length // timedelta(microseconds=1)
    seconds, micros = divmod(micros, 1_000_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}.{micros:06}"


def _sql_string(literal: str) -> str | None:
    """The text of a string literal as CockroachDB renders it in reloptions, or None.

    A string with a quote or another character to escape is rendered as an
    escape string, e'...', with backslash escapes; any other as '...'.
    """
    if len(literal) >= 3 and literal.startswith("e'") and literal.endswith("'"):
        return re.sub(
            r"\\(x[0-9a-fA-F]{2}|.)",
            lambda m: chr(int(m[1][1:], 16)) if len(m[1]) == 3 else _SQL_ESCAPES.get(m[1], m[1]),
            literal[2:-1],
            flags=re.DOTALL,
        )
    if len(literal) >= 2 and literal.startswith("'") and literal.endswith("'"):
        return literal[1:-1].replace("''", "'")
    return None


def _expression_setting(expression: str) -> tuple[timedelta, int | None] | None:
    """The interval of an expression enable_ttl wrote, and the divisor of its grace.

    The divisor is None for the expression without the grace, as 0.5 wrote
    it. Returns None for any other expression, and for an interval too long
    for a timedelta.
    """
    found = _BLOB_TTL_WITH_GRACE.fullmatch(expression)
    divisor: int | None = None
    if found is not None:
        again = found["cast"] if found["cast"] is not None else found["colons"]
        if again != found["interval"]:
            return None
        divisor = int(found["fraction"])
    else:
        found = _BLOB_TTL_WITHOUT_GRACE.fullmatch(expression)
        if found is None:
            return None
    try:
        length = _interval_length(found["interval"])
    except OverflowError:
        return None
    return None if length is None else (length, divisor)


def _refresh_window(expression: str, fraction: int) -> timedelta | None:
    """The refresh window of an expression enable_ttl wrote, or None for any other.

    With the grace, its divisor sets the window, so the window never exceeds
    the grace in effect. Without it, ``fraction`` does.
    """
    setting = _expression_setting(expression)
    if setting is None:
        return None
    length, divisor = setting
    try:
        window = length / (fraction if divisor is None else divisor)
    except ArithmeticError:
        # A saver's fraction of 0
        return None
    return window if window <= _LONGEST_DURATION else None


def _parse_blob_ttl(
    reloptions: Sequence[str] | None, fraction: int
) -> tuple[timedelta | None, str | None]:
    """What the TTL setting of checkpoint_blobs means for puts.

    ``reloptions`` is the table's pg_class.reloptions. For an expression
    enable_ttl wrote, returns the refresh window and None: with the grace,
    its divisor sets the window, and without it, as 0.5 wrote it,
    ``fraction`` does. Returns None and None when TTL is off, and None and
    the setting when TTL is on with a setting enable_ttl did not write.
    """
    option = _expression_option(reloptions)
    if option is None:
        ttl = [o for o in reloptions or [] if o.startswith("ttl")]
        return (None, ", ".join(ttl)) if "ttl='on'" in ttl else (None, None)
    text = _sql_string(option[len(_TTL_EXPRESSION_KEY) :])
    window = None if text is None else _refresh_window(text, fraction)
    return (window, None) if window is not None else (None, option)


def _expression_option(reloptions: Sequence[str] | None) -> str | None:
    """The ttl_expiration_expression entry of a table's reloptions, as rendered, or None."""
    return next((o for o in reloptions or [] if o.startswith(_TTL_EXPRESSION_KEY)), None)


def _blob_expression(reloptions: Sequence[str] | None) -> str | None:
    """The TTL expression in a table's reloptions, as written, or None."""
    option = _expression_option(reloptions)
    return None if option is None else _sql_string(option[len(_TTL_EXPRESSION_KEY) :])


@dataclass(frozen=True)
class _TTLState:
    """What a saver read about TTL on checkpoint_blobs, and when, by monotonic().

    ``window`` is the refresh window, or None when puts refresh nothing.
    """

    window: timedelta | None
    read_at: float


@dataclass(frozen=True)
class _TTLPlan:
    """What enable_ttl writes, and what the refresh it runs first needs.

    ``stored`` is the interval as stored, with months and years as days.
    ``interval`` is its length, and ``window`` the refresh window.
    """

    statements: tuple[str, ...]
    blob_expression: str
    stored: str
    interval: timedelta
    window: timedelta


class _DeltaHistoryWalk:
    """Collect DeltaChannel history over ancestors visited newest first.

    This is the accumulation in BaseCheckpointSaver.get_delta_channel_history,
    kept the same so the result is the same. Only how the ancestors are
    fetched differs: in pages, not one get_tuple per ancestor.
    """

    def __init__(self, channels: Sequence[str]) -> None:
        self.channels = list(channels)
        self.collected: dict[str, list[Any]] = {c: [] for c in self.channels}
        self.seed: dict[str, Any] = {}
        self.remaining = set(self.channels)
        # The missing ancestor the walk stopped at while channels had no seed
        self.cut: RunnableConfig | None = None

    def visit(self, tup: CheckpointTuple) -> None:
        if tup.pending_writes:
            for write in reversed(tup.pending_writes):
                if write[1] in self.remaining:
                    self.collected[write[1]].append(write)
        for channel in list(self.remaining):
            if channel in tup.checkpoint["channel_values"]:
                self.seed[channel] = tup.checkpoint["channel_values"][channel]
                self.remaining.discard(channel)

    def next_start(
        self, start: RunnableConfig, page: Sequence[CheckpointTuple], limit: int
    ) -> RunnableConfig | None:
        """Where the next page of ancestors starts, or None when the walk is over.

        ``page`` is what the page that began at ``start`` held. A full page
        goes on from the parent of its last checkpoint. A shorter one ends
        the chain: at the root, or at an ancestor that is missing, as when
        row-level TTL deleted it. When the chain ends at a missing ancestor
        while a channel still has no seed, the history of that channel is
        cut: ``cut`` keeps the missing ancestor, for the saver to log. The
        result stays what the walk found.
        """
        if not self.remaining:
            return None
        if page and len(page) == limit:
            return page[-1].parent_config
        self.cut = page[-1].parent_config if page else start
        return None

    def result(self) -> dict[str, Any]:
        history: dict[str, Any] = {}
        for channel in self.channels:
            entry: dict[str, Any] = {"writes": list(reversed(self.collected[channel]))}
            if channel in self.seed:
                entry["seed"] = self.seed[channel]
            history[channel] = entry
        return history


class BaseCockroachDBSaver(BaseCheckpointSaver[str]):
    """Base class for CockroachDB checkpoint savers.

    Provides shared SQL constants, serialization/deserialization logic,
    and migration definitions.
    """

    SELECT_SQL = SELECT_SQL
    SELECT_BLOBS_SQL = SELECT_BLOBS_SQL
    SELECT_WRITES_SQL = SELECT_WRITES_SQL
    SELECT_BLOBS_BATCH_SQL = SELECT_BLOBS_BATCH_SQL
    SELECT_WRITES_BATCH_SQL = SELECT_WRITES_BATCH_SQL
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    ENABLE_TTL_SQL = ENABLE_TTL_SQL
    DISABLE_TTL_SQL = DISABLE_TTL_SQL
    SELECT_BLOB_TTL_SQL = SELECT_BLOB_TTL_SQL
    REFRESH_CHECKPOINT_BLOBS_SQL = REFRESH_CHECKPOINT_BLOBS_SQL
    SELECT_CHECKPOINT_PAGE_SQL = SELECT_CHECKPOINT_PAGE_SQL
    SELECT_CHECKPOINT_PAGE_AFTER_SQL = SELECT_CHECKPOINT_PAGE_AFTER_SQL
    REFRESH_USED_BLOBS_SQL = REFRESH_USED_BLOBS_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL
    INSERT_MIGRATION_SQL = INSERT_MIGRATION_SQL
    SELECT_PARENT_CHAIN_SQL = SELECT_PARENT_CHAIN_SQL
    # Most ancestors fetched per round trip when rebuilding DeltaChannel history.
    # Pages start at DELTA_HISTORY_FIRST_PAGE and grow fourfold up to this.
    # Sizes below 1 count as 1.
    DELTA_HISTORY_PAGE_SIZE = 256
    DELTA_HISTORY_FIRST_PAGE = 16
    # With TTL on, the refresh window, and the grace enable_ttl gives blob
    # rows, are the TTL interval divided by this whole number
    TTL_REFRESH_FRACTION = TTL_REFRESH_FRACTION
    # Seconds the saver uses what it read about TTL before it reads it again
    TTL_STATE_CACHE_SECONDS = TTL_STATE_CACHE_SECONDS
    # Checkpoints one statement of the refresh in enable_ttl reads, and blob
    # rows one of its UPDATEs changes at most. Sizes below 1 count as 1.
    TTL_REFRESH_BATCH_SIZE = TTL_REFRESH_BATCH_SIZE
    TTL_REFRESH_UPDATE_SIZE = TTL_REFRESH_UPDATE_SIZE
    supports_pipeline: bool
    retry_policy: RetryPolicy
    # What the saver last read about TTL on checkpoint_blobs; None to read it
    _ttl_state: _TTLState | None = None
    # The cut DeltaChannel chains the saver has logged, as (thread_id,
    # checkpoint_ns, missing checkpoint_id). Made on first use, so a subclass
    # with an __init__ of its own needs nothing.
    _cut_chains_logged: set[tuple[str, str, str]]

    def _chain_page_params(self, start: RunnableConfig, limit: int) -> dict[str, Any]:
        configurable = start["configurable"]
        return {
            "thread_id": configurable["thread_id"],
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            "checkpoint_id": get_checkpoint_id(start),
            "limit": limit,
        }

    def _chain_tuples_query(self, start: RunnableConfig, ids: list[str]) -> tuple[str, list[Any]]:
        configurable = start["configurable"]
        return (
            self.SELECT_SQL
            + "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = ANY(%s)",
            [configurable["thread_id"], configurable.get("checkpoint_ns", ""), ids],
        )

    @staticmethod
    def _validate_ttl_params(ttl_interval: str, cron: str) -> None:
        """Validate TTL parameters to prevent SQL injection."""
        if not _TTL_INTERVAL.match(ttl_interval):
            raise ValueError(
                f"Invalid ttl_interval: {ttl_interval!r}. "
                "Must be like '7 days', '30 days', '1 hour', etc."
            )
        if not re.match(r"^[@\w\s\*/,-]+$", cron):
            raise ValueError(
                f"Invalid cron: {cron!r}. "
                "Must be a valid cron expression like '@daily', '* * * * *', etc."
            )

    def _ttl_plan(self, ttl_interval: str, cron: str) -> _TTLPlan:
        """What enable_ttl writes, after validating what goes in it, before any I/O.

        Months and years are stored as days, 30 to a month and 365 to a year,
        so the stored interval has a fixed length.

        Raises:
            ValueError: If the interval, the cron or TTL_REFRESH_FRACTION is invalid.
        """
        self._validate_ttl_params(ttl_interval, cron)
        fraction = self.TTL_REFRESH_FRACTION
        if isinstance(fraction, bool) or not isinstance(fraction, int) or fraction < 1:
            raise ValueError(
                f"Invalid TTL_REFRESH_FRACTION: {fraction!r}. Must be a whole number of at least 1."
            )
        stored = _fixed_interval(ttl_interval)
        try:
            length = _interval_length(stored)
        except OverflowError:
            length = None
        # The refresh binds the interval as a duration, which CockroachDB
        # would wrap to a negative one past this
        if length is None or length > _LONGEST_DURATION:
            raise ValueError(
                f"Invalid ttl_interval: {ttl_interval!r}. Must be at most "
                f"{_LONGEST_DURATION.days} days, the longest duration CockroachDB keeps."
            )
        values = {"interval": stored, "cron": cron, "fraction": fraction}
        return _TTLPlan(
            statements=tuple(sql.format(**values) for sql in self.ENABLE_TTL_SQL),
            blob_expression=BLOB_TTL_EXPRESSION.format(**values),
            stored=stored,
            interval=length,
            window=length / fraction,
        )

    @staticmethod
    def _refresh_needed(before: Sequence[str] | None, plan: _TTLPlan) -> bool:
        """Whether enable_ttl refreshes blob rows before it changes the settings.

        ``before`` is the reloptions of checkpoint_blobs as enable_ttl found
        them. No refresh when they already hold this expression: enable_ttl
        refreshes before it writes an expression, so a refresh has finished,
        and puts have refreshed blob rows since. Nor when they hold an
        expression with the grace that this one only lengthens: every blob
        row is then within the old window of its newest checkpoint, so within
        the new, longer one too.
        """
        expression = _blob_expression(before)
        if expression is None:
            return True
        if expression == plan.blob_expression:
            return False
        setting = _expression_setting(expression)
        if setting is None:
            return True
        length, divisor = setting
        if divisor is None:
            # The 0.5 expression, with no grace
            return True
        return length > plan.interval or length / divisor > plan.window

    def _checkpoint_page(
        self, after: tuple[str, str, str] | None, plan: _TTLPlan, limit: int
    ) -> tuple[str, list[Any]]:
        """The statement and parameters that read the page of checkpoints after ``after``."""
        interval = _duration(plan.interval)
        if after is None:
            return self.SELECT_CHECKPOINT_PAGE_SQL, [interval, limit]
        return self.SELECT_CHECKPOINT_PAGE_AFTER_SQL, [interval, *after, limit]

    @staticmethod
    def _used_blobs(
        page: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, str, str, str, datetime]]:
        """The blob rows that the page's checkpoints inside the interval use, each once.

        As (thread_id, checkpoint_ns, channel, version, newest), where newest
        is when the newest of those checkpoints that uses the row was
        written. Sorted by row, so that two refreshes at once update rows in
        the same order.
        """
        newest: dict[tuple[str, str, str, str], datetime] = {}
        for row in page:
            for channel, version in (row["channel_versions"] or {}).items():
                key = (row["thread_id"], row["checkpoint_ns"], channel, str(version))
                if key not in newest or row["created_at"] > newest[key]:
                    newest[key] = row["created_at"]
        return [(*key, written) for key, written in sorted(newest.items())]

    def _refresh_batches(
        self, used: list[tuple[str, str, str, str, datetime]], plan: _TTLPlan
    ) -> list[list[Any]]:
        """The parameters of each UPDATE that refreshes ``used``, TTL_REFRESH_UPDATE_SIZE rows each."""
        size = max(1, self.TTL_REFRESH_UPDATE_SIZE)
        window = _duration(plan.window)
        return [
            [*(list(column) for column in zip(*used[start : start + size], strict=True)), window]
            for start in range(0, len(used), size)
        ]

    @staticmethod
    def _page_end(
        page: Sequence[Mapping[str, Any]], after: tuple[str, str, str] | None
    ) -> tuple[str, str, str] | None:
        """The key the next page starts after."""
        if not page:
            return after
        last = page[-1]
        return (last["thread_id"], last["checkpoint_ns"], last["checkpoint_id"])

    @staticmethod
    def _refresh_started(plan: _TTLPlan, limit: int) -> float:
        """Log that enable_ttl's refresh starts, and return when, by monotonic()."""
        logger.info(
            "enable_ttl: bringing forward the blob rows that checkpoints written in the last %s "
            "use, so TTL keeps them as long as those checkpoints. This reads each checkpoint "
            "once, %d at a time, and rewrites only blob rows left unchanged for longer than the "
            "refresh window.",
            plan.stored,
            limit,
        )
        return monotonic()

    @staticmethod
    def _refresh_finished(refreshed: int, started: float) -> None:
        logger.info(
            "enable_ttl: finished in %.1f seconds. Blob rows brought forward: %d.",
            monotonic() - started,
            refreshed,
        )

    @staticmethod
    def _refresh_candidates(
        checkpoint: Checkpoint, blob_values: dict[str, Any], blob_versions: ChannelVersions
    ) -> list[tuple[str, str]]:
        """The blob rows a checkpoint uses but its put does not write, as (channel, version).

        They hold the channels stored as blobs, not inline, whose version did
        not change in this put.
        """
        versions = checkpoint["channel_versions"]
        return [
            (channel, cast(str, versions[channel]))
            for channel in blob_values
            if channel not in blob_versions and channel in versions
        ]

    def _fresh_ttl_state(self) -> _TTLState | None:
        """What the saver read about TTL, or None when it must read it again."""
        state = self._ttl_state
        if state is None or monotonic() - state.read_at >= self.TTL_STATE_CACHE_SECONDS:
            return None
        return state

    def _keep_ttl_state(self, reloptions: Sequence[str] | None) -> _TTLState:
        """Keep what the saver read about TTL on checkpoint_blobs, for the cache interval.

        A TTL setting that enable_ttl did not write turns the refresh off. It
        is logged at INFO once per process.
        """
        window, foreign = _parse_blob_ttl(reloptions, self.TTL_REFRESH_FRACTION)
        if foreign is not None and ("checkpoint_blobs", foreign) not in _LOGGED_TTL_SETTINGS:
            _LOGGED_TTL_SETTINGS.add(("checkpoint_blobs", foreign))
            logger.info(
                "checkpoint_blobs has a row-level TTL setting that enable_ttl did not write "
                "(%s). Puts do not refresh the blob rows that new checkpoints still use, so TTL "
                "can delete values that an active thread still uses. Run enable_ttl to replace "
                "the setting.",
                foreign,
            )
        return self._set_ttl_state(window)

    def _skips_ttl_read(self, error: BaseException) -> bool:
        """Whether a put goes on without the TTL setting after reading it failed with ``error``.

        Only after a database error that the retry policy does not retry,
        such as a missing privilege, and not inside a transaction the caller
        opened, which the error has aborted. Other errors are raised, as the
        put's own statements would raise them.
        """
        return (
            get_sqlstate(error) is not None
            and not self.retry_policy.is_retryable(error)
            and not in_caller_transaction(error)
        )

    def _ttl_read_failed(self, error: BaseException) -> _TTLState:
        """Keep "no refresh" for the cache interval, and log a WARNING once per process."""
        key = ("checkpoint_blobs", get_sqlstate(error) or "")
        if key not in _LOGGED_TTL_READ_ERRORS:
            _LOGGED_TTL_READ_ERRORS.add(key)
            logger.warning(
                "Cannot read the row-level TTL setting of checkpoint_blobs (%s). Until a read "
                "works, puts do not refresh the blob rows that new checkpoints still use, so if "
                "TTL is on it can delete values that an active thread still uses. The saver "
                "reads again every %g seconds.",
                _describe(error),
                self.TTL_STATE_CACHE_SECONDS,
            )
        return self._set_ttl_state(None)

    def _set_ttl_state(self, window: timedelta | None) -> _TTLState:
        state = _TTLState(window, monotonic())
        self._ttl_state = state
        return state

    def _forget_ttl_state(self) -> None:
        """Make the next put that needs the TTL setting read it again."""
        self._ttl_state = None

    def _warn_cut_chain(self, walk: _DeltaHistoryWalk) -> None:
        """Log a WARNING when ``walk`` stopped at a missing ancestor, once per such ancestor."""
        if walk.cut is None:
            return
        configurable = walk.cut["configurable"]
        key = (
            str(configurable["thread_id"]),
            str(configurable.get("checkpoint_ns", "")),
            str(configurable.get("checkpoint_id")),
        )
        logged = getattr(self, "_cut_chains_logged", None)
        if logged is None:
            logged = self._cut_chains_logged = set()
        if key in logged:
            return
        if len(logged) >= _CUT_CHAINS_REMEMBERED:
            logged.clear()
        logged.add(key)
        logger.warning(
            "The DeltaChannel history of thread %s (checkpoint_ns %r) stops at checkpoint %s, "
            "which is missing, before a snapshot of %s, so the state of those channels may be "
            "incomplete. Row-level TTL deletes old checkpoints by age: pick a TTL longer than "
            "such threads stay active, or delete finished threads with delete_thread. This "
            "saver logs each missing checkpoint once.",
            *key,
            ", ".join(sorted(walk.remaining)),
        )

    def _get_channel_version_pairs(self, checkpoint: dict[str, Any]) -> list[tuple[str, str]]:
        """Extract (channel, version) pairs from checkpoint's channel_versions."""
        cv = checkpoint.get("channel_versions") or {}
        return [(k, v) for k, v in cv.items()]

    def _migrate_pending_sends(
        self,
        pending_sends: list[dict[str, str]],
        checkpoint: dict[str, Any],
        channel_values: dict[str, Any],
    ) -> None:
        """Migrate pending sends from previous checkpoint format."""
        if not pending_sends:
            return
        enc, blob = self.serde.dumps_typed(
            [
                self.serde.loads_typed((s["type"], bytes.fromhex(s["blob"]) if s["blob"] else b""))
                for s in pending_sends
            ],
        )
        channel_values[TASKS] = self.serde.loads_typed((enc, blob))
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_rows: list[tuple[str, str, bytes | None]] | list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Load channel blobs from raw rows (channel, type, blob)."""
        if not blob_rows:
            return {}
        result = {}
        for row in blob_rows:
            if isinstance(row, dict):
                channel, typ, blob = row["channel"], row["type"], row["blob"]
            else:
                channel, typ, blob = row
            if typ == "empty" or blob is None:
                continue
            if isinstance(blob, memoryview):
                blob = bytes(blob)
            result[channel] = self.serde.loads_typed((typ, blob))
        return result

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        if not versions:
            return []
        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(self.serde.dumps_typed(values[k]) if k in values else ("empty", None)),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, write_rows: list[tuple[str, str, str, bytes]] | list[dict[str, Any]] | None
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes from raw rows (task_id, channel, type, blob)."""
        if not write_rows:
            return []
        result = []
        for row in write_rows:
            if isinstance(row, dict):
                task_id, channel, typ, blob = (
                    row["task_id"],
                    row["channel"],
                    row["type"],
                    row["blob"],
                )
            else:
                task_id, channel, typ, blob = row
            if isinstance(blob, memoryview):
                blob = bytes(blob)
            elif blob is None:
                blob = b""
            result.append((task_id, channel, self.serde.loads_typed((typ, blob))))
        return result

    def _load_blobs_batch(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Load blobs from batch query rows, grouped by checkpoint_id."""
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            cid = row["checkpoint_id"]
            channel, typ = row["channel"], row["type"]
            blob = row["blob"]
            if typ == "empty" or blob is None:
                continue
            if isinstance(blob, memoryview):
                blob = bytes(blob)
            if cid not in grouped:
                grouped[cid] = {}
            grouped[cid][channel] = self.serde.loads_typed((typ, blob))
        return grouped

    def _load_writes_batch(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, list[tuple[str, str, Any]]]:
        """Load writes from batch query rows, grouped by checkpoint_id."""
        grouped: dict[str, list[tuple[str, str, Any]]] = {}
        for row in rows:
            cid = row["checkpoint_id"]
            blob = row["blob"]
            if isinstance(blob, memoryview):
                blob = bytes(blob)
            elif blob is None:
                blob = b""
            if cid not in grouped:
                grouped[cid] = []
            grouped[cid].append(
                (row["task_id"], row["channel"], self.serde.loads_typed((row["type"], blob)))
            )
        return grouped

    def _build_checkpoint_tuple(
        self,
        value: dict[str, Any],
        blob_values: dict[str, Any],
        pending_writes: list[tuple[str, str, Any]],
    ) -> CheckpointTuple:
        """Build a CheckpointTuple from pre-fetched data."""
        thread_id = value["thread_id"]
        checkpoint_ns = value["checkpoint_ns"]
        checkpoint_id = value["checkpoint_id"]
        checkpoint = value["checkpoint"]

        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            # Safe: checkpoint from DB has all required Checkpoint keys; we only update channel_values
            cast(
                Checkpoint,
                {
                    **checkpoint,
                    "channel_values": {
                        **(checkpoint.get("channel_values") or {}),
                        **blob_values,
                    },
                },
            ),
            value["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            pending_writes,
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for list() given config, filter, before."""
        wheres: list[str] = []
        param_values: list[Any] = []

        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)
            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )
