"""Base class for CockroachDB checkpoint savers.

Contains SQL constants, migrations, and shared serialization logic.
Adapted from langgraph-checkpoint-postgres for CockroachDB compatibility:
- Replaces CREATE INDEX CONCURRENTLY with CREATE INDEX (CockroachDB runs
  index creation as non-blocking background jobs by default)
- Uses separate queries instead of correlated subqueries for performance
- Same table schema as PostgresSaver for potential migration compatibility
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg.types.json import Jsonb

MetadataInput = dict[str, Any] | None

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

# SQL templates for enabling TTL on checkpoint tables.
# Uses ttl_expiration_expression (recommended) to avoid full table rewrites.
ENABLE_TTL_SQL = [
    """ALTER TABLE checkpoints SET (
        ttl_expiration_expression = $$(created_at + '{interval}')$$,
        ttl_job_cron = '{cron}'
    );""",
    """ALTER TABLE checkpoint_blobs SET (
        ttl_expiration_expression = $$(created_at + '{interval}')$$,
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

# Fetch pending writes for a single checkpoint.
SELECT_WRITES_SQL = """
select cw.task_id, cw.channel, cw.type, cw.blob
from checkpoint_writes cw
where cw.thread_id = %s
    and cw.checkpoint_ns = %s
    and cw.checkpoint_id = %s
order by cw.task_id, cw.idx
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
order by cw.checkpoint_id, cw.task_id, cw.idx
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
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL
    supports_pipeline: bool

    @staticmethod
    def _validate_ttl_params(ttl_interval: str, cron: str) -> None:
        """Validate TTL parameters to prevent SQL injection."""
        if not re.match(
            r"^[\d]+\s+(microsecond|millisecond|second|minute|hour|day|week|month|year)s?$",
            ttl_interval,
        ):
            raise ValueError(
                f"Invalid ttl_interval: {ttl_interval!r}. "
                "Must be like '7 days', '30 days', '1 hour', etc."
            )
        if not re.match(r"^[@\w\s\*/,-]+$", cron):
            raise ValueError(
                f"Invalid cron: {cron!r}. "
                "Must be a valid cron expression like '@daily', '* * * * *', etc."
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
            {
                **checkpoint,
                "channel_values": {
                    **(checkpoint.get("channel_values") or {}),
                    **blob_values,
                },
            },
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
