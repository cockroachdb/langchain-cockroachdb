"""Base class for CockroachDB checkpoint savers.

Contains SQL constants, migrations, and shared serialization logic.
Adapted from langgraph-checkpoint-postgres for CockroachDB compatibility:
- Replaces CREATE INDEX CONCURRENTLY with CREATE INDEX (CockroachDB runs
  index creation as non-blocking background jobs by default)
- Same table schema as PostgresSaver for potential migration compatibility
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
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
]

# CockroachDB does not support multidimensional arrays (array_agg(array[...])).
# We use jsonb_agg(jsonb_build_object(...)) with hex-encoded blobs instead.
# The _load_blobs / _load_writes methods parse this JSON format.
SELECT_SQL = """
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select jsonb_agg(jsonb_build_object(
            'channel', bl.channel,
            'type', bl.type,
            'blob', encode(bl.blob, 'hex')
        ))
        from jsonb_each_text(checkpoints.checkpoint -> 'channel_versions') AS jt
        inner join checkpoint_blobs bl
            on bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns = checkpoints.checkpoint_ns
            and bl.channel = jt.key
            and bl.version = jt.value
    ) as channel_values,
    (
        select jsonb_agg(jsonb_build_object(
            'task_id', cw.task_id,
            'channel', cw.channel,
            'type', cw.type,
            'blob', encode(cw.blob, 'hex')
        ) order by cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes
from checkpoints """

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
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL
    supports_pipeline: bool

    def _migrate_pending_sends(
        self,
        pending_sends: list[dict[str, str]],
        checkpoint: dict[str, Any],
        channel_values: list[dict[str, str]],
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
        channel_values.append({"channel": TASKS, "type": enc, "blob": blob.hex()})
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(self, blob_values: list[dict[str, str]] | None) -> dict[str, Any]:
        """Load channel blobs from JSONB format returned by SELECT_SQL."""
        if not blob_values:
            return {}
        return {
            item["channel"]: self.serde.loads_typed(
                (item["type"], bytes.fromhex(item["blob"]) if item["blob"] else b"")
            )
            for item in blob_values
            if item["type"] != "empty"
        }

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

    def _load_writes(self, writes: list[dict[str, str]] | None) -> list[tuple[str, str, Any]]:
        """Load pending writes from JSONB format returned by SELECT_SQL."""
        if not writes:
            return []
        return [
            (
                w["task_id"],
                w["channel"],
                self.serde.loads_typed((w["type"], bytes.fromhex(w["blob"]) if w["blob"] else b"")),
            )
            for w in writes
        ]

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
