"""Synchronous CockroachDB checkpoint saver for LangGraph.

Provides the same interface as langgraph-checkpoint-postgres PostgresSaver,
adapted for CockroachDB's isolation levels (SERIALIZABLE default, READ COMMITTED supported)
and online schema changes.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from langchain_cockroachdb.checkpointer import _internal
from langchain_cockroachdb.checkpointer.base import BaseCockroachDBSaver

Conn = _internal.Conn


def _sanitize_conn_string(conn_string: str) -> str:
    """Convert cockroachdb:// URLs to postgresql:// for psycopg compatibility."""
    if conn_string.startswith("cockroachdb://"):
        return conn_string.replace("cockroachdb://", "postgresql://", 1)
    if conn_string.startswith("cockroachdb+psycopg://"):
        return conn_string.replace("cockroachdb+psycopg://", "postgresql://", 1)
    return conn_string


class CockroachDBSaver(BaseCockroachDBSaver):
    """Synchronous checkpointer that stores checkpoints in CockroachDB.

    Uses raw psycopg3 connections (not SQLAlchemy) for compatibility
    with LangGraph's checkpointer interface.

    Example:
        ```python
        from langchain_cockroachdb import CockroachDBSaver

        DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
        with CockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
            # Use with LangGraph
            graph = workflow.compile(checkpointer=checkpointer)
        ```
    """

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )
        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> Iterator[CockroachDBSaver]:
        """Create a new CockroachDBSaver from a connection string.

        Args:
            conn_string: CockroachDB connection string.
                Supports cockroachdb:// and postgresql:// formats.
            pipeline: Whether to use Pipeline mode (may not be supported
                by all CockroachDB versions).

        Yields:
            CockroachDBSaver instance.
        """
        conn_string = _sanitize_conn_string(conn_string)
        with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=5, row_factory=dict_row
        ) as conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield cls(conn, pipe)
            else:
                yield cls(conn)

    def setup(self) -> None:
        """Set up the checkpoint database.

        Creates necessary tables and runs migrations. MUST be called
        the first time the checkpointer is used.
        """
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            results = cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
            row = results.fetchone()
            version = -1 if row is None else row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
                strict=False,
            ):
                cur.execute(migration)
                cur.execute("INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,))
            if self.pipe:
                self.pipe.sync()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        Args:
            config: Config for filtering checkpoints.
            filter: Additional metadata filtering criteria.
            before: Only return checkpoints before this checkpoint ID.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple for each matching checkpoint.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        params = list(args)
        if limit is not None:
            query += " LIMIT %s"
            params.append(int(limit))

        with self._cursor() as cur:
            cur.execute(query, params, binary=True)
            values = cur.fetchall()
            if not values:
                return

            if to_migrate := [
                v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (
                        values[0]["thread_id"],
                        [v["parent_checkpoint_id"] for v in to_migrate],
                    ),
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)
                for sends in cur:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value.setdefault("_blob_values", {}),
                        )

            # Batch-fetch blobs and writes grouped by (thread_id, checkpoint_ns)
            groups: dict[tuple[str, str], list[str]] = defaultdict(list)
            for v in values:
                groups[(v["thread_id"], v["checkpoint_ns"])].append(v["checkpoint_id"])

            blobs_by_id: dict[str, dict[str, Any]] = {}
            writes_by_id: dict[str, list[tuple[str, str, Any]]] = {}
            for (tid, cns), cids in groups.items():
                cur.execute(self.SELECT_BLOBS_BATCH_SQL, (tid, cns, cids), binary=True)
                blobs_by_id.update(self._load_blobs_batch(cur.fetchall()))
                cur.execute(self.SELECT_WRITES_BATCH_SQL, (tid, cns, cids), binary=True)
                writes_by_id.update(self._load_writes_batch(cur.fetchall()))

            for value in values:
                cid = value["checkpoint_id"]
                blob_values = {**(value.get("_blob_values") or {}), **(blobs_by_id.get(cid) or {})}
                pending_writes = writes_by_id.get(cid, [])
                yield self._build_checkpoint_tuple(value, blob_values, pending_writes)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        Args:
            config: Config identifying the checkpoint.

        Returns:
            CheckpointTuple or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = (
                "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"
            )

        with self._cursor() as cur:
            cur.execute(self.SELECT_SQL + where, args, binary=True)
            value = cur.fetchone()
            if value is None:
                return None

            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                if sends := cur.fetchone():
                    blob_values: dict[str, Any] = {}
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        blob_values,
                    )
                    value["_blob_values"] = blob_values

            return self._load_checkpoint_tuple_with_cursor(cur, value)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        Args:
            config: Config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration with checkpoint ID.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()

        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        with self._cursor(pipeline=True) as cur:
            if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
                cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    self._dump_blobs(thread_id, checkpoint_ns, blob_values, blob_versions),
                )
            cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(copy),
                    Jsonb(get_serializable_checkpoint_metadata(config, metadata)),
                ),
            )

        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of (channel, value) pairs to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        with self._cursor(pipeline=True) as cur:
            cur.executemany(
                query,
                self._dump_writes(
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    task_path,
                    writes,
                ),
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.

        Args:
            thread_id: The thread ID to delete.
        """
        with self._cursor(pipeline=True) as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    def enable_ttl(self, ttl_interval: str = "7 days", cron: str = "@daily") -> None:
        """Enable CockroachDB row-level TTL on checkpoint tables.

        Rows older than ttl_interval will be automatically deleted by a
        background CockroachDB job on the specified cron schedule.

        Args:
            ttl_interval: Interval after which rows expire (e.g., '7 days', '30 days').
            cron: Cron schedule for the TTL deletion job (default: '@daily').
        """
        self._validate_ttl_params(ttl_interval, cron)
        with self._cursor() as cur:
            for sql_template in self.ENABLE_TTL_SQL:
                cur.execute(sql_template.format(interval=ttl_interval, cron=cron))

    def disable_ttl(self) -> None:
        """Disable CockroachDB row-level TTL on checkpoint tables."""
        with self._cursor() as cur:
            for sql in self.DISABLE_TTL_SQL:
                cur.execute(sql)

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
        """Create a database cursor as a context manager."""
        with self.lock, _internal.get_connection(self.conn) as conn:
            if self.pipe:
                try:
                    with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        self.pipe.sync()
            elif pipeline:
                if self.supports_pipeline:
                    with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    def _load_checkpoint_tuple_with_cursor(
        self, cur: Cursor[DictRow], value: DictRow
    ) -> CheckpointTuple:
        """Convert a database row into a CheckpointTuple, fetching blobs and writes separately."""
        thread_id = value["thread_id"]
        checkpoint_ns = value["checkpoint_ns"]
        checkpoint_id = value["checkpoint_id"]
        checkpoint = value["checkpoint"]

        # Fetch blobs via separate query
        cv_pairs = self._get_channel_version_pairs(checkpoint)
        blob_values: dict[str, Any] = value.get("_blob_values") or {}
        if cv_pairs:
            cur.execute(
                self.SELECT_BLOBS_SQL,
                (thread_id, checkpoint_ns, cv_pairs),
                binary=True,
            )
            blob_values = {**blob_values, **self._load_blobs(cur.fetchall())}

        # Fetch writes via separate query
        cur.execute(
            self.SELECT_WRITES_SQL,
            (thread_id, checkpoint_ns, checkpoint_id),
            binary=True,
        )
        pending_writes = self._load_writes(cur.fetchall())

        return self._build_checkpoint_tuple(value, blob_values, pending_writes)
