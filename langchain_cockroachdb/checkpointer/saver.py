"""Synchronous CockroachDB checkpoint saver for LangGraph.

Provides the same interface as langgraph-checkpoint-postgres PostgresSaver,
adapted for CockroachDB's isolation levels (SERIALIZABLE default, READ COMMITTED
supported) and online schema changes. Every operation retries transient
CockroachDB errors under the saver's RetryPolicy. All writes are UPSERTs or
DELETEs, so retrying after an ambiguous result is safe.
"""

from __future__ import annotations

import builtins
import contextlib
import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

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

from langchain_cockroachdb._connection import to_libpq_url
from langchain_cockroachdb._guard import abandon_after
from langchain_cockroachdb.checkpointer import _internal
from langchain_cockroachdb.checkpointer.base import BaseCockroachDBSaver, connection_options
from langchain_cockroachdb.retry import RetryPolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")

Conn = _internal.Conn


def _sanitize_conn_string(conn_string: str) -> str:
    """Convert any supported URL spelling to postgresql:// for psycopg."""
    return to_libpq_url(conn_string)


class CockroachDBSaver(BaseCockroachDBSaver):
    """Synchronous checkpointer that stores checkpoints in CockroachDB.

    Uses raw psycopg3 connections (not SQLAlchemy) for compatibility
    with LangGraph's checkpointer interface. For production, pass a
    ``psycopg_pool.ConnectionPool``: it replaces broken connections, while a
    single connection cannot recover from a dropped connection.

    Example:
        ```python
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        from langchain_cockroachdb import CockroachDBSaver, RetryPolicy

        pool = ConnectionPool(
            "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        checkpointer = CockroachDBSaver(pool, retry_policy=RetryPolicy(max_attempts=8))
        checkpointer.setup()
        graph = workflow.compile(checkpointer=checkpointer)
        ```
    """

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
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
        self.retry_policy = retry_policy or RetryPolicy()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
        retry_policy: RetryPolicy | None = None,
        connect_timeout: float | None = 10.0,
        prepare_threshold: int | None = 5,
        connect_kwargs: Mapping[str, Any] | None = None,
    ) -> Iterator[CockroachDBSaver]:
        """Create a saver on a single connection.

        Args:
            conn_string: CockroachDB connection string (cockroachdb://, postgresql:// and similar)
            pipeline: Whether to use Pipeline mode
            serde: Custom serializer protocol
            retry_policy: How operations retry (default: RetryPolicy())
            connect_timeout: Seconds to wait when connecting, or None (default: 10)
            prepare_threshold: psycopg prepared statement threshold, None to disable (default: 5)
            connect_kwargs: Extra psycopg connection options, e.g. application_name

        Yields:
            CockroachDBSaver instance.
        """
        policy = retry_policy or RetryPolicy()
        url = _sanitize_conn_string(conn_string)
        options = connection_options(url, connect_kwargs, connect_timeout, prepare_threshold)
        conn = policy.run(lambda: Connection.connect(url, **options), label="connect checkpointer")
        with conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield cls(conn, pipe, serde=serde, retry_policy=policy)
            else:
                yield cls(conn, serde=serde, retry_policy=policy)

    def _retry(self, fn: Callable[[], T], *, label: str) -> T:
        """Run ``fn`` under the retry policy."""
        return self.retry_policy.run(fn, label=label, can_retry=self._can_retry)

    def _sync_pipeline(self) -> None:
        """Sync pipeline if one is in use. Each statement runs in its own transaction."""
        if self.pipe:
            self.pipe.sync()

    def _can_retry(self, error: BaseException) -> bool:
        if isinstance(self.conn, Connection) and self.conn.closed:
            logger.warning(
                "Not retrying: the checkpointer's only connection is closed and cannot be "
                "reopened. Pass a psycopg_pool.ConnectionPool so broken connections are replaced."
            )
            return False
        return True

    def setup(self) -> None:
        """Set up the checkpoint database.

        Creates necessary tables and runs migrations. MUST be called the first
        time the checkpointer is used. Safe to run from several processes at
        once: every migration is idempotent and versions are recorded with
        ON CONFLICT DO NOTHING.
        """

        def init_and_read_version() -> int:
            with self._cursor(
                timeout=self.retry_policy.ddl_attempt_timeout, label="checkpointer migration 0"
            ) as cur:
                cur.execute(self.MIGRATIONS[0])
                self._sync_pipeline()
                results = cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
                self._sync_pipeline()
                row = results.fetchone()
                return -1 if row is None else row["v"]

        version = self._retry(init_and_read_version, label="checkpointer migration 0")

        for v in range(version + 1, len(self.MIGRATIONS)):

            def apply_migration(migration_v: int = v) -> None:
                with self._cursor(
                    timeout=self.retry_policy.ddl_attempt_timeout,
                    label=f"checkpointer migration {migration_v}",
                ) as cur:
                    cur.execute(self.MIGRATIONS[migration_v])
                    self._sync_pipeline()
                    cur.execute(self.INSERT_MIGRATION_SQL, (migration_v,))
                    self._sync_pipeline()

            self._retry(apply_migration, label=f"checkpointer migration {v}")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        Matching checkpoints are fetched inside one retried call and yielded
        afterwards, so the saver's lock is not held while the caller iterates.

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
        yield from self._retry(lambda: self._fetch_list(query, params), label="list checkpoints")

    def _fetch_list(self, query: str, params: builtins.list[Any]) -> builtins.list[CheckpointTuple]:
        with self._cursor(
            timeout=self.retry_policy.attempt_timeout, label="list checkpoints"
        ) as cur:
            cur.execute(query, params, binary=True)
            values = cur.fetchall()
            if not values:
                return []

            if to_migrate := [
                v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (values[0]["thread_id"], [v["parent_checkpoint_id"] for v in to_migrate]),
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

        return [
            self._build_checkpoint_tuple(
                value,
                {
                    **(value.get("_blob_values") or {}),
                    **(blobs_by_id.get(value["checkpoint_id"]) or {}),
                },
                writes_by_id.get(value["checkpoint_id"], []),
            )
            for value in values
        ]

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
        return self._retry(
            lambda: self._fetch_tuple(where, args, thread_id), label="get checkpoint"
        )

    def _fetch_tuple(
        self, where: str, args: tuple[Any, ...], thread_id: str
    ) -> CheckpointTuple | None:
        with self._cursor(timeout=self.retry_policy.attempt_timeout, label="get checkpoint") as cur:
            cur.execute(self.SELECT_SQL + where, args, binary=True)
            value = cur.fetchone()
            if value is None:
                return None
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL, (thread_id, [value["parent_checkpoint_id"]])
                )
                if sends := cur.fetchone():
                    blob_values: dict[str, Any] = {}
                    self._migrate_pending_sends(sends["sends"], value["checkpoint"], blob_values)
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
        next_config: RunnableConfig = {
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

        # Serialize once, outside the retry loop and the lock
        blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}
        blob_rows = (
            self._dump_blobs(thread_id, checkpoint_ns, blob_values, blob_versions)
            if blob_versions
            else []
        )
        checkpoint_row = (
            thread_id,
            checkpoint_ns,
            checkpoint["id"],
            checkpoint_id,
            Jsonb(copy),
            Jsonb(get_serializable_checkpoint_metadata(config, metadata)),
        )

        def write() -> None:
            with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="put checkpoint",
            ) as cur:
                if blob_rows:
                    cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_rows)
                cur.execute(self.UPSERT_CHECKPOINTS_SQL, checkpoint_row)

        self._retry(write, label="put checkpoint")
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
        rows = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )

        def write() -> None:
            with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="put checkpoint writes",
            ) as cur:
                cur.executemany(query, rows)

        self._retry(write, label="put checkpoint writes")

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.

        Args:
            thread_id: The thread ID to delete.
        """

        def delete() -> None:
            with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="delete thread",
            ) as cur:
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (str(thread_id),))
                cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (str(thread_id),))
                cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (str(thread_id),))

        self._retry(delete, label="delete thread")

    def enable_ttl(self, ttl_interval: str = "7 days", cron: str = "@daily") -> None:
        """Enable CockroachDB row-level TTL on checkpoint tables.

        Rows older than ttl_interval will be automatically deleted by a
        background CockroachDB job on the specified cron schedule.

        Args:
            ttl_interval: Interval after which rows expire (e.g., '7 days', '30 days').
            cron: Cron schedule for the TTL deletion job (default: '@daily').
        """
        self._validate_ttl_params(ttl_interval, cron)
        statements = [sql.format(interval=ttl_interval, cron=cron) for sql in self.ENABLE_TTL_SQL]
        for statement in statements:
            self._run_ddl(statement, "enable TTL")

    def disable_ttl(self) -> None:
        """Disable CockroachDB row-level TTL on checkpoint tables."""
        for statement in self.DISABLE_TTL_SQL:
            self._run_ddl(statement, "disable TTL")

    def _run_ddl(self, statement: str, label: str) -> None:
        """Run a single DDL statement with retry and timeout."""

        def execute() -> None:
            with self._cursor(timeout=self.retry_policy.ddl_attempt_timeout, label=label) as cur:
                cur.execute(statement)
                self._sync_pipeline()

        self._retry(execute, label=label)

    @contextmanager
    def _cursor(
        self,
        *,
        pipeline: bool = False,
        timeout: float | None = None,
        label: str = "checkpoint operation",
    ) -> Iterator[Cursor[DictRow]]:
        """Create a database cursor as a context manager, with the hang guard."""
        with self.lock, _internal.get_connection(self.conn) as conn:
            fileno = conn.fileno() if timeout is not None else None
            with abandon_after(fileno, timeout, label=label):
                if self.pipe:
                    try:
                        with conn.cursor(binary=True, row_factory=dict_row) as cur:
                            yield cur
                        if pipeline:
                            self.pipe.sync()
                    except BaseException:
                        # Resync while still holding the lock, so other threads' queued statements are untouched
                        with contextlib.suppress(Exception):
                            self.pipe.sync()
                        raise
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

        cv_pairs = self._get_channel_version_pairs(checkpoint)
        blob_values: dict[str, Any] = value.get("_blob_values") or {}
        if cv_pairs:
            cur.execute(self.SELECT_BLOBS_SQL, (thread_id, checkpoint_ns, cv_pairs), binary=True)
            blob_values = {**blob_values, **self._load_blobs(cur.fetchall())}

        cur.execute(self.SELECT_WRITES_SQL, (thread_id, checkpoint_ns, checkpoint_id), binary=True)
        pending_writes = self._load_writes(cur.fetchall())

        return self._build_checkpoint_tuple(value, blob_values, pending_writes)
