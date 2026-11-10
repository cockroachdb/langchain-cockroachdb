"""Synchronous CockroachDB checkpoint saver for LangGraph.

Provides the same interface as langgraph-checkpoint-postgres PostgresSaver,
adapted for CockroachDB's isolation levels (SERIALIZABLE default, READ COMMITTED
supported) and online schema changes. Every operation retries transient
CockroachDB errors under the saver's RetryPolicy. All writes are UPSERTs or
DELETEs, so retrying after an ambiguous result is safe.
"""

from __future__ import annotations

import builtins
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
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
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import _psycopg_conn
from langchain_cockroachdb._connection import to_libpq_url
from langchain_cockroachdb._psycopg_ops import PsycopgOps
from langchain_cockroachdb.checkpointer.base import (
    BaseCockroachDBSaver,
    _DeltaHistoryWalk,
    _duration,
    _TTLPlan,
    connection_options,
)
from langchain_cockroachdb.retry import RetryPolicy, schema_change_policy

logger = logging.getLogger(__name__)

# What enable_ttl waits with while TTL is paused; tests replace it
_sleep = time.sleep

Conn = _psycopg_conn.Conn


def _sanitize_conn_string(conn_string: str) -> str:
    """Convert any supported URL spelling to postgresql:// for psycopg."""
    return to_libpq_url(conn_string)


class CockroachDBSaver(PsycopgOps, BaseCockroachDBSaver):
    """Synchronous checkpointer that stores checkpoints in CockroachDB.

    Uses raw psycopg3 connections (not SQLAlchemy) for compatibility
    with LangGraph's checkpointer interface. For production, pass a
    ``psycopg_pool.ConnectionPool``: it replaces broken connections, while a
    single connection cannot recover from a dropped connection.

    On a pool, operations run concurrently, each on its own pooled
    connection. On a single connection they run one at a time. A write with
    several statements runs them in one transaction, so it is saved completely
    or not at all, even when the caller stops partway through. On a single
    connection the caller also uses, an operation inside a transaction the
    caller opened joins that transaction, and its errors are raised at once,
    not retried, so the caller can retry the whole transaction.

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

    component = "checkpointer"
    _log = logger

    # Runs operations one at a time on a single connection; a pool needs no lock
    lock: threading.Lock

    def __init__(
        self,
        conn: _psycopg_conn.Conn,
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
        # Set while a write batch of ours is between BEGIN and COMMIT, and left
        # set if its rollback did not finish. Only a single connection sets it,
        # under the lock: a pool rolls back the connections it gets back. It
        # cannot tell our stopped batch from a transaction the caller opens on
        # the shared connection after ours ended, before the next saver call.
        # Sharing a saver's single connection that way is outside the
        # documented use: use a pool, or keep the connection to the saver.
        self._batch_open = False

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
            prepare_threshold: psycopg prepared statement threshold, None to disable
                (default: 5). Statements sent in a pipeline are never prepared
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

    def setup(self) -> None:
        """Set up the checkpoint database.

        Creates necessary tables and runs migrations. MUST be called the first
        time the checkpointer is used. Safe to run from several processes at
        once: every migration is idempotent and versions are recorded with
        ON CONFLICT DO NOTHING.
        """

        def init_and_read_version() -> int:
            with self._cursor(
                timeout=self.retry_policy.ddl_attempt_timeout, label="checkpointer setup"
            ) as cur:
                cur.execute(self.MIGRATIONS[0])
                self._sync_pipeline()
                results = cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
                self._sync_pipeline()
                row = results.fetchone()
                return -1 if row is None else row["v"]

        version = self._retry(init_and_read_version, label="checkpointer setup")

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
        afterwards, so no connection or lock is held while the caller iterates.

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

    def _fetch_list(
        self, query: str, params: builtins.list[Any], *, label: str = "list checkpoints"
    ) -> builtins.list[CheckpointTuple]:
        with self._cursor(timeout=self.retry_policy.attempt_timeout, label=label) as cur:
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

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        """Rebuild DeltaChannel history for a checkpoint.

        LangGraph's default walks the parent chain with one get_tuple per
        ancestor, which is three queries per ancestor. This fetches the chain
        a page at a time instead, in pages that grow up to DELTA_HISTORY_PAGE_SIZE,
        and returns the same result. A walk that stops at a missing ancestor,
        as when row-level TTL deleted it, before a channel's snapshot logs a
        WARNING: that channel's state may be incomplete.

        Args:
            config: Config identifying the checkpoint.
            channels: DeltaChannel names to rebuild.

        Returns:
            Per-channel writes, and the seed value where one was found.
        """
        if not channels:
            return {}
        walk = _DeltaHistoryWalk(channels)
        target = self.get_tuple(config)
        start = target.parent_config if target else None
        # A page always holds its start row, so a limit below 1 would make the
        # first page look like the end of the chain
        limit = max(1, min(self.DELTA_HISTORY_FIRST_PAGE, self.DELTA_HISTORY_PAGE_SIZE))
        while start is not None and walk.remaining:
            page = self._retry(
                partial(self._fetch_chain_page, start, limit), label="get delta channel history"
            )
            for tup in page:
                walk.visit(tup)
                if not walk.remaining:
                    break
            start = walk.next_start(start, page, limit)
            limit = max(1, min(limit * 4, self.DELTA_HISTORY_PAGE_SIZE))
        self._warn_cut_chain(walk)
        return walk.result()

    def _fetch_chain_page(
        self, start: RunnableConfig, limit: int
    ) -> builtins.list[CheckpointTuple]:
        with self._cursor(
            timeout=self.retry_policy.attempt_timeout, label="get delta channel history"
        ) as cur:
            cur.execute(self.SELECT_PARENT_CHAIN_SQL, self._chain_page_params(start, limit))
            ids = [row["checkpoint_id"] for row in cur.fetchall()]
        if not ids:
            return []
        query, params = self._chain_tuples_query(start, ids)
        by_id = {
            tup.config["configurable"]["checkpoint_id"]: tup
            for tup in self._fetch_list(query, params, label="get delta channel history")
        }
        # Keep chain order and stop at the first missing ancestor, as the default does
        page = []
        for checkpoint_id in ids:
            if checkpoint_id not in by_id:
                break
            page.append(by_id[checkpoint_id])
        return page

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        With row-level TTL on, the put also brings forward created_at on the
        blob rows the checkpoint uses but does not write, once they are
        older than the refresh window, in the same transaction, so TTL keeps
        them as long as the checkpoint.

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
        # Only a put that uses blob rows it does not write needs the TTL setting
        candidates = self._refresh_candidates(checkpoint, blob_values, blob_versions)

        def write() -> None:
            # Decided in each attempt, before its cursor opens, so a retry that
            # comes after the saver learns that TTL is on refreshes too
            window = self._ttl_refresh_window() if candidates else None
            refresh = (
                None
                if window is None
                else (thread_id, checkpoint_ns, candidates, _duration(window))
            )
            with self._cursor(
                pipeline=True,
                transaction=bool(blob_rows) or refresh is not None,
                timeout=self.retry_policy.attempt_timeout,
                label="put checkpoint",
            ) as cur:
                if blob_rows:
                    cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_rows)
                if refresh is not None:
                    cur.execute(self.REFRESH_CHECKPOINT_BLOBS_SQL, refresh)
                cur.execute(self.UPSERT_CHECKPOINTS_SQL, checkpoint_row)

        self._retry(write, label="put checkpoint")
        return next_config

    def _ttl_refresh_window(self) -> timedelta | None:
        """The refresh window for blob rows, or None when puts refresh nothing.

        The saver reads the TTL setting of checkpoint_blobs from the catalog
        once TTL_STATE_CACHE_SECONDS have passed since its last read started,
        since what a read finds may be that old, or once its own enable_ttl
        or disable_ttl has run. The read is one attempt: the put calls this
        in each of its attempts, and its retry covers a transient error. If
        the read fails with a database error that the retry policy does
        not retry, such as a missing privilege, the put goes on without
        refreshing, and the saver logs a WARNING and reads again after the
        cache interval.
        """
        state = self._fresh_ttl_state()
        if state is None:
            started = self._ttl_read_starts()
            try:
                reloptions = self._read_blob_ttl()
            except Exception as error:
                if not self._skips_ttl_read(error):
                    raise
                state = self._ttl_read_failed(error, started)
            else:
                state = self._keep_ttl_state(reloptions, started)
        return state.window

    def _read_blob_ttl(self) -> builtins.list[str] | None:
        """The pg_class.reloptions of checkpoint_blobs, or None if it has none."""
        with self._cursor(
            timeout=self.retry_policy.attempt_timeout, label="read checkpoint TTL"
        ) as cur:
            cur.execute(self.SELECT_BLOB_TTL_SQL, ("checkpoint_blobs",))
            row = cur.fetchone()
        return None if row is None else row["reloptions"]

    def _refresh_used_blobs(self, plan: _TTLPlan) -> None:
        """Bring forward the blob rows that checkpoints inside the interval use.

        enable_ttl runs this once, while TTL is paused and after the wait, so
        the blob rows of threads that are idle then live as long as their
        newest checkpoints. It reads the checkpoints a page at a time in
        primary key order, and updates the rows a page needs
        TTL_REFRESH_UPDATE_SIZE at a time. Each read and each UPDATE is its
        own retried operation under the policy's ddl_attempt_timeout, so no
        transaction grows with the table. A row is brought forward to now
        only when it is older than its newest checkpoint by more than the
        refresh window, so only rows left unchanged that long are rewritten,
        and running a page twice changes nothing more.
        """
        limit = max(1, self.TTL_REFRESH_BATCH_SIZE)
        started = self._refresh_started(plan, limit)
        after: tuple[str, str, str] | None = None
        refreshed = 0
        while True:
            page = self._retry(
                partial(self._read_checkpoint_page, after, plan, limit),
                label="refresh checkpoint blobs",
            )
            for params in self._refresh_batches(self._used_blobs(page), plan):
                refreshed += self._retry(
                    partial(self._refresh_blob_rows, params), label="refresh checkpoint blobs"
                )
            if len(page) < limit:
                break
            after = self._page_end(page, after)
        self._refresh_finished(refreshed, started)

    def _read_checkpoint_page(
        self, after: tuple[str, str, str] | None, plan: _TTLPlan, limit: int
    ) -> builtins.list[DictRow]:
        """The page of checkpoints after ``after``, for the refresh in enable_ttl."""
        query, params = self._checkpoint_page(after, plan, limit)
        with self._cursor(
            timeout=self.retry_policy.ddl_attempt_timeout, label="refresh checkpoint blobs"
        ) as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def _refresh_blob_rows(self, params: builtins.list[Any]) -> int:
        """Run one UPDATE of the refresh in enable_ttl, and return the rows it changed."""
        with self._cursor(
            timeout=self.retry_policy.ddl_attempt_timeout, label="refresh checkpoint blobs"
        ) as cur:
            cur.execute(self.REFRESH_USED_BLOBS_SQL, params)
            self._sync_pipeline()
            return max(cur.rowcount, 0)

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
                transaction=len(rows) > 1,
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
                transaction=True,
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
        background CockroachDB job on the specified cron schedule. Blob rows,
        which hold channel values, expire a tenth of ttl_interval later
        (TTL_REFRESH_FRACTION), and each put brings forward those its
        checkpoint still uses. So a thread keeps its state while it is
        active, and expires once it has been idle for ttl_interval. Months
        and years are stored as days, 30 to a month and 365 to a year, so the
        interval has a fixed length.

        When the TTL setting of checkpoint_blobs is to change, it writes the
        new settings with TTL paused, so nothing is deleted, and waits
        TTL_STATE_CACHE_SECONDS plus TTL_PAUSE_MARGIN_SECONDS, or plus the
        retry policy's attempt_timeout when that is longer, so that every
        saver reads them and refreshes on its puts, and puts already under
        way finish; savers refresh while TTL is paused. Then it makes one
        pass, so threads idle at that moment keep their state too: each blob
        row that checkpoints inside the interval use, and that is older than
        the newest of them by more than the refresh window, is brought
        forward to now. Only rows left unchanged that long are rewritten. The
        pass reads every checkpoint once, TTL_REFRESH_BATCH_SIZE at a time,
        changes at most TTL_REFRESH_UPDATE_SIZE rows per UPDATE, and logs at
        INFO when it starts and ends. Then it reads the setting again and
        resumes TTL, unless another run has written a different setting
        since: that run resumes it, and this one logs at INFO. So it blocks
        for about 15 seconds by default, plus the pass, whenever the setting
        changes, on a table of any size. With the same setting in place and
        running, it only updates the cron and returns at once; any other
        change makes the whole run, a longer interval too. A run that stops
        partway leaves TTL paused, on checkpoint_blobs at least, and the next
        run finishes it, even with the same arguments; to force a pass, call
        disable_ttl first. If startup probes are tight, run it once from a
        single process or a one-off job before a rollout. Use one interval,
        and change it from one process. After upgrading from 0.5, run it
        again with the same arguments, which adds the grace and covers
        threads idle since the upgrade. Intervals over 106,751 days, the
        longest duration CockroachDB keeps, are refused.

        Safe to run from several processes at once. On CockroachDB v25.4 a TTL
        change fails while another one runs on the same table; that error is
        retried with the budget that schema_change_policy gives it, up to ten
        minutes by default, even if the policy's ``retry_on`` would refuse it.
        ``max_attempts=1`` turns this off. Other savers see the change within
        TTL_STATE_CACHE_SECONDS.

        Args:
            ttl_interval: Interval after which rows expire (e.g., '7 days', '30 days').
            cron: Cron schedule for the TTL deletion job (default: '@daily').
        """
        plan = self._ttl_plan(ttl_interval, cron)
        policy = schema_change_policy(self.retry_policy)
        try:
            before = self._retry(self._read_blob_ttl, label="read checkpoint TTL")
            if not self._refresh_needed(before, plan):
                # The same setting is in place and running; the cron may change
                for statement in plan.statements:
                    self._run_ddl(statement, "enable TTL", policy=policy)
                return
            # The new settings with TTL paused, so nothing is deleted yet
            for statement in plan.paused:
                self._run_ddl(statement, "enable TTL", policy=policy)
            # Every saver reads them again and refreshes on its puts from then on
            self._forget_ttl_state()
            self._wait_for_savers(plan)
            self._refresh_used_blobs(plan)
            # Only this run's own setting: a run that wrote another since resumes it
            current = self._retry(self._read_blob_ttl, label="read checkpoint TTL")
            if self._resume_allowed(current, plan):
                # checkpoint_blobs last, so a run that stops here is redone in full.
                # CockroachDB may skip the next scheduled TTL job once after this,
                # for the recent schema change, which delays deletion one cron period
                for statement in plan.resume:
                    self._run_ddl(statement, "enable TTL", policy=policy)
                self._pause_ended()
        finally:
            # Even a failed run may have changed a table
            self._forget_ttl_state()

    def _wait_for_savers(self, plan: _TTLPlan) -> None:
        """Wait, holding no connection, until every saver has read the TTL setting again."""
        seconds = self._savers_wait()
        self._pause_started(plan, seconds)
        _sleep(seconds)

    def disable_ttl(self) -> None:
        """Disable CockroachDB row-level TTL on checkpoint tables.

        Safe to run from several processes at once. On CockroachDB v25.4 a TTL
        change fails while another one runs on the same table; that error is
        retried with the budget that schema_change_policy gives it, up to ten
        minutes by default, even if the policy's ``retry_on`` would refuse it.
        ``max_attempts=1`` turns this off. Other savers see the change within
        TTL_STATE_CACHE_SECONDS.
        """
        policy = schema_change_policy(self.retry_policy)
        try:
            for statement in self.DISABLE_TTL_SQL:
                self._run_ddl(statement, "disable TTL", policy=policy)
        finally:
            self._forget_ttl_state()

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
