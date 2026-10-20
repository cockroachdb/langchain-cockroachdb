"""Asynchronous CockroachDB checkpoint saver for LangGraph.

Provides the same interface as langgraph-checkpoint-postgres AsyncPostgresSaver,
adapted for CockroachDB's isolation levels (SERIALIZABLE default, READ COMMITTED
supported) and online schema changes. Every operation retries transient
CockroachDB errors under the saver's RetryPolicy. All writes are UPSERTs or
DELETEs, so retrying after an ambiguous result is safe.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
from collections import defaultdict
from collections.abc import (
    AsyncIterator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
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
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb._connection import to_libpq_url
from langchain_cockroachdb._loop import LoopMessages
from langchain_cockroachdb._psycopg_ops import AsyncPsycopgOps
from langchain_cockroachdb.checkpointer import _ainternal
from langchain_cockroachdb.checkpointer.base import BaseCockroachDBSaver, connection_options
from langchain_cockroachdb.retry import RetryPolicy, schema_change_policy

logger = logging.getLogger(__name__)

Conn = _ainternal.Conn

_OWN_LOOP = (
    "Synchronous calls to AsyncCockroachDBSaver are only allowed from a "
    "different thread. From the main thread, use the async interface."
)

# What the sync methods say when they give up on the saver's event loop
_SAVER_LOOP = "The event loop this AsyncCockroachDBSaver was created on"
_SAVER_ADVICE = (
    "Call the sync methods from other threads while that loop runs, or use "
    "CockroachDBSaver in sync code."
)
_NEW_SAVER = "Create a new saver in this process."
_LOOP_MESSAGES = LoopMessages(
    closed=f"{_SAVER_LOOP} is closed. {_SAVER_ADVICE}",
    not_running=f"{_SAVER_LOOP} is not running, so the call would wait forever. {_SAVER_ADVICE}",
    closed_while_waiting=(
        f"{_SAVER_LOOP} was closed before the call finished. The work may or may not "
        f"have taken effect. {_SAVER_ADVICE}"
    ),
    stopped_while_waiting=(
        f"{_SAVER_LOOP} stopped before the call finished. The work may or may not have "
        f"taken effect. {_SAVER_ADVICE}"
    ),
    forked_loop=(
        f"{_SAVER_LOOP} was running in the parent process when this process was forked. "
        f"Its copy here can never run, so calls could never finish. {_NEW_SAVER}"
    ),
    forked_child=(
        "This AsyncCockroachDBSaver was created in process {owner_pid}, and this is "
        "process {pid}, a forked child of it. No thread runs its event loop here, so "
        "the call could never finish. "
    )
    + _NEW_SAVER,
)


def _sanitize_conn_string(conn_string: str) -> str:
    """Convert any supported URL spelling to postgresql:// for psycopg."""
    return to_libpq_url(conn_string)


class AsyncCockroachDBSaver(AsyncPsycopgOps, BaseCockroachDBSaver):
    """Asynchronous checkpointer that stores checkpoints in CockroachDB.

    Uses raw psycopg3 async connections (not SQLAlchemy) for compatibility
    with LangGraph's checkpointer interface. For production, pass a
    ``psycopg_pool.AsyncConnectionPool``: it replaces broken connections,
    while a single connection cannot recover from a dropped connection.

    On a pool, operations run concurrently, each on its own pooled
    connection. On a single connection they run one at a time. A write with
    several statements runs them in one transaction, so it is saved completely
    or not at all, even when the caller is cancelled partway through. On a
    single connection the caller also uses, an operation inside a transaction
    the caller opened joins that transaction, and its errors are raised at
    once, not retried, so the caller can retry the whole transaction.

    The sync methods (``get_tuple``, ``put`` and the others) run the async
    ones on the event loop the saver was created on, so call them from other
    threads while that loop runs. From the loop itself they raise
    ``asyncio.InvalidStateError``. When the loop is closed or stops, or in a
    forked child, they raise ``RuntimeError`` instead of waiting forever.

    Example:
        ```python
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        from langchain_cockroachdb import AsyncCockroachDBSaver, RetryPolicy

        async with AsyncConnectionPool(
            "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
            kwargs={"autocommit": True, "row_factory": dict_row},
        ) as pool:
            checkpointer = AsyncCockroachDBSaver(pool, retry_policy=RetryPolicy(max_attempts=8))
            await checkpointer.setup()
            graph = workflow.compile(checkpointer=checkpointer)
        ```
    """

    component = "checkpointer"
    _log = logger
    _own_loop_message = _OWN_LOOP
    _loop_messages = _LOOP_MESSAGES

    # Runs operations one at a time on a single connection; a pool needs no lock
    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        pipe: AsyncPipeline | None = None,
        serde: SerializerProtocol | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, "
                "not AsyncConnectionPool."
            )
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self._pid = os.getpid()
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
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
        retry_policy: RetryPolicy | None = None,
        connect_timeout: float | None = 10.0,
        prepare_threshold: int | None = 5,
        connect_kwargs: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AsyncCockroachDBSaver]:
        """Create a saver on a single connection.

        Args:
            conn_string: CockroachDB connection string (cockroachdb://, postgresql:// and similar)
            pipeline: Whether to use AsyncPipeline mode
            serde: Custom serializer protocol
            retry_policy: How operations retry (default: RetryPolicy())
            connect_timeout: Seconds to wait when connecting, or None (default: 10)
            prepare_threshold: psycopg prepared statement threshold, None to disable (default: 5)
            connect_kwargs: Extra psycopg connection options, e.g. application_name

        Yields:
            AsyncCockroachDBSaver instance.
        """
        policy = retry_policy or RetryPolicy()
        url = _sanitize_conn_string(conn_string)
        options = connection_options(url, connect_kwargs, connect_timeout, prepare_threshold)
        conn = await policy.arun(
            lambda: AsyncConnection.connect(url, **options), label="connect checkpointer"
        )
        async with conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield cls(conn=conn, pipe=pipe, serde=serde, retry_policy=policy)
            else:
                yield cls(conn=conn, serde=serde, retry_policy=policy)

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        Creates necessary tables and runs migrations. MUST be called the first
        time the checkpointer is used. Safe to run from several processes at
        once: every migration is idempotent and versions are recorded with
        ON CONFLICT DO NOTHING.
        """

        async def init_and_read_version() -> int:
            async with self._cursor(
                timeout=self.retry_policy.ddl_attempt_timeout, label="checkpointer setup"
            ) as cur:
                await cur.execute(self.MIGRATIONS[0])
                await self._sync_pipeline()
                results = await cur.execute(
                    "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
                )
                await self._sync_pipeline()
                row = await results.fetchone()
                return -1 if row is None else row["v"]

        version = await self._retry(init_and_read_version, label="checkpointer setup")

        for v in range(version + 1, len(self.MIGRATIONS)):

            async def apply_migration(migration_v: int = v) -> None:
                async with self._cursor(
                    timeout=self.retry_policy.ddl_attempt_timeout,
                    label=f"checkpointer migration {migration_v}",
                ) as cur:
                    await cur.execute(self.MIGRATIONS[migration_v])
                    await self._sync_pipeline()
                    await cur.execute(self.INSERT_MIGRATION_SQL, (migration_v,))
                    await self._sync_pipeline()

            await self._retry(apply_migration, label=f"checkpointer migration {v}")

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        Matching checkpoints are fetched inside one retried call and yielded
        afterwards, so no connection or lock is held while the caller iterates.

        Args:
            config: Base configuration for filtering checkpoints.
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
        for item in await self._retry(
            lambda: self._fetch_list(query, params), label="list checkpoints"
        ):
            yield item

    async def _fetch_list(
        self, query: str, params: builtins.list[Any]
    ) -> builtins.list[CheckpointTuple]:
        async with self._cursor(
            timeout=self.retry_policy.attempt_timeout, label="list checkpoints"
        ) as cur:
            await cur.execute(query, params, binary=True)
            values = await cur.fetchall()
            if not values:
                return []

            if to_migrate := [
                v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (values[0]["thread_id"], [v["parent_checkpoint_id"] for v in to_migrate]),
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)
                async for sends in cur:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value.setdefault("_blob_values", {}),
                        )

            groups: dict[tuple[str, str], list[str]] = defaultdict(list)
            for v in values:
                groups[(v["thread_id"], v["checkpoint_ns"])].append(v["checkpoint_id"])

            blobs_by_id: dict[str, dict[str, Any]] = {}
            writes_by_id: dict[str, list[tuple[str, str, Any]]] = {}
            for (tid, cns), cids in groups.items():
                await cur.execute(self.SELECT_BLOBS_BATCH_SQL, (tid, cns, cids), binary=True)
                blobs_by_id.update(
                    await asyncio.to_thread(self._load_blobs_batch, await cur.fetchall())
                )
                await cur.execute(self.SELECT_WRITES_BATCH_SQL, (tid, cns, cids), binary=True)
                writes_by_id.update(
                    await asyncio.to_thread(self._load_writes_batch, await cur.fetchall())
                )

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

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

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
        return await self._retry(
            lambda: self._fetch_tuple(where, args, thread_id), label="get checkpoint"
        )

    async def _fetch_tuple(
        self, where: str, args: tuple[Any, ...], thread_id: str
    ) -> CheckpointTuple | None:
        async with self._cursor(
            timeout=self.retry_policy.attempt_timeout, label="get checkpoint"
        ) as cur:
            await cur.execute(self.SELECT_SQL + where, args, binary=True)
            value = await cur.fetchone()
            if value is None:
                return None
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL, (thread_id, [value["parent_checkpoint_id"]])
                )
                if sends := await cur.fetchone():
                    blob_values: dict[str, Any] = {}
                    self._migrate_pending_sends(sends["sends"], value["checkpoint"], blob_values)
                    value["_blob_values"] = blob_values
            return await self._load_checkpoint_tuple_with_cursor(cur, value)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

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
            await asyncio.to_thread(
                self._dump_blobs, thread_id, checkpoint_ns, blob_values, blob_versions
            )
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

        async def write() -> None:
            async with self._cursor(
                pipeline=True,
                transaction=bool(blob_rows),
                timeout=self.retry_policy.attempt_timeout,
                label="put checkpoint",
            ) as cur:
                if blob_rows:
                    await cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_rows)
                await cur.execute(self.UPSERT_CHECKPOINTS_SQL, checkpoint_row)

        await self._retry(write, label="put checkpoint")
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

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
        rows = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )

        async def write() -> None:
            async with self._cursor(
                pipeline=True,
                transaction=len(rows) > 1,
                timeout=self.retry_policy.attempt_timeout,
                label="put checkpoint writes",
            ) as cur:
                await cur.executemany(query, rows)

        await self._retry(write, label="put checkpoint writes")

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.

        Args:
            thread_id: The thread ID to delete.
        """

        async def delete() -> None:
            async with self._cursor(
                pipeline=True,
                transaction=True,
                timeout=self.retry_policy.attempt_timeout,
                label="delete thread",
            ) as cur:
                await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (str(thread_id),))
                await cur.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s", (str(thread_id),)
                )
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s", (str(thread_id),)
                )

        await self._retry(delete, label="delete thread")

    # Sync wrappers that run the async methods on the loop the saver was created
    # on and wait for them, so they are for other threads
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints (sync wrapper for alist)."""
        yield from self._run_sync(
            self._alist_all(config, filter=filter, before=before, limit=limit)
        )

    async def _alist_all(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> builtins.list[CheckpointTuple]:
        """Collect what alist yields, so list() goes to the loop once."""
        return [
            item async for item in self.alist(config, filter=filter, before=before, limit=limit)
        ]

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple (sync wrapper for aget_tuple)."""
        return self._run_sync(self.aget_tuple(config))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint (sync wrapper for aput)."""
        return self._run_sync(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes (sync wrapper for aput_writes)."""
        return self._run_sync(self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        """Delete thread (sync wrapper for adelete_thread)."""
        return self._run_sync(self.adelete_thread(thread_id))

    async def _load_checkpoint_tuple_with_cursor(
        self, cur: AsyncCursor[DictRow], value: DictRow
    ) -> CheckpointTuple:
        """Convert a database row into a CheckpointTuple, fetching blobs and writes separately."""
        thread_id = value["thread_id"]
        checkpoint_ns = value["checkpoint_ns"]
        checkpoint_id = value["checkpoint_id"]
        checkpoint = value["checkpoint"]

        cv_pairs = self._get_channel_version_pairs(checkpoint)
        blob_values: dict[str, Any] = value.get("_blob_values") or {}
        if cv_pairs:
            await cur.execute(
                self.SELECT_BLOBS_SQL, (thread_id, checkpoint_ns, cv_pairs), binary=True
            )
            blob_values = {
                **blob_values,
                **await asyncio.to_thread(self._load_blobs, await cur.fetchall()),
            }

        await cur.execute(
            self.SELECT_WRITES_SQL, (thread_id, checkpoint_ns, checkpoint_id), binary=True
        )
        pending_writes = await asyncio.to_thread(self._load_writes, await cur.fetchall())

        return self._build_checkpoint_tuple(value, blob_values, pending_writes)

    async def aenable_ttl(self, ttl_interval: str = "7 days", cron: str = "@daily") -> None:
        """Enable CockroachDB row-level TTL on checkpoint tables.

        Safe to run from several processes at once. On CockroachDB v25.4 a TTL
        change fails while another one runs on the same table; that error is
        retried under the saver's retry policy, even if its ``retry_on`` would
        refuse it. ``max_attempts=1`` turns this off.

        Args:
            ttl_interval: Interval after which rows expire (e.g., '7 days', '30 days').
            cron: Cron schedule for the TTL deletion job (default: '@daily').
        """
        self._validate_ttl_params(ttl_interval, cron)
        statements = [sql.format(interval=ttl_interval, cron=cron) for sql in self.ENABLE_TTL_SQL]
        policy = schema_change_policy(self.retry_policy)
        for statement in statements:
            await self._run_ddl(statement, "enable TTL", policy=policy)

    async def adisable_ttl(self) -> None:
        """Disable CockroachDB row-level TTL on checkpoint tables.

        Safe to run from several processes at once. On CockroachDB v25.4 a TTL
        change fails while another one runs on the same table; that error is
        retried under the saver's retry policy, even if its ``retry_on`` would
        refuse it. ``max_attempts=1`` turns this off.
        """
        policy = schema_change_policy(self.retry_policy)
        for statement in self.DISABLE_TTL_SQL:
            await self._run_ddl(statement, "disable TTL", policy=policy)
