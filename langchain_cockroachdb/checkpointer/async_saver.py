"""Asynchronous CockroachDB checkpoint saver for LangGraph.

Provides the same interface as langgraph-checkpoint-postgres AsyncPostgresSaver,
adapted for CockroachDB's isolation levels (SERIALIZABLE default, READ COMMITTED supported)
and online schema changes.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
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
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb.checkpointer import _ainternal
from langchain_cockroachdb.checkpointer.base import BaseCockroachDBSaver

Conn = _ainternal.Conn


def _sanitize_conn_string(conn_string: str) -> str:
    """Convert cockroachdb:// URLs to postgresql:// for psycopg compatibility."""
    if conn_string.startswith("cockroachdb://"):
        return conn_string.replace("cockroachdb://", "postgresql://", 1)
    if conn_string.startswith("cockroachdb+psycopg://"):
        return conn_string.replace("cockroachdb+psycopg://", "postgresql://", 1)
    return conn_string


class AsyncCockroachDBSaver(BaseCockroachDBSaver):
    """Asynchronous checkpointer that stores checkpoints in CockroachDB.

    Uses raw psycopg3 async connections (not SQLAlchemy) for compatibility
    with LangGraph's checkpointer interface.

    Example:
        ```python
        from langchain_cockroachdb import AsyncCockroachDBSaver

        DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
        async with AsyncCockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            # Use with LangGraph
            graph = workflow.compile(checkpointer=checkpointer)
        ```
    """

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        pipe: AsyncPipeline | None = None,
        serde: SerializerProtocol | None = None,
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
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[AsyncCockroachDBSaver]:
        """Create a new AsyncCockroachDBSaver from a connection string.

        Args:
            conn_string: CockroachDB connection string.
                Supports cockroachdb:// and postgresql:// formats.
            pipeline: Whether to use AsyncPipeline mode.
            serde: Custom serializer protocol.

        Yields:
            AsyncCockroachDBSaver instance.
        """
        conn_string = _sanitize_conn_string(conn_string)
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield cls(conn=conn, pipe=pipe, serde=serde)
            else:
                yield cls(conn=conn, serde=serde)

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        Creates necessary tables and runs migrations. MUST be called
        the first time the checkpointer is used.
        """
        async with self._cursor() as cur:
            await cur.execute(self.MIGRATIONS[0])
            results = await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await results.fetchone()
            version = -1 if row is None else row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
                strict=False,
            ):
                await cur.execute(migration)
                await cur.execute("INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,))
            if self.pipe:
                await self.pipe.sync()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

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

        async with self._cursor() as cur:
            await cur.execute(query, params, binary=True)
            values = await cur.fetchall()
            if not values:
                return

            if to_migrate := [
                v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (
                        values[0]["thread_id"],
                        [v["parent_checkpoint_id"] for v in to_migrate],
                    ),
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)
                async for sends in cur:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        if value["channel_values"] is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value["channel_values"],
                        )

            for value in values:
                yield await self._load_checkpoint_tuple(value)

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

        async with self._cursor() as cur:
            await cur.execute(self.SELECT_SQL + where, args, binary=True)
            value = await cur.fetchone()
            if value is None:
                return None

            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                if sends := await cur.fetchone():
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )

            return await self._load_checkpoint_tuple(value)

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

        async with self._cursor(pipeline=True) as cur:
            if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
                await cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    await asyncio.to_thread(
                        self._dump_blobs,
                        thread_id,
                        checkpoint_ns,
                        blob_values,
                        blob_versions,
                    ),
                )
            await cur.execute(
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
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(query, params)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.

        Args:
            thread_id: The thread ID to delete.
        """
        async with self._cursor(pipeline=True) as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    # Sync wrappers that delegate to async methods via the event loop
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints (sync wrapper for alist)."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCockroachDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore[arg-type]
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple (sync wrapper for aget_tuple)."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCockroachDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(self.aget_tuple(config), self.loop).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint (sync wrapper for aput)."""
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes (sync wrapper for aput_writes)."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete thread (sync wrapper for adelete_thread)."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCockroachDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(self.adelete_thread(thread_id), self.loop).result()

    @asynccontextmanager
    async def _cursor(self, *, pipeline: bool = False) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager."""
        async with self.lock, _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                if self.supports_pipeline:
                    async with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    async with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    async def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
        """Convert a database row into a CheckpointTuple."""
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            {
                **value["checkpoint"],
                "channel_values": {
                    **(value["checkpoint"].get("channel_values") or {}),
                    **self._load_blobs(value["channel_values"]),
                },
            },
            value["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            await asyncio.to_thread(self._load_writes, value["pending_writes"]),
        )
