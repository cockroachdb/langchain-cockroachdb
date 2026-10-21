"""Asynchronous CockroachDB store for LangGraph long-term memory.

Provides the interface of langgraph-checkpoint-postgres's AsyncPostgresStore
on CockroachDB. It plans batches with the same code as CockroachDBStore and
runs the same SQL. Every operation retries transient errors under the
store's RetryPolicy. Every write is an UPSERT, a DELETE, or an UPDATE that
pushes back an expiry, so a retry after an ambiguous result is safe.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from langgraph.store.base import InvalidNamespaceError, Op, Result, TTLConfig
from langgraph.store.base.batch import AsyncBatchedBaseStore
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb._connection import connection_options, to_libpq_url
from langchain_cockroachdb._loop import LoopMessages
from langchain_cockroachdb._psycopg_ops import AsyncPsycopgOps
from langchain_cockroachdb.checkpointer import _ainternal
from langchain_cockroachdb.retry import RetryPolicy, schema_change_policy
from langchain_cockroachdb.store.base import (
    BaseCockroachDBStore,
    BatchPlan,
    PutPlan,
)
from langchain_cockroachdb.store.config import (
    CockroachDBIndexConfig,
    CockroachDBTTLConfig,
    PoolConfig,
    resolve_index_config,
)
from langchain_cockroachdb.store.namespaces import validate_labels
from langchain_cockroachdb.store.store import DEFAULT_SWEEP_INTERVAL_MINUTES

logger = logging.getLogger(__name__)

_OWN_LOOP = (
    "Synchronous calls to AsyncCockroachDBStore are only allowed from a "
    "different thread. From the main thread, use the async interface."
)
_STORE_LOOP = "The event loop this AsyncCockroachDBStore was created on"
_LOOP_MESSAGES = LoopMessages(
    closed=f"{_STORE_LOOP} is closed.",
    not_running=f"{_STORE_LOOP} is not running, so the call would wait forever.",
    closed_while_waiting=f"{_STORE_LOOP} was closed before the call finished.",
    stopped_while_waiting=f"{_STORE_LOOP} stopped before the call finished.",
    forked_loop=f"{_STORE_LOOP} belongs to the parent process. Create a new store here.",
    forked_child=(
        "This AsyncCockroachDBStore was created in process {owner_pid}, and this is "
        "process {pid}, a forked child of it. Create a new store here."
    ),
)


class AsyncCockroachDBStore(AsyncBatchedBaseStore, BaseCockroachDBStore, AsyncPsycopgOps):
    """Async LangGraph store that keeps long-term memory in CockroachDB.

    The async twin of CockroachDBStore: same tables, same SQL, same
    behavior. For production, pass a ``psycopg_pool.AsyncConnectionPool``
    or use ``from_conn_string`` with ``pool_config``: a pool replaces broken
    connections, while a single connection cannot recover from a dropped one.

    ``aget``, ``aput``, ``asearch`` and the other awaited calls, and the sync
    wrappers, go through AsyncBatchedBaseStore's queue, which runs one
    ``abatch`` at a time. So a pool does not run those calls in parallel, and
    a large pool does not speed them up. What the queue gives instead is
    coalescing: ops queued in the same tick run as one batch, with one round
    trip per read group and one transaction for the writes. Only direct
    ``abatch`` calls, or sync ``batch`` calls from other threads, run
    concurrently on a pool.

    The constructor must run inside a running event loop, because
    AsyncBatchedBaseStore starts a task on it there. Awaited calls made in
    the same tick (``aget``, ``aput``, ``asearch`` and the others) are
    coalesced into one ``abatch``. The sync methods (``get``, ``put`` and the
    others) are meant for other threads while that loop runs. From the loop
    itself they raise ``asyncio.InvalidStateError``.

    A batch runs its reads first, then all its puts and deletes in one
    transaction. Its gets, its searches and its listings take one round trip
    each. Semantic searches under a namespace add one before the searches, to
    find their namespaces, and one after them when a search must run again
    exactly, as the long-term memory guide explains.

    Example:
        ```python
        from langchain_cockroachdb import AsyncCockroachDBStore

        async with AsyncCockroachDBStore.from_conn_string(
            "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
        ) as store:
            await store.setup()
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```
    """

    supports_ttl: bool = True
    _log = logger
    _own_loop_message = _OWN_LOOP
    _loop_messages = _LOOP_MESSAGES

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        pipe: AsyncPipeline | None = None,
        index: CockroachDBIndexConfig | None = None,
        ttl: TTLConfig | CockroachDBTTLConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        table_name: str = "store",
        schema: str = "public",
    ) -> None:
        """Create a store on a connection or a pool, inside a running event loop.

        Args:
            conn: A psycopg AsyncConnection, or an AsyncConnectionPool
                (recommended). It must be in autocommit mode and use dict_row.
            pipe: A pipeline on ``conn``, only with a single connection
            index: Semantic search settings. When given, setup() also creates
                the vectors table and its C-SPANN index, and puts embed the
                configured fields of each item.
            ttl: TTL settings. When given, setup() also turns on CockroachDB
                row-level TTL for expired rows.
            retry_policy: How operations retry (default: RetryPolicy())
            table_name: Name of the items table (default "store")
            schema: Schema of the store's tables (default "public")

        Raises:
            RuntimeError: If no event loop is running
            ValueError: If a pipeline is given with a pool, or a name or the
                TTL or index settings are invalid
        """
        # Validated first, so a bad index fails before anything starts
        resolved_index = resolve_index_config(index)
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, "
                "not AsyncConnectionPool."
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError(
                "AsyncCockroachDBStore must be created inside a running event loop, "
                "for example in an async function. In sync code, use CockroachDBStore."
            ) from error
        super().__init__()
        self._configure(index=resolved_index, ttl=ttl, table_name=table_name, schema=schema)
        # BaseStore reads ttl_config for the default TTL and refresh_on_read
        self.ttl_config = ttl
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self._pid = os.getpid()
        self.supports_pipeline = Capabilities().has_pipeline()
        self.retry_policy = retry_policy or RetryPolicy()
        # Set while a write batch of ours is between BEGIN and COMMIT on a
        # single connection. See AsyncCockroachDBSaver for the details.
        self._batch_open = False
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        pool_config: PoolConfig | None = None,
        index: CockroachDBIndexConfig | None = None,
        ttl: TTLConfig | CockroachDBTTLConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        connect_timeout: float | None = 10.0,
        prepare_threshold: int | None = 5,
        connect_kwargs: Mapping[str, Any] | None = None,
        table_name: str = "store",
        schema: str = "public",
    ) -> AsyncIterator[AsyncCockroachDBStore]:
        """Create a store on a new connection, or on a new pool with ``pool_config``.

        Args:
            conn_string: CockroachDB connection string (cockroachdb://, postgresql:// and similar)
            pipeline: Whether to use pipeline mode on the single connection.
                Ignored with pool_config.
            pool_config: Open an AsyncConnectionPool with these settings instead of one connection
            index: Semantic search settings
            ttl: TTL settings
            retry_policy: How operations retry (default: RetryPolicy())
            connect_timeout: Seconds to wait when connecting, or None (default: 10)
            prepare_threshold: psycopg prepared statement threshold, None to disable (default: 5)
            connect_kwargs: Extra psycopg connection options, e.g. application_name
            table_name: Name of the items table (default "store")
            schema: Schema of the store's tables (default "public")

        Yields:
            AsyncCockroachDBStore instance. The connection or pool closes on exit.

        With ``pool_config``, ``pipeline`` is ignored, as in AsyncPostgresStore:
        a pipeline needs a single connection.
        """
        policy = retry_policy or RetryPolicy()
        url = to_libpq_url(conn_string)
        options = dict(connect_kwargs or {})

        def make(conn: _ainternal.Conn, pipe: AsyncPipeline | None = None) -> AsyncCockroachDBStore:
            return cls(
                conn,
                pipe=pipe,
                index=index,
                ttl=ttl,
                retry_policy=policy,
                table_name=table_name,
                schema=schema,
            )

        if pool_config is not None:
            options.update(pool_config.get("kwargs") or {})
            kwargs = connection_options(
                url, options, connect_timeout, prepare_threshold, component="store"
            )
            pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
                url,
                min_size=pool_config.get("min_size", 1),
                max_size=pool_config.get("max_size"),
                kwargs=kwargs,
                open=False,
            )
            async with pool:
                yield make(pool)
            return

        kwargs = connection_options(
            url, options, connect_timeout, prepare_threshold, component="store"
        )
        conn: AsyncConnection[DictRow] = await policy.arun(
            lambda: AsyncConnection.connect(url, **kwargs), label="connect store"
        )
        async with conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield make(conn, pipe)
            else:
                yield make(conn)

    async def setup(self) -> None:
        """Create the store's tables and run its migrations.

        MUST be called the first time the store is used. Safe to run from
        several processes at once: every migration is idempotent and versions
        are recorded with ON CONFLICT DO NOTHING. With an index, it creates
        the vectors table and its C-SPANN index (unless ``flat``) on every
        call, then checks an existing index against the configuration. With
        a TTL configuration, it then turns on CockroachDB row-level TTL on
        ``expires_at``.

        On CockroachDB v25.4 a TTL change fails while another setup() changes
        the same table. That error is retried under the store's retry policy,
        even if its ``retry_on`` would refuse it; ``max_attempts=1`` turns this
        off.

        Raises:
            ValueError: If the vectors table exists with other dims, or its
                index uses another distance. A different partition size, or
                an index while ``flat`` is set, logs a WARNING instead.
        """
        migrations = self.sql.migrations()
        version_sql = f"SELECT v FROM {self.sql.migrations_table} ORDER BY v DESC LIMIT 1"
        record_sql = (
            f"INSERT INTO {self.sql.migrations_table} (v) VALUES (%s) ON CONFLICT DO NOTHING"
        )

        async def init_and_read_version() -> int:
            async with self._cursor(
                timeout=self.retry_policy.ddl_attempt_timeout, label="store setup"
            ) as cur:
                await cur.execute(migrations[0])
                await self._sync_pipeline()
                results = await cur.execute(version_sql)
                await self._sync_pipeline()
                row = await results.fetchone()
                return -1 if row is None else row["v"]

        version = await self._retry(init_and_read_version, label="store setup")

        for v in range(version + 1, len(migrations)):

            async def apply_migration(migration_v: int = v) -> None:
                async with self._cursor(
                    timeout=self.retry_policy.ddl_attempt_timeout,
                    label=f"store migration {migration_v}",
                ) as cur:
                    await cur.execute(migrations[migration_v])
                    await self._sync_pipeline()
                    await cur.execute(record_sql, (migration_v,))
                    await self._sync_pipeline()

            await self._retry(apply_migration, label=f"store migration {v}")

        for statement in self.sql.vector_statements():
            await self._run_ddl(statement, "store vector setup")
        if self.index is not None:
            await self._check_index()
        if self._ttl_given:
            policy = schema_change_policy(self.retry_policy)
            for statement in self.sql.ttl_statements(self.ttl.cron):
                await self._run_ddl(statement, "store TTL setup", policy=policy)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Run a batch of operations.

        Every op is validated, and every put value serialized, before any
        statement runs. Gets, searches and listings run first, then all puts
        and deletes commit together in one transaction.

        Args:
            ops: GetOp, SearchOp, ListNamespacesOp and PutOp operations.

        Returns:
            One result per op, in the order of ``ops``.
        """
        plan = self._plan_batch(ops)
        if plan.search_to_embed:
            await self._embed_queries(plan)
        if plan.search_lookups:
            plan.bind_namespaces(
                await self._read_all(plan.lookup_statements(), "store search namespaces")
            )
        for label, statements, fill, session in plan.reads():
            fill(await self._read_all(statements, label, session))
        exact = plan.exact_statements()
        if exact:
            plan.fill_exact(await self._read_all(exact, "store search exact"))
        if plan.puts is not None:
            await self._run_puts(plan.puts)
        return plan.results

    async def _read_all(
        self, statements: list[tuple[str, list]], label: str, session: list[str] | None = None
    ) -> list[list[DictRow]]:
        """Run read statements as one round trip and return the rows of each.

        Every statement goes on its own cursor and all are sent before any
        result is fetched, so in pipeline mode the group costs one round
        trip. Without pipeline support they run one after another. The rows
        are fetched once the block has synced the pipeline.

        ``session`` statements run first, and then the group runs in one
        transaction: SET LOCAL only takes effect inside a transaction block.
        On v25.4, v26.2 and v26.3 it is ignored in the implicit transaction
        of a pipeline, and works between BEGIN and COMMIT sent in one batch.
        Inside a transaction the caller opened on a single connection shared
        with the store, the group nests in a savepoint, and the setting
        lasts until the caller's transaction ends.
        """

        async def read() -> list[list[DictRow]]:
            cursors: list[AsyncCursor[DictRow]] = []
            try:
                async with self._cursor(
                    pipeline=True,
                    transaction=bool(session),
                    timeout=self.retry_policy.attempt_timeout,
                    label=label,
                ) as cur:
                    for setting in session or ():
                        await cur.execute(setting, prepare=False)
                    conn = cur.connection
                    for statement in statements:
                        cursors.append(conn.cursor(binary=True, row_factory=dict_row))
                        await cursors[-1].execute(*statement)
                return [await statement_cursor.fetchall() for statement_cursor in cursors]
            finally:
                for statement_cursor in cursors:
                    await statement_cursor.close()

        return await self._retry(read, label=label)

    async def _embed_queries(self, plan: BatchPlan) -> None:
        """Embed each distinct query text of the batch once and bind the vectors.

        Runs before the first attempt of any read, so a retry reuses the
        vectors. aembed_query, not aembed_documents: models that embed
        queries and documents differently get the query form.
        """
        assert self.index is not None
        texts = plan.query_texts()
        tasks = [asyncio.ensure_future(self.index.embeddings.aembed_query(text)) for text in texts]
        try:
            vectors = await asyncio.gather(*tasks)
        except BaseException:
            # gather leaves the other calls running when one fails
            for task in tasks:
                task.cancel()
            raise
        self._bind_query_vectors(plan, texts, vectors)

    async def adrop_vector_index(self) -> None:
        """Drop the store's vector index, if it exists.

        Run setup() afterwards to build it again with the store's current
        configuration, for example to change the distance. Searches scan
        the vectors until the new index is built.
        """
        await self._run_ddl(self.sql.drop_vector_index(), "store drop vector index")

    async def _check_index(self) -> None:
        """Read the vectors table's definition and check it against the configuration."""

        async def read() -> str:
            async with self._cursor(
                timeout=self.retry_policy.attempt_timeout, label="store index check"
            ) as cur:
                await cur.execute(self.sql.show_create_vectors())
                await self._sync_pipeline()
                row = await cur.fetchone()
                return "" if row is None else row["create_statement"]

        self._check_vector_index(await self._retry(read, label="store index check"))

    async def _run_puts(self, plan: PutPlan) -> None:
        # Embed once, before the first attempt: a retry reuses the vectors
        embeddings: list[list[float]] = []
        if self.index is not None and plan.embed_requests:
            embeddings = await self.index.embeddings.aembed_documents(
                [text for _, _, _, text in plan.embed_requests]
            )
        statements = self._write_statements(plan, self._vector_rows(plan, embeddings))
        if not statements:
            return

        async def write() -> None:
            async with self._cursor(
                pipeline=True,
                transaction=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store put",
            ) as cur:
                for statement in statements:
                    await cur.execute(*statement)

        await self._retry(write, label="store put")

    async def adelete_namespace(self, namespace_prefix: tuple[str, ...]) -> int:
        """Delete every item in a namespace and its descendants.

        ``("user", "a")`` deletes ``("user", "a")`` and ``("user", "a", "b")``,
        not ``("user", "ab")``. One statement deletes the items and their vectors.

        Args:
            namespace_prefix: The namespace to delete. It cannot be empty.

        Returns:
            The number of items deleted. If a retry follows a commit whose
            result was lost, the count is that of the retry, usually 0.

        Raises:
            InvalidNamespaceError: If the prefix is empty or a label is invalid
        """
        prefix = validate_labels(namespace_prefix)
        if not prefix:
            raise InvalidNamespaceError(
                "adelete_namespace needs a non-empty namespace prefix. To clear the whole "
                "store on purpose, delete each top-level namespace from alist_namespaces("
                "max_depth=1), or drop the store's tables."
            )
        statement = self.sql.delete_namespace(prefix)

        async def write() -> int:
            # One statement, so items and vectors commit together. pipeline=True
            # makes a component-wide pipeline sync, which commits it
            async with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store delete namespace",
            ) as cur:
                await cur.execute(*statement)
                return len(await cur.fetchall())

        return await self._retry(write, label="store delete namespace")

    async def sweep_ttl(self) -> int:
        """Delete expired items now, without waiting for CockroachDB's TTL job.

        Returns:
            The number of items deleted.
        """

        async def sweep() -> int:
            count = 0
            async with self._cursor(
                timeout=self.retry_policy.attempt_timeout, label="store TTL sweep"
            ) as cur:
                for statement in self.sql.sweep():
                    await cur.execute(*statement)
                    await self._sync_pipeline()
                    # The items statement comes last
                    count = cur.rowcount
            return max(count, 0)

        return await self._retry(sweep, label="store TTL sweep")

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """Sweep expired items periodically in a background task.

        CockroachDB's row-level TTL job, turned on by setup() with a TTL
        configuration, already deletes expired rows. The sweeper deletes them
        sooner. Expired items are never returned either way.

        Args:
            sweep_interval_minutes: Minutes between sweeps. Defaults to the
                TTL configuration's sweep_interval_minutes, else 5.

        Returns:
            The sweeper task. It finishes when the sweeper stops. Without a
            TTL configuration, a task that finishes at once.
        """
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task is not None and not self._ttl_sweeper_task.done():
            logger.info("TTL sweeper task is already running")
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes
            or self.ttl_config.get("sweep_interval_minutes")
            or DEFAULT_SWEEP_INTERVAL_MINUTES
        )
        logger.info("Starting store TTL sweeper with interval %s minutes", interval)

        async def sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    await asyncio.wait_for(self._ttl_stop_event.wait(), timeout=interval * 60)
                    break
                except asyncio.TimeoutError:
                    pass
                try:
                    expired = await self.sweep_ttl()
                    if expired > 0:
                        logger.info("Store swept %d expired items", expired)
                except Exception:
                    logger.exception("Store TTL sweep iteration failed")

        task = asyncio.create_task(sweep_loop(), name="ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper task if it is running.

        Args:
            timeout: Seconds to wait for the task to stop, or None to wait.

        Returns:
            True if the task stopped or was not running, False on timeout.
        """
        task = self._ttl_sweeper_task
        if task is None or task.done():
            return True
        logger.info("Stopping TTL sweeper task")
        self._ttl_stop_event.set()
        try:
            # shield: a timeout must not cancel the sweep that is running
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for TTL sweeper task to stop")
            return False
        self._ttl_sweeper_task = None
        logger.info("TTL sweeper task stopped")
        return True

    async def adisable_ttl(self) -> None:
        """Turn off CockroachDB row-level TTL on the store's tables.

        Expired items are still left out of reads while omit_expired is on.
        Safe to run from several processes at once. On CockroachDB v25.4 a
        TTL change fails while another schema change runs on the same table;
        that error is retried under the store's retry policy, even if its
        ``retry_on`` would refuse it. ``max_attempts=1`` turns this off.
        """
        policy = schema_change_policy(self.retry_policy)
        for statement in self.sql.disable_ttl_statements():
            await self._run_ddl(statement, "store disable TTL", policy=policy)

    def __del__(self) -> None:
        # AsyncBatchedBaseStore.__del__ reads _task, which a failed __init__ never set
        if getattr(self, "_task", None) is not None:
            super().__del__()

    async def __aenter__(self) -> AsyncCockroachDBStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Signal the sweeper to stop without waiting for it, as
        # AsyncPostgresStore does. It finishes on its own.
        if self._ttl_sweeper_task is not None:
            self._ttl_stop_event.set()
