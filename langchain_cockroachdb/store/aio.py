"""Asynchronous CockroachDB store for LangGraph long-term memory.

Provides the interface of langgraph-checkpoint-postgres's AsyncPostgresStore
on CockroachDB. It plans batches, reconciliations and conditional writes
with the same code as CockroachDBStore and runs the same SQL. Every
operation retries transient errors under the store's RetryPolicy. Every
write is an UPSERT, a DELETE, an UPDATE that pushes back an expiry or
confirms a memory once per confirmation id, or an INSERT into history that
skips versions and write ids already there, so a retry after an ambiguous
result is safe.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from types import TracebackType
from typing import Any, Literal, TypeVar

import psycopg
from langgraph.store.base import (
    NOT_PROVIDED,
    Item,
    NamespacePath,
    NotProvided,
    Op,
    Result,
    SearchItem,
    TTLConfig,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb import _psycopg_conn
from langchain_cockroachdb._connection import connection_options, to_libpq_url
from langchain_cockroachdb._loop import LoopMessages
from langchain_cockroachdb._psycopg_ops import AsyncPsycopgOps
from langchain_cockroachdb.retry import RetryPolicy, may_have_committed, schema_change_policy
from langchain_cockroachdb.store.base import (
    BaseCockroachDBStore,
    BatchPlan,
    ConditionalPlan,
    PutPlan,
    SchemaCheck,
    _carries_confirm_id,
    _check_limit,
    _check_policy,
    _memory_prefix,
    _namespace_to_delete,
    _resolve_confirm_ttl,
    _row_to_record,
    _row_to_version,
    _validate_confirm_ttl,
)
from langchain_cockroachdb.store.config import (
    DEFAULT_RECONCILE_SCAN_LIMIT,
    CockroachDBIndexConfig,
    CockroachDBTTLConfig,
    HistoryConfig,
    PoolConfig,
    resolve_history_config,
    resolve_index_config,
    resolve_reconcile_scan_limit,
)
from langchain_cockroachdb.store.reconcile import (
    ConditionalWrite,
    Conflicts,
    Decision,
    MemoryRecord,
    MemoryVersion,
    Policy,
    ReconcileResult,
)
from langchain_cockroachdb.store.sql import VectorRow
from langchain_cockroachdb.store.store import DEFAULT_SWEEP_INTERVAL_MINUTES

logger = logging.getLogger(__name__)

T = TypeVar("T")

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
    others) are meant for other threads while that loop runs. They run the
    async methods on it, so ``get``, ``put``, ``search``, ``delete`` and
    ``list_namespaces`` join the same queue, while ``batch`` runs ``abatch``
    directly. From the loop itself they raise ``asyncio.InvalidStateError``. When the loop is closed or not
    running, or the call comes from a forked child of the process that
    created the store, they raise ``RuntimeError`` instead of waiting
    forever, as AsyncCockroachDBSaver's sync methods do.

    A batch runs its reads first, then all its puts and deletes in one
    transaction. Its gets, its searches and its listings take one round trip
    each. On a store with a vector index, semantic searches under a namespace
    add one before the searches, to find their namespaces, and one after them
    when a search must run again exactly. A flat store adds neither. The
    long-term memory guide explains why.

    With ``history``, a batch that overwrites or deletes a memory first
    copies the version it ends into a history table, in its transaction.
    ``adelete`` then keeps the memory in history; ``apurge`` removes a memory
    with its history, and ``ahistory`` lists the old versions.

    ``afind_conflicts``, ``aresolve``, ``areconcile``, ``aconfirm``,
    ``arecord``, ``aput_if`` and ``adelete_if`` are CockroachDBStore's
    methods of the same names without the prefix, as coroutines. They run on
    the caller's event loop, not through the queue, so on a pool they run
    concurrently. A reconcile policy and a confirm TTL function are plain
    functions that run on the event loop inside the transaction.

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
        conn: _psycopg_conn.AsyncConn,
        *,
        pipe: AsyncPipeline | None = None,
        index: CockroachDBIndexConfig | None = None,
        ttl: TTLConfig | CockroachDBTTLConfig | None = None,
        history: bool | HistoryConfig | None = None,
        reconcile_scan_limit: int = DEFAULT_RECONCILE_SCAN_LIMIT,
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
            history: Version history: True, or a HistoryConfig to give old
                versions a TTL. setup() then creates the history table, and
                every batch keeps the versions it overwrites or deletes there.
                Default None, no history.
            reconcile_scan_limit: The most vectors a namespace may hold for
                find_conflicts to rank every one of them, by primary key
                (default 2000). A namespace with more finds its similar
                candidates through the C-SPANN index, and ranks every vector
                only when the index's window may have left matches out.
            retry_policy: How operations retry (default: RetryPolicy())
            table_name: Name of the items table (default "store")
            schema: Schema of the store's tables (default "public")

        Raises:
            RuntimeError: If no event loop is running
            ValueError: If a pipeline is given with a pool, or a name or the
                TTL, index, history or reconcile_scan_limit settings are invalid
        """
        # Validated first, so a bad index fails before anything starts
        resolved_index = resolve_index_config(index)
        resolved_history = resolve_history_config(history)
        scan_limit = resolve_reconcile_scan_limit(reconcile_scan_limit)
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
        self._configure(
            index=resolved_index,
            ttl=ttl,
            table_name=table_name,
            schema=schema,
            history=resolved_history,
            reconcile_scan_limit=scan_limit,
        )
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
        history: bool | HistoryConfig | None = None,
        reconcile_scan_limit: int = DEFAULT_RECONCILE_SCAN_LIMIT,
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
            history: Version history settings (default None, no history)
            reconcile_scan_limit: The most vectors a namespace may hold for
                find_conflicts to rank every one of them (default 2000)
            retry_policy: How operations retry (default: RetryPolicy())
            connect_timeout: Seconds to wait when connecting, or None (default: 10)
            prepare_threshold: psycopg prepared statement threshold, None to disable
                (default: 5). Statements sent in a pipeline are never prepared
            connect_kwargs: Extra psycopg connection options, e.g. application_name
            table_name: Name of the items table (default "store")
            schema: Schema of the store's tables (default "public")

        Yields:
            AsyncCockroachDBStore instance. The connection or pool closes on exit.

        With ``pool_config``, ``pipeline`` is ignored, as in AsyncPostgresStore:
        a pipeline needs a single connection.

        Raises:
            ValueError: If the index, history or reconcile_scan_limit settings
                are invalid, before any connection is opened
        """
        # Checked before connecting, so a bad setting costs no connection and
        # no connect retries. The store reuses the embedding model this built
        resolved_index = resolve_index_config(index)
        resolve_history_config(history)
        resolve_reconcile_scan_limit(reconcile_scan_limit)
        if index is not None and resolved_index is not None:
            index = index.copy()
            index["embed"] = resolved_index.embeddings
        policy = retry_policy or RetryPolicy()
        url = to_libpq_url(conn_string)
        options = dict(connect_kwargs or {})

        def make(
            conn: _psycopg_conn.AsyncConn, pipe: AsyncPipeline | None = None
        ) -> AsyncCockroachDBStore:
            return cls(
                conn,
                pipe=pipe,
                index=index,
                ttl=ttl,
                history=history,
                reconcile_scan_limit=reconcile_scan_limit,
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
        history, it creates the history table, and with ``ttl_days`` turns on
        CockroachDB row-level TTL there. With a TTL configuration, it then
        turns on CockroachDB row-level TTL on ``expires_at``.

        On CockroachDB v25.4 a TTL change fails while another setup() changes
        the same table. That error is retried under the store's retry policy,
        even if its ``retry_on`` would refuse it; ``max_attempts=1`` turns this
        off.

        An application role that did not run setup() needs SELECT, INSERT,
        UPDATE and DELETE on the store's items and vectors tables, and on
        the history table when history is on. Without SELECT on the migrations
        table, or without a migrations table, as when the tables were created
        without setup(), the store finds migration 2 through the items table's
        columns instead. On a multi-node cluster, one write can then fail in the
        seconds after setup() applies migration 2. Running setup() before
        rolling out 0.6.0, or granting SELECT on the migrations table, avoids
        that. The first store call inside a transaction the caller opened gets
        the permission error once. Only the SELECT grant avoids it.

        Raises:
            ValueError: If the vectors table exists with other dims, or its
                index uses another distance. A different partition size, or
                an index while ``flat`` is set, logs a WARNING instead.
        """
        migrations = self.sql.migrations()
        version_sql = self.sql.migration_version()
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

        # A write before setup() may have cached the schema without
        # migration 2. The next write reads it again.
        self._columns_ready = None

        for statement in self.sql.vector_statements():
            await self._run_ddl(statement, "store vector setup")
        if self.index is not None:
            await self._check_index()
        if self._history is not None:
            policy = schema_change_policy(self.retry_policy)
            for statement in self.sql.history_statements(self._history.ttl_days):
                await self._run_ddl(statement, "store history setup", policy=policy)
            # A write before setup() may have found no history table. The
            # next write reads the schema again.
            self._history_ready = None
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

    # The sync methods. AsyncBatchedBaseStore's versions wait on the loop with
    # no limit, so a stopped loop or a forked child would hang them.
    # _run_sync checks the loop first and while it waits.

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Run ``abatch`` on the store's event loop, from another thread.

        Raises:
            asyncio.InvalidStateError: If called from the store's loop
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or in a forked child; or, before
                any of these checks, if called from inside a reconcile policy
                or a confirm TTL function of the store
        """
        return self._run_sync(self.abatch(ops))

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Run ``aget`` on the store's event loop, from another thread.

        Raises:
            asyncio.InvalidStateError: If called from the store's loop
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or in a forked child; or, before
                any of these checks, if called from inside a reconcile policy
                or a confirm TTL function of the store
        """
        return self._run_sync(self.aget(namespace, key, refresh_ttl=refresh_ttl))

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Run ``asearch`` on the store's event loop, from another thread.

        Raises:
            asyncio.InvalidStateError: If called from the store's loop
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or in a forked child; or, before
                any of these checks, if called from inside a reconcile policy
                or a confirm TTL function of the store
        """
        return self._run_sync(
            self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        )

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Run ``aput`` on the store's event loop, from another thread.

        Raises:
            asyncio.InvalidStateError: If called from the store's loop
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or in a forked child; or, before
                any of these checks, if called from inside a reconcile policy
                or a confirm TTL function of the store
        """
        self._run_sync(self.aput(namespace, key, value, index, ttl=ttl))

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Run ``adelete`` on the store's event loop, from another thread.

        Raises:
            asyncio.InvalidStateError: If called from the store's loop
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or in a forked child; or, before
                any of these checks, if called from inside a reconcile policy
                or a confirm TTL function of the store
        """
        self._run_sync(self.adelete(namespace, key))

    def list_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Run ``alist_namespaces`` on the store's event loop, from another thread.

        Raises:
            asyncio.InvalidStateError: If called from the store's loop
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or in a forked child; or, before
                any of these checks, if called from inside a reconcile policy
                or a confirm TTL function of the store
        """
        return self._run_sync(
            self.alist_namespaces(
                prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset
            )
        )

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """AsyncPsycopgOps._run_sync, refused while this thread runs a reconcile policy of the store.

        A policy and a confirm TTL function are plain functions, so the sync
        methods are the store calls they can make. They run on the store's
        event loop, inside a transaction of the store, so such a call fails
        at once, with the RuntimeError that CockroachDBStore raises, before
        the check for the store's own loop would raise InvalidStateError.
        """
        try:
            self._refuse_inside_user_callable()
        except RuntimeError:
            coro.close()
            raise
        return super()._run_sync(coro)

    @asynccontextmanager
    async def _cursor(
        self,
        *,
        pipeline: bool = False,
        transaction: bool = False,
        timeout: float | None = None,
        label: str | None = None,
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """AsyncPsycopgOps._cursor, refused while this thread runs a reconcile policy of the store.

        Every connection checkout of the store comes through here. Inside a
        policy the store holds its connection, so a coroutine of the store
        that a nested event loop runs from the policy fails here, instead of
        waiting forever for that connection or reading outside the
        transaction on a pool: see _refuse_inside_user_callable.
        """
        self._refuse_inside_user_callable()
        async with super()._cursor(
            pipeline=pipeline, transaction=transaction, timeout=timeout, label=label
        ) as cur:
            yield cur

    @staticmethod
    async def _send(
        conn: AsyncConnection[DictRow],
        statements: Sequence[tuple[str, list]],
        cursors: list[AsyncCursor[DictRow]],
    ) -> list[AsyncCursor[DictRow]]:
        """Execute each statement on a cursor of its own, without fetching, and return the cursors.

        They are added to ``cursors`` as well, for the caller to close. In a
        pipeline, statements sent together take one round trip, and fetching
        their rows flushes without a sync, so an open transaction stays open.
        """
        sent: list[AsyncCursor[DictRow]] = []
        for statement in statements:
            sent.append(conn.cursor(binary=True, row_factory=dict_row))
            cursors.append(sent[-1])
            await sent[-1].execute(*statement)
        return sent

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
        queries and documents differently get the query form. Inside a
        reconcile policy it refuses before calling the model.
        """
        assert self.index is not None
        self._refuse_inside_user_callable()
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

    async def _aschema(self, *, refresh: bool = False) -> SchemaCheck:
        """What the schema check knows: migration 2's columns and the history table.

        It reads whether setup() has recorded migration 2, and on a store
        with history whether the history table exists, in one round trip. A
        role denied the migrations table, or a store with no migrations
        table, reads the items table's columns instead (see
        _schema_statements). It reads them the first time, and
        again when an answer it found missing is older than
        _SCHEMA_RECHECK_SECONDS, or ``refresh`` is set. What it found present
        it trusts for good.
        """
        schema = self._cached_schema(refresh=refresh)
        if schema is None:
            schema = self._cache_schema(await self._aread_schema())
        return schema

    async def _aread_schema(self) -> list[list[DictRow]]:
        """Send the schema check, with the column read if the migrations read fails.

        The migrations read falls back when the store's role has no grant on
        the migrations table (42501), or there is no migrations table
        (42P01), as when the tables were created without setup(). Only the
        schema check's own round trip is caught: either error from any other
        statement still raises.
        """
        read_migrations = not self._migrations_unreadable
        try:
            return await self._read_all(
                self._schema_statements(read_migrations=read_migrations), "store schema check"
            )
        except (psycopg.errors.InsufficientPrivilege, psycopg.errors.UndefinedTable) as error:
            if not self._fall_back_to_columns(error, read_migrations=read_migrations):
                raise
        return await self._read_all(
            self._schema_statements(read_migrations=False), "store schema check"
        )

    async def _areconcile_columns(self, *, refresh: bool = False) -> bool:
        """Whether migration 2's columns exist, as the schema check last read it.

        It reads the schema the first time, and again when it last found the
        columns missing and that answer is older than _SCHEMA_RECHECK_SECONDS,
        or ``refresh`` is set. Methods that need the columns pass ``refresh``,
        so they see a setup() that another process ran since. Until setup()
        applies migration 2, puts write with the 0.5 statements and those
        methods raise ValueError.
        """
        return (await self._aschema(refresh=refresh)).columns

    async def _ahistory_schema(self) -> SchemaCheck:
        """The schema check for an operation that needs the history table.

        It runs with ``refresh``: what it found present it trusts for good,
        and what it found missing it reads again, once, so it sees a setup()
        that another process ran since before it refuses.

        Raises:
            ValueError: If the store keeps no history, or setup() has not
                created the history table yet.
        """
        self._require_history()
        schema = await self._aschema(refresh=True)
        self._require_history_table(schema)
        return schema

    async def _ahistory_table_exists(self) -> bool:
        """Whether the history table exists, for adelete_namespace on a store without history.

        True is trusted for good. Until then every call reads. Only
        adelete_namespace asks, and it is a rare call: plain writes never do.
        """
        if self._history_ready is True:
            return True
        [rows] = await self._read_all([self.sql.history_exists()], "store history check")
        return self._cache_history(rows)

    async def _embed_put(self, plan: PutPlan) -> list[VectorRow]:
        """Embed the texts a put plan writes, once, and pair the vectors with their rows.

        Puts, afind_conflicts, areconcile and aput_if call it before their
        first attempt, so a retry reuses the vectors. All embed documents,
        with aembed_documents: the vectors are the ones the write stores.
        Inside a reconcile policy it refuses before calling the model.
        """
        self._refuse_inside_user_callable()
        embeddings: list[list[float]] = []
        if self.index is not None and plan.embed_requests:
            embeddings = await self.index.embeddings.aembed_documents(
                [text for _, _, _, text in plan.embed_requests]
            )
        return self._vector_rows(plan, embeddings)

    async def _run_puts(self, plan: PutPlan) -> None:
        # The schema check comes first. With history, a write refuses to run
        # until setup() has created the history table, since it could not
        # keep what it replaces, and a refused batch costs no embedding call
        schema = (
            await self._ahistory_schema() if self._history is not None else await self._aschema()
        )
        vector_rows = await self._embed_put(plan)
        # The write id too: a retry after a commit whose result was lost
        # finds it on the rows and changes nothing
        if schema.columns:
            plan.write_id = str(uuid.uuid4())
        statements = self._write_statements(plan, vector_rows)
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
        not ``("user", "ab")``. One statement deletes the items, their
        vectors and their history. A store without history deletes history
        too when the history table exists, as when the store kept history
        before, so nothing of the namespace is left.

        Args:
            namespace_prefix: The namespace to delete. It cannot be empty.

        Returns:
            The number of items deleted. If a retry follows a commit whose
            result was lost, the count is that of the retry, usually 0.

        Raises:
            InvalidNamespaceError: If the prefix is empty or a label is invalid
            ValueError: If the store keeps history and setup() has not
                created the history table
        """
        prefix = _namespace_to_delete(namespace_prefix, "adelete_namespace", "alist_namespaces")
        if self._history is not None:
            await self._ahistory_schema()
            history = True
        else:
            history = await self._ahistory_table_exists()
        statement = self.sql.delete_namespace(prefix, history=history)

        async def write() -> int:
            # One statement, so items, vectors and history commit together. pipeline=True
            # makes a component-wide pipeline sync, which commits it
            async with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store delete namespace",
            ) as cur:
                await cur.execute(*statement)
                return len(await cur.fetchall())

        return await self._retry(write, label="store delete namespace")

    async def ahistory(
        self, namespace: tuple[str, ...], key: str, *, limit: int = 20
    ) -> list[MemoryVersion]:
        """The old versions of a memory, newest first.

        A store with history keeps every version that a write ends: one a
        put overwrites (reason "overwritten"), one a delete ends
        ("deleted"), and one that reconcile or resolve supersedes
        ("superseded", with ``superseded_by`` naming the memory that
        replaced it). A new memory that reconcile or resolve records as
        already superseded by a newer fact is kept there too, as
        "superseded". The live value is not one of them; aget() returns it.

        Args:
            namespace: The memory's namespace.
            key: The memory's key.
            limit: Most versions to return (default 20).

        Returns:
            Up to ``limit`` versions, highest version first, each with the
            confirmations it had when it ended. Versions past the history's
            ``ttl_days`` are left out.

        Raises:
            ValueError: If the store keeps no history (create it with
                history=True and run setup()), setup() has not created the
                history table, or ``limit`` is not a non-negative int
            InvalidNamespaceError: If the namespace is empty or a label is invalid
        """
        prefix = _memory_prefix(namespace, "ahistory")
        _check_limit(limit)
        await self._ahistory_schema()
        statement = self.sql.history_versions(prefix, str(key), limit=limit)
        [rows] = await self._read_all([statement], "store history")
        return [_row_to_version(tuple(namespace), str(key), row) for row in rows]

    async def apurge(self, namespace: tuple[str, ...], key: str) -> bool:
        """Forget a memory: delete the item, its vectors and its history.

        On a store with history, adelete() keeps the version it ends, so
        apurge is how to remove a memory for good. One statement deletes all
        of it.

        Args:
            namespace: The memory's namespace.
            key: The memory's key.

        Returns:
            Whether anything was deleted: the memory, or only its history
            after a delete. If a retry follows a commit whose result was
            lost, the retry finds nothing and returns False.

        Raises:
            ValueError: If the store keeps no history (create it with
                history=True and run setup(); without history, adelete()
                already removes a memory and its vectors), or setup() has
                not created the history table
            InvalidNamespaceError: If the namespace is empty or a label is invalid
        """
        prefix = _memory_prefix(namespace, "apurge")
        await self._ahistory_schema()
        statement = self.sql.purge(prefix, str(key))

        async def write() -> bool:
            # One statement, so the item, its vectors and its history commit
            # together. pipeline=True makes a component-wide pipeline sync
            async with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store purge",
            ) as cur:
                await cur.execute(*statement)
                row = await cur.fetchone()
                return row is not None and bool(row["removed"])

        return await self._retry(write, label="store purge")

    async def afind_conflicts(
        self,
        namespace: tuple[str, ...],
        key: str | None,
        value: dict[str, Any],
        *,
        slot: str | None = None,
        threshold: float | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        index: Literal[False] | list[str] | None = None,
        ttl: float | None | NotProvided = NOT_PROVIDED,
        provenance: dict[str, Any] | None = None,
        valid_at: datetime | None = None,
    ) -> Conflicts:
        """Find the memories a new one may conflict with, without writing anything.

        The async form of CockroachDBStore.find_conflicts, which describes
        the candidates in full. They are live memories in ``namespace``
        itself, for one of three reasons, strongest first: the memory under
        ``key``; with ``slot``, the memories whose value holds the same value
        in that field; and on a store with an index, the memories whose
        vectors score at or above ``threshold`` against the new memory's,
        with the store's score function. Candidates come strongest reason
        first, then best score, then key. Expired memories are never
        candidates, and no read pushes back an expiry.

        The index fields of ``value`` are embedded once, with
        aembed_documents, and the embedding is kept for aresolve. Then two
        round trips read the candidates, or three when the C-SPANN index's
        window may have left matches out.

        Args:
            namespace: The new memory's namespace. It cannot be empty.
            key: The new memory's key, or None for a new uuid4.
            value: The new memory's value.
            slot: A top-level field of ``value`` naming the fact it states,
                holding a string, number or boolean, or None.
            threshold: The lowest score of a similar candidate. On a cosine
                store it defaults to 0.8 and must lie between -1 and 1. On an
                l2 or inner_product store, where scores are negated distances,
                it must be given. Ignored without an index or with
                ``index=False``.
            limit: Most similar candidates (default 10). The key and slot
                matches come on top of them; 0 keeps only those.
            filter: Narrows the slot and similar candidates by value, as in
                search. The memory under ``key`` is a candidate whatever it says.
            index: Which fields of ``value`` to embed, as in put. False embeds
                nothing and finds no similar candidates.
            ttl: Minutes the new memory lives once written, or None for no
                expiry. Defaults to the store's default TTL.
            provenance: Where the new memory comes from, as a JSON object.
            valid_at: When the fact became true, as a timezone-aware datetime.
                By default the database's current time, raised to the latest
                candidate's updated_at if that is later.

        Returns:
            The new memory as a Proposal, its candidates, and the version
            and updated_at of the live memory under ``key``. Nothing was written.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid
            ValueError: If another argument is invalid (above), or setup() has
                not added the columns this method needs
        """
        plan = self._plan_proposal(
            namespace,
            key,
            value,
            slot=slot,
            threshold=threshold,
            limit=limit,
            filter=filter,
            index=index,
            ttl=ttl,
            provenance=provenance,
            valid_at=valid_at,
            method="afind_conflicts",
        )
        self._require_columns(await self._areconcile_columns(refresh=True))
        vector_rows = await self._embed_put(plan.put)
        read = self._key_slot_read(
            plan, await self._read_all(self._key_slot_statements(plan), "store find conflicts")
        )
        similar = self._similar_read(plan, vector_rows, read)
        statements = similar.statements()
        found: Sequence[Sequence[dict[str, Any]]] = (
            await self._read_all(statements, "store find conflicts similar", similar.session)
            if statements
            else []
        )
        exact = similar.exact_statements(found)
        if exact:
            found = similar.fill_exact(
                found, await self._read_all(exact, "store find conflicts exact")
            )
        return self._conflicts(plan, vector_rows, read, similar, found)

    async def arecord(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None:
        """Read a live memory with its version, provenance, valid time and confirmations.

        Reading a memory does not confirm it, so its expiry is never pushed back.

        Args:
            namespace: The memory's namespace.
            key: The memory's key.

        Returns:
            The memory, or None if no live memory has that key: there is
            none, or it expired.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid
            ValueError: If setup() has not added the columns this method needs
        """
        prefix = _memory_prefix(namespace, "arecord")
        self._require_columns(await self._areconcile_columns(refresh=True))
        [rows] = await self._read_all([self.sql.record(prefix, str(key))], "store record")
        return _row_to_record(tuple(namespace), rows[0]) if rows else None

    async def aconfirm(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        ttl: float | None | NotProvided | Callable[[MemoryRecord], float | None] = NOT_PROVIDED,
    ) -> bool:
        """Mark a live memory as confirmed, renewing its expiry and counting confirmations.

        The async form of CockroachDBStore.confirm. This counts the
        confirmation and renews the expiry. That way expiry can follow how
        often a fact is confirmed, not how often it is read. Confirming does
        not change the version or updated_at, so it never invalidates a
        pending afind_conflicts. Confirmations belong to the memory's value:
        a write that changes it starts them again from none. Each call picks
        a confirmation id of its own, so a retry after a lost commit counts
        nothing again, and the memory's write id stays as it is.

        Args:
            namespace: The memory's namespace.
            key: The memory's key.
            ttl: How long the memory stays after confirmation. NOT_PROVIDED
                renews from the row's own TTL. None removes the expiry. A
                number is minutes. A callable takes the MemoryRecord read in
                the same transaction and returns a number of minutes or None.
                It is a plain function that runs on the event loop: keep it
                fast and deterministic, since it runs again on each retry,
                and it must not call the store. An async function is
                refused before any I/O.

        Returns:
            True if a live memory was confirmed, False if no live memory has
            that key: there is none, or it expired.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid
            ValueError: If ttl is not NOT_PROVIDED, None, a positive finite
                number of minutes, or a plain callable; if setup() has not
                added the columns this method needs; or if the callable
                returns an invalid value (not None or a positive finite
                number, a coroutine included)
        """
        prefix = _memory_prefix(namespace, "aconfirm")
        _validate_confirm_ttl(ttl)
        self._require_columns(await self._areconcile_columns(refresh=True))

        # Choose the confirm id before the first attempt
        confirm_id = str(uuid.uuid4())
        key_str = str(key)

        async def attempt() -> bool:
            cursors: list[AsyncCursor[DictRow]] = []
            try:
                async with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label="store confirm",
                ) as cur:
                    conn = cur.connection

                    # If ttl is a callable, read the record first
                    resolved_ttl: float | None | NotProvided
                    if callable(ttl):
                        read_sent = await self._send(
                            conn, [self.sql.record(prefix, key_str)], cursors
                        )
                        rows = await read_sent[0].fetchall()
                        if not rows:
                            # No live memory to confirm
                            return False
                        record = _row_to_record(tuple(namespace), rows[0])
                        # If the record already carries this confirm id, an earlier
                        # attempt committed. Return True without calling the callable again
                        if _carries_confirm_id(rows[0], confirm_id):
                            return True
                        resolved_ttl = self._run_user_callable(_resolve_confirm_ttl, ttl, record)  # type: ignore[assignment]
                    else:
                        resolved_ttl = ttl

                    # Execute the confirm and the confirm id check in the same transaction
                    sent = await self._send(
                        conn,
                        [
                            self.sql.confirm(
                                prefix, key_str, confirm_id=confirm_id, ttl=resolved_ttl
                            ),
                            self.sql.confirm_check(prefix, key_str, confirm_id=confirm_id),
                        ],
                        cursors,
                    )

                # Transaction committed. Interpret the result
                rowcount = sent[0].rowcount
                check_rows = await sent[1].fetchall()
                return self._confirm_result(rowcount, check_rows)
            finally:
                for cursor in cursors:
                    await cursor.close()

        return await self._retry(attempt, label="store confirm")

    async def areconcile(
        self,
        namespace: tuple[str, ...],
        key: str | None,
        value: dict[str, Any],
        *,
        policy: Policy,
        slot: str | None = None,
        threshold: float | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        index: Literal[False] | list[str] | None = None,
        ttl: float | None | NotProvided = NOT_PROVIDED,
        provenance: dict[str, Any] | None = None,
        valid_at: datetime | None = None,
    ) -> ReconcileResult:
        """Write a memory and settle the memories it may conflict with, in one transaction.

        The async form of CockroachDBStore.reconcile, which describes the
        decisions and the result in full. The candidates are the ones
        afind_conflicts finds. They are read in the transaction, ``policy``
        decides what to do with the new memory and with each candidate, and
        the decision is applied before the transaction commits. Serializable
        isolation covers all of it: if another writer changes what the policy
        saw, one of the two transactions retries, and its policy sees the
        other's write. Two agents that reconcile contradicting memories for
        one user cannot both win.

        ``policy`` is a plain function, not an async one, which is refused
        before any I/O. It runs on the event loop, inside the transaction,
        and runs again on every retry, so it must be fast and deterministic:
        the loop runs nothing else while it runs. An LLM judge, or a person,
        belongs in afind_conflicts and aresolve, which hold no transaction
        while the decision is made. A policy must not call the store either:
        the store holds its connection while the policy runs, so such a call
        raises RuntimeError at once. A policy that raises rolls the
        transaction back, and its exception propagates unchanged, without a
        retry. So does the ValueError for an invalid decision.

        The index fields of ``value`` are embedded once, with
        aembed_documents, before the first attempt. If an attempt commits
        and its result is lost, as when the connection drops while the
        COMMIT's answer travels, the retry finds the rows that carry its
        write id and writes nothing. It returns the result rebuilt from
        those rows, with ``already_applied`` set, listing only what the rows
        prove: ``kept`` is empty, and on a store without history so are
        ``superseded`` and ``deleted``.

        Args:
            namespace: The new memory's namespace. It cannot be empty.
            key: The new memory's key, or None for a new uuid4.
            value: The new memory's value.
            policy: Decides, from the new memory as a Proposal and its
                candidates, what to do with each: a function that returns a
                Decision. langchain_cockroachdb.store has built-in ones.
            slot: A top-level field of ``value`` naming the fact it states,
                as in afind_conflicts.
            threshold: The lowest score of a similar candidate, as in
                afind_conflicts: 0.8 by default on a cosine store, required on
                an l2 or inner_product store, ignored without an index.
            limit: Most similar candidates (default 10), on top of the key
                and slot matches.
            filter: Narrows the slot and similar candidates by value, as in
                search.
            index: Which fields of ``value`` to embed, as in put.
            ttl: Minutes the new memory lives once written, or None for no
                expiry. Defaults to the store's default TTL.
            provenance: Where the new memory comes from, as a JSON object.
                It is stored in its own column and kept in history.
            valid_at: When the fact became true, as a timezone-aware datetime.
                By default the database's current time, raised to the latest
                candidate's updated_at if that is later. The memory then
                stores no valid time and is valid from its updated_at.

        Returns:
            What happened: the new memory's key, whether it was written,
            skipped or superseded, the version written, and the keys of the
            candidates superseded, deleted, confirmed and kept.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid
            ValueError: If another argument is invalid, as in afind_conflicts;
                if the policy is async or returns an invalid decision; or if
                setup() has not added the columns this method needs or, on a
                store with history, created the history table
        """
        plan = self._plan_proposal(
            namespace,
            key,
            value,
            slot=slot,
            threshold=threshold,
            limit=limit,
            filter=filter,
            index=index,
            ttl=ttl,
            provenance=provenance,
            valid_at=valid_at,
            method="areconcile",
        )
        _check_policy(policy)
        # The schema check comes first, so a store that cannot write costs no
        # embedding call. With history, the write also needs the history table
        if self._history is not None:
            await self._ahistory_schema()
        else:
            self._require_columns(await self._areconcile_columns(refresh=True))
        vector_rows = await self._embed_put(plan.put)
        # The keys that the decisions of earlier attempts named. A retry
        # looks for its write id under them only: an attempt that committed
        # wrote nowhere else, and one that reached no decision wrote nothing
        named: set[str] = set()

        async def attempt() -> ReconcileResult:
            cursors: list[AsyncCursor[DictRow]] = []
            try:
                async with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label="store reconcile",
                ) as cur:
                    conn = cur.connection

                    async def fetch(statements: Sequence[tuple[str, list]]) -> list[list[DictRow]]:
                        # In a pipeline the fetch flushes without a sync, so the
                        # transaction stays open
                        return [
                            await statement_cursor.fetchall()
                            for statement_cursor in await self._send(conn, statements, cursors)
                        ]

                    # A retry first looks for the rows its write id is on, in
                    # the same round trip as the first candidate read. If an
                    # earlier attempt committed, it returns that result
                    reads = self._key_slot_statements(plan)
                    checks = self._applied_statements(plan, named)
                    found = await fetch([*reads, *checks])
                    applied = self._applied_result(plan, found[len(reads) :]) if checks else None
                    if applied is not None:
                        return applied
                    read = self._key_slot_read(plan, found[: len(reads)])
                    similar = self._similar_read(plan, vector_rows, read)
                    for setting in similar.session:
                        await cur.execute(setting, prepare=False)
                    scored: Sequence[Sequence[dict[str, Any]]] = await fetch(similar.statements())
                    exact = similar.exact_statements(scored)
                    if exact:
                        scored = similar.fill_exact(scored, await fetch(exact))
                    conflicts = self._conflicts(plan, vector_rows, read, similar, scored)
                    proposal, candidates = conflicts.proposal, conflicts.candidates
                    decision = self._checked_decision(
                        proposal, candidates, self._decide(policy, proposal, candidates)
                    )
                    named.update(self._decision_keys(plan, decision))
                    statements = self._decision_statements(
                        plan,
                        candidates,
                        decision,
                        vector_rows=vector_rows,
                        valid_at=proposal.valid_at,
                    )
                    writes = await self._send(conn, statements, cursors)
                # Committed. Only the item upsert returns rows: the version written
                returned = [
                    await statement_cursor.fetchall()
                    for statement_cursor in writes
                    if statement_cursor.description is not None
                ]
                return self._reconcile_result(plan, candidates, decision, returned)
            finally:
                for statement_cursor in cursors:
                    await statement_cursor.close()

        return await self._retry(attempt, label="store reconcile")

    async def aresolve(self, conflicts: Conflicts, decision: Decision) -> ReconcileResult:
        """Apply a decision about what afind_conflicts found, in one transaction.

        The async form of CockroachDBStore.resolve, which describes the
        checks and the retries in full. This is the second step of the
        two-step flow: afind_conflicts reads the candidates, an LLM or a
        person decides while no transaction is open, and aresolve applies
        the decision as areconcile applies a policy's. It writes the new
        memory that afind_conflicts planned, with its provenance, valid time
        and write id, and the vectors it embedded then, so nothing is
        embedded again. For the same reason it refuses a proposal edited
        after afind_conflicts.

        In the same transaction, before it writes, aresolve checks that the
        memory under the new memory's key, and every candidate the decision
        supersedes, deletes or confirms, or names as superseding the new
        memory, still has the version and updated_at that afind_conflicts
        read. If one changed, it raises ConflictsChanged with those keys,
        writes nothing and does not retry: find the conflicts again and
        decide again. Candidates the decision keeps are not checked, and a
        confirmation in between never counts.

        If a resolve of the same Conflicts already committed, aresolve
        writes nothing and returns that write's result, rebuilt from its
        rows, with ``already_applied`` set. Once an attempt of the call may
        have committed (it sent its COMMIT and failed with an error other
        than a serialization failure or a deadlock), a memory the decision
        supersedes or deletes counts as unchanged when it is gone, for the
        rest of the call. A Conflicts and a Decision can wait in a LangGraph
        checkpoint while a person decides: aresolve takes them as the
        checkpoint gives them back.

        Args:
            conflicts: What afind_conflicts returned, on a store with the
                same table and index, sync or async.
            decision: What to do with the new memory and each candidate.

        Returns:
            What happened, as areconcile reports it.

        Raises:
            ConflictsChanged: If a memory the decision depends on changed
                since afind_conflicts. The attempt that found it wrote nothing.
            ValueError: Before any I/O: if ``conflicts`` holds no plan from
                afind_conflicts, as when it was built by hand; if a store
                with another table or index made it; if its proposal or its
                candidates changed since afind_conflicts; or if ``decision``
                is not a valid Decision for its proposal and candidates.
                After a schema read: if setup() has not added the columns
                this method needs or, on a store with history, created the
                history table.
        """
        resolving = self._plan_resolve(conflicts, decision, method="aresolve")
        # As in areconcile: with history, the write also needs the history table
        if self._history is not None:
            await self._ahistory_schema()
        else:
            self._require_columns(await self._areconcile_columns(refresh=True))
        # Whether an earlier attempt of this call may have committed: it sent
        # its COMMIT and failed with an error that is not a definite abort
        committed = False

        async def attempt() -> ReconcileResult:
            nonlocal committed
            committing = False
            cursors: list[AsyncCursor[DictRow]] = []
            try:
                async with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label="store resolve",
                ) as cur:
                    conn = cur.connection
                    # The check and the reads of an earlier commit go out as one
                    # round trip. In a pipeline the fetch flushes without a sync,
                    # so the transaction stays open
                    reads = await self._send(conn, resolving.reads, cursors)
                    found = [await statement_cursor.fetchall() for statement_cursor in reads]
                    applied = self._check_resolve(resolving, found, may_have_committed=committed)
                    if applied is not None:
                        return applied
                    writes = await self._send(conn, resolving.writes, cursors)
                    # The COMMIT goes out as the block ends
                    committing = True
                # Committed. Only the item upsert returns rows: the version written
                returned = [
                    await statement_cursor.fetchall()
                    for statement_cursor in writes
                    if statement_cursor.description is not None
                ]
                return self._reconcile_result(
                    resolving.plan, resolving.candidates, resolving.decision, returned
                )
            except Exception as error:
                committed = committed or (committing and may_have_committed(error))
                raise
            finally:
                for statement_cursor in cursors:
                    await statement_cursor.close()

        return await self._retry(attempt, label="store resolve")

    async def aput_if(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *,
        expected: MemoryRecord | None,
        index: Literal[False] | list[str] | None = None,
        ttl: float | None | NotProvided = NOT_PROVIDED,
        provenance: dict[str, Any] | None = None,
        valid_at: datetime | None = None,
    ) -> ConditionalWrite:
        """Write a memory only if it is still the one the caller read, in one transaction.

        The async form of CockroachDBStore.put_if. Two tasks that read the
        same memory with arecord() and both write it back with aput would
        lose one update without a word. aput_if writes only if the live
        memory under ``key`` still has the version and updated_at of
        ``expected``: a put moves the version, and a 0.5.x store, which does
        not, still moves updated_at. With ``expected=None`` it writes only if
        no live memory has the key, so a memory is created once. An expired
        memory counts as absent.

        On a match it writes as aput does, in the transaction that checked,
        keeping the version it replaces in history on a store with history.
        On a mismatch it writes nothing and returns ``applied=False`` with
        ``current``, the live memory as the same transaction read it, or
        None. A conflict is an expected outcome here, so it is returned, not
        raised. If an attempt commits and its result is lost, the retry finds
        the write id on the row and returns ``applied=True`` with
        ``already_applied``. If another writer replaced the memory before the
        retry, the retry reports a conflict with that writer's memory.

        Args:
            namespace: The memory's namespace. It cannot be empty.
            key: The memory's key.
            value: The new value.
            expected: The MemoryRecord that arecord() returned for this
                namespace and key, or None to write only if no live memory
                has the key.
            index: Which fields of ``value`` to embed, as in put. False
                embeds nothing.
            ttl: Minutes the memory lives once written, or None for no
                expiry. Defaults to the store's default TTL.
            provenance: Where the memory comes from, as a JSON object. It is
                stored in its own column and kept in history.
            valid_at: When the fact became true, as a timezone-aware
                datetime. Without one the memory stores no valid time and is
                valid from its updated_at.

        Returns:
            Whether the write applied, the version written, and on a
            conflict the current memory.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid
            ValueError: Before any I/O: if ``value`` is not a dict that
                serializes as JSON; if ``expected`` is not None or the record
                of this namespace and key; or if ``ttl``, ``provenance`` or
                ``valid_at`` is invalid. After a schema read: if setup() has
                not added the columns this method needs or, on a store with
                history, created the history table.
        """
        plan = self._plan_put_if(
            namespace,
            key,
            value,
            expected=expected,
            index=index,
            ttl=ttl,
            provenance=provenance,
            valid_at=valid_at,
            method="aput_if",
        )
        # The schema check comes first, so a store that cannot write costs no
        # embedding call. With history, the write also needs the history table
        if self._history is not None:
            await self._ahistory_schema()
        else:
            self._require_columns(await self._areconcile_columns(refresh=True))
        vector_rows = await self._embed_put(plan.put)
        return await self._arun_conditional(
            plan, self._conditional_writes(plan, vector_rows), label="store put_if"
        )

    async def adelete_if(
        self, namespace: tuple[str, ...], key: str, *, expected: MemoryRecord | None
    ) -> ConditionalWrite:
        """Delete a memory only if it is still the one the caller read, in one transaction.

        The async form of CockroachDBStore.delete_if, which describes the
        retries in full. adelete_if deletes only if the live memory under
        ``key`` still has the version and updated_at of ``expected``, as
        aput_if checks them. On a match it deletes as adelete does: on a
        store with history the memory is kept in history as deleted, and its
        vectors go. With ``expected=None`` it applies only while no live
        memory has the key, so it deletes nothing live.

        ``applied=True`` means the memory matched ``expected`` and this call
        deleted it, or, after an earlier attempt of the call that may have
        committed, that the memory is gone and was this call's to delete. On
        a mismatch it deletes nothing and returns ``applied=False`` with
        ``current``, the live memory as the same transaction read it, or
        None if there is none. A conflict is returned, not raised.

        Args:
            namespace: The memory's namespace. It cannot be empty.
            key: The memory's key.
            expected: The MemoryRecord that arecord() returned for this
                namespace and key, or None.

        Returns:
            Whether the delete applied, and on a conflict the current memory.

        Raises:
            InvalidNamespaceError: If the namespace is empty or a label is invalid
            ValueError: Before any I/O: if ``expected`` is not None or the
                record of this namespace and key. After a schema read: if
                setup() has not added the columns this method needs or, on a
                store with history, created the history table.
        """
        plan = self._plan_delete_if(namespace, key, expected=expected, method="adelete_if")
        # As in aput_if: with history, the delete also needs the history table
        if self._history is not None:
            await self._ahistory_schema()
        else:
            self._require_columns(await self._areconcile_columns(refresh=True))
        return await self._arun_conditional(
            plan, self._conditional_writes(plan), label="store delete_if"
        )

    async def _arun_conditional(
        self, plan: ConditionalPlan, writes: Sequence[tuple[str, list]], *, label: str
    ) -> ConditionalWrite:
        """Run the attempts of aput_if or adelete_if: read and check, then write and commit.

        Each attempt sends the reads of ``plan`` as one round trip and checks
        their rows while the transaction stays open. A conflict, or a write
        an earlier attempt already committed, returns at once and writes
        nothing. Otherwise ``writes`` go out with the COMMIT.
        """
        # Whether an earlier attempt of this call may have committed: it sent
        # its COMMIT and failed with an error that is not a definite abort.
        # Until one may have, every attempt checks as strictly as the first
        committed = False

        async def attempt() -> ConditionalWrite:
            nonlocal committed
            committing = False
            cursors: list[AsyncCursor[DictRow]] = []
            try:
                async with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label=label,
                ) as cur:
                    conn = cur.connection
                    # In a pipeline the fetch flushes without a sync, so the
                    # transaction stays open
                    reads = await self._send(
                        conn, plan.reads(may_have_committed=committed), cursors
                    )
                    found = [await statement_cursor.fetchall() for statement_cursor in reads]
                    decided = self._check_conditional(plan, found, may_have_committed=committed)
                    if decided is not None:
                        return decided
                    sent = await self._send(conn, writes, cursors)
                    # The COMMIT goes out as the block ends
                    committing = True
                # Committed. Only aput_if's item upsert returns rows: the version written
                returned = [
                    await statement_cursor.fetchall()
                    for statement_cursor in sent
                    if statement_cursor.description is not None
                ]
                return self._conditional_result(plan, returned)
            except Exception as error:
                committed = committed or (committing and may_have_committed(error))
                raise
            finally:
                for statement_cursor in cursors:
                    await statement_cursor.close()

        return await self._retry(attempt, label=label)

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
