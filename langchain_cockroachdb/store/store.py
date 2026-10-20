"""Synchronous CockroachDB store for LangGraph long-term memory.

Provides the interface of langgraph-checkpoint-postgres's PostgresStore on
CockroachDB. Every operation retries transient errors under the store's
RetryPolicy, and every write is an UPSERT or a DELETE, so a retry after an
ambiguous result is safe.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from langgraph.store.base import (
    BaseStore,
    InvalidNamespaceError,
    Op,
    Result,
    TTLConfig,
)
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb._connection import connection_options, to_libpq_url
from langchain_cockroachdb._psycopg_ops import PsycopgOps
from langchain_cockroachdb.checkpointer import _internal
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

logger = logging.getLogger(__name__)

# Minutes between sweeps when neither start_ttl_sweeper nor the TTL config sets
# an interval, as in PostgresStore
DEFAULT_SWEEP_INTERVAL_MINUTES = 5


class CockroachDBStore(BaseStore, BaseCockroachDBStore, PsycopgOps):
    """LangGraph store that keeps long-term memory in CockroachDB.

    Items live in one table keyed by namespace and key. Namespaces are stored
    as their labels joined with ".". For production, pass a
    ``psycopg_pool.ConnectionPool`` or use ``from_conn_string`` with
    ``pool_config``: a pool replaces broken connections and runs batches
    concurrently, while a single connection runs one batch at a time.

    A batch runs its gets, searches and listings first, each group as one
    round of reads, then all its puts and deletes in one transaction.

    Example:
        ```python
        from langchain_cockroachdb import CockroachDBStore

        with CockroachDBStore.from_conn_string(
            "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
        ) as store:
            store.setup()
            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")
        ```
    """

    supports_ttl: bool = True
    _log = logger

    # Runs operations one at a time on a single connection; a pool needs no lock
    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        *,
        pipe: Pipeline | None = None,
        index: CockroachDBIndexConfig | None = None,
        ttl: TTLConfig | CockroachDBTTLConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        table_name: str = "store",
        schema: str = "public",
    ) -> None:
        """Create a store on a connection or a pool.

        Args:
            conn: A psycopg connection, or a ConnectionPool (recommended). It
                must be in autocommit mode and use dict_row.
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
            ValueError: If a pipeline is given with a pool, or a name or the
                TTL or index settings are invalid
        """
        # Validated first, so a bad index fails before anything starts
        resolved_index = resolve_index_config(index)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )
        self._configure(index=resolved_index, ttl=ttl, table_name=table_name, schema=schema)
        # BaseStore reads ttl_config for the default TTL and refresh_on_read
        self.ttl_config = ttl
        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()
        self.retry_policy = retry_policy or RetryPolicy()
        # Set while a write batch of ours is between BEGIN and COMMIT on a
        # single connection. See CockroachDBSaver for the details.
        self._batch_open = False
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    @classmethod
    @contextmanager
    def from_conn_string(
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
    ) -> Iterator[CockroachDBStore]:
        """Create a store on a new connection, or on a new pool with ``pool_config``.

        Args:
            conn_string: CockroachDB connection string (cockroachdb://, postgresql:// and similar)
            pipeline: Whether to use pipeline mode on the single connection.
                Ignored with pool_config.
            pool_config: Open a ConnectionPool with these settings instead of one connection
            index: Semantic search settings
            ttl: TTL settings
            retry_policy: How operations retry (default: RetryPolicy())
            connect_timeout: Seconds to wait when connecting, or None (default: 10)
            prepare_threshold: psycopg prepared statement threshold, None to disable (default: 5)
            connect_kwargs: Extra psycopg connection options, e.g. application_name
            table_name: Name of the items table (default "store")
            schema: Schema of the store's tables (default "public")

        Yields:
            CockroachDBStore instance. The connection or pool closes on exit.

        With ``pool_config``, ``pipeline`` is ignored, as in PostgresStore: a
        pipeline needs a single connection.
        """
        policy = retry_policy or RetryPolicy()
        url = to_libpq_url(conn_string)
        options = dict(connect_kwargs or {})

        def make(conn: _internal.Conn, pipe: Pipeline | None = None) -> CockroachDBStore:
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
            pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
                url,
                min_size=pool_config.get("min_size", 1),
                max_size=pool_config.get("max_size"),
                kwargs=kwargs,
                open=False,
            )
            with pool:
                yield make(pool)
            return

        kwargs = connection_options(
            url, options, connect_timeout, prepare_threshold, component="store"
        )
        conn: Connection[DictRow] = policy.run(
            lambda: Connection.connect(url, **kwargs), label="connect store"
        )
        with conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield make(conn, pipe)
            else:
                yield make(conn)

    def setup(self) -> None:
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

        def init_and_read_version() -> int:
            with self._cursor(
                timeout=self.retry_policy.ddl_attempt_timeout, label="store setup"
            ) as cur:
                cur.execute(migrations[0])
                self._sync_pipeline()
                results = cur.execute(version_sql)
                self._sync_pipeline()
                row = results.fetchone()
                return -1 if row is None else row["v"]

        version = self._retry(init_and_read_version, label="store setup")

        for v in range(version + 1, len(migrations)):

            def apply_migration(migration_v: int = v) -> None:
                with self._cursor(
                    timeout=self.retry_policy.ddl_attempt_timeout,
                    label=f"store migration {migration_v}",
                ) as cur:
                    cur.execute(migrations[migration_v])
                    self._sync_pipeline()
                    cur.execute(record_sql, (migration_v,))
                    self._sync_pipeline()

            self._retry(apply_migration, label=f"store migration {v}")

        for statement in self.sql.vector_statements():
            self._run_ddl(statement, "store vector setup")
        if self.index is not None:
            self._check_index()
        if self._ttl_given:
            policy = schema_change_policy(self.retry_policy)
            for statement in self.sql.ttl_statements(self.ttl.cron):
                self._run_ddl(statement, "store TTL setup", policy=policy)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
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
            self._embed_queries(plan)
        if plan.search_lookups:
            plan.bind_namespaces(
                self._read_all(plan.lookup_statements(), "store search namespaces")
            )
        for label, statements, fill, session in plan.reads():
            fill(self._read_all(statements, label, session))
        if plan.puts is not None:
            self._run_puts(plan.puts)
        return plan.results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Run ``batch`` in a worker thread, as PostgresStore does."""
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, list(ops))

    def _read_all(
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

        def read() -> list[list[DictRow]]:
            cursors: list[Cursor[DictRow]] = []
            try:
                with self._cursor(
                    pipeline=True,
                    transaction=bool(session),
                    timeout=self.retry_policy.attempt_timeout,
                    label=label,
                ) as cur:
                    for setting in session or ():
                        cur.execute(setting, prepare=False)
                    conn = cur.connection
                    for statement in statements:
                        cursors.append(conn.cursor(binary=True, row_factory=dict_row))
                        cursors[-1].execute(*statement)
                return [statement_cursor.fetchall() for statement_cursor in cursors]
            finally:
                for statement_cursor in cursors:
                    statement_cursor.close()

        return self._retry(read, label=label)

    def _embed_queries(self, plan: BatchPlan) -> None:
        """Embed each distinct query text of the batch once and bind the vectors.

        Runs before the first attempt of any read, so a retry reuses the
        vectors. embed_query, not embed_documents: models that embed
        queries and documents differently get the query form.
        """
        assert self.index is not None
        texts = plan.query_texts()
        vectors = [self.index.embeddings.embed_query(text) for text in texts]
        self._bind_query_vectors(plan, texts, vectors)

    def drop_vector_index(self) -> None:
        """Drop the store's vector index, if it exists.

        Run setup() afterwards to build it again with the store's current
        configuration, for example to change the distance. Searches scan
        the vectors until the new index is built.
        """
        self._run_ddl(self.sql.drop_vector_index(), "store drop vector index")

    def _check_index(self) -> None:
        """Read the vectors table's definition and check it against the configuration."""

        def read() -> str:
            with self._cursor(
                timeout=self.retry_policy.attempt_timeout, label="store index check"
            ) as cur:
                cur.execute(self.sql.show_create_vectors())
                self._sync_pipeline()
                row = cur.fetchone()
                return "" if row is None else row["create_statement"]

        self._check_vector_index(self._retry(read, label="store index check"))

    def _run_puts(self, plan: PutPlan) -> None:
        # Embed once, before the first attempt: a retry reuses the vectors
        embeddings: list[list[float]] = []
        if self.index is not None and plan.embed_requests:
            embeddings = self.index.embeddings.embed_documents(
                [text for _, _, _, text in plan.embed_requests]
            )
        statements = self._write_statements(plan, self._vector_rows(plan, embeddings))
        if not statements:
            return

        def write() -> None:
            with self._cursor(
                pipeline=True,
                transaction=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store put",
            ) as cur:
                for statement in statements:
                    cur.execute(*statement)

        self._retry(write, label="store put")

    def delete_namespace(self, namespace_prefix: tuple[str, ...]) -> int:
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
                "delete_namespace needs a non-empty namespace prefix. To clear the whole "
                "store on purpose, delete each top-level namespace from list_namespaces("
                "max_depth=1), or drop the store's tables."
            )
        statement = self.sql.delete_namespace(prefix)

        def write() -> int:
            # One statement, so items and vectors commit together. pipeline=True
            # makes a component-wide pipeline sync, which commits it
            with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store delete namespace",
            ) as cur:
                cur.execute(*statement)
                return len(cur.fetchall())

        return self._retry(write, label="store delete namespace")

    def sweep_ttl(self) -> int:
        """Delete expired items now, without waiting for CockroachDB's TTL job.

        Returns:
            The number of items deleted.
        """

        def sweep() -> int:
            count = 0
            with self._cursor(
                timeout=self.retry_policy.attempt_timeout, label="store TTL sweep"
            ) as cur:
                for statement in self.sql.sweep():
                    cur.execute(*statement)
                    self._sync_pipeline()
                    # The items statement comes last
                    count = cur.rowcount
            return max(count, 0)

        return self._retry(sweep, label="store TTL sweep")

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Sweep expired items periodically in a background thread.

        CockroachDB's row-level TTL job, turned on by setup() with a TTL
        configuration, already deletes expired rows. The sweeper deletes them
        sooner. Expired items are never returned either way.

        Args:
            sweep_interval_minutes: Minutes between sweeps. Defaults to the
                TTL configuration's sweep_interval_minutes, else 5.

        Returns:
            A future that completes when the sweeper stops. Cancelling it
            stops the sweeper.
        """
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes
            or self.ttl_config.get("sweep_interval_minutes")
            or DEFAULT_SWEEP_INTERVAL_MINUTES
        )
        logger.info("Starting store TTL sweeper with interval %s minutes", interval)
        future = concurrent.futures.Future()

        def sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        expired = self.sweep_ttl()
                        if expired > 0:
                            logger.info("Store swept %d expired items", expired)
                    except Exception:
                        logger.exception("Store TTL sweep iteration failed")
                future.set_result(None)
            except Exception as error:
                future.set_exception(error)

        thread = threading.Thread(target=sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()
        future.add_done_callback(lambda f: self._ttl_stop_event.set() if f.cancelled() else None)
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread if it is running.

        Args:
            timeout: Seconds to wait for the thread to stop, or None to wait.

        Returns:
            True if the thread stopped or was not running, False on timeout.
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        stopped = not self._ttl_sweeper_thread.is_alive()
        if stopped:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")
        return stopped

    def disable_ttl(self) -> None:
        """Turn off CockroachDB row-level TTL on the store's tables.

        Expired items are still left out of reads while omit_expired is on.
        """
        for statement in self.sql.disable_ttl_statements():
            self._run_ddl(statement, "store disable TTL")

    def __del__(self) -> None:
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)
