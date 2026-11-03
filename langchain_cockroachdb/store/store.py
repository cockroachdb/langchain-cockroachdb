"""Synchronous CockroachDB store for LangGraph long-term memory.

Provides the interface of langgraph-checkpoint-postgres's PostgresStore on
CockroachDB. Every operation retries transient errors under the store's
RetryPolicy. Every write is an UPSERT, a DELETE, an UPDATE that pushes back
an expiry or confirms a memory once per confirmation id, or an INSERT into
history that skips versions and write ids already there, so a retry after
an ambiguous result is safe.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal

import psycopg
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    NotProvided,
    Op,
    Result,
    TTLConfig,
)
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import _psycopg_conn
from langchain_cockroachdb._connection import connection_options, to_libpq_url
from langchain_cockroachdb._psycopg_ops import PsycopgOps
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
    _MigrationWait,
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
from langchain_cockroachdb.store.sql import MIGRATION_RECONCILE, TTLChange, VectorRow

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

    A batch runs its reads first, then all its puts and deletes in one
    transaction. Its gets, its searches and its listings take one round trip
    each. On a store with a vector index, semantic searches under a namespace
    add one before the searches, to find their namespaces, and one after them
    when a search must run again exactly. A flat store adds neither. The
    long-term memory guide explains why.

    With ``history``, a batch that overwrites or deletes a memory first
    copies the version it ends into a history table, in its transaction.
    ``delete`` then keeps the memory in history; ``purge`` removes a memory
    with its history, and ``history`` lists the old versions.

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
        conn: _psycopg_conn.Conn,
        *,
        pipe: Pipeline | None = None,
        index: CockroachDBIndexConfig | None = None,
        ttl: TTLConfig | CockroachDBTTLConfig | None = None,
        history: bool | HistoryConfig | None = None,
        reconcile_scan_limit: int = DEFAULT_RECONCILE_SCAN_LIMIT,
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
            ValueError: If a pipeline is given with a pool, or a name or the
                TTL, index, history or reconcile_scan_limit settings are invalid
        """
        # Validated first, so a bad index fails before anything starts
        resolved_index = resolve_index_config(index)
        resolved_history = resolve_history_config(history)
        scan_limit = resolve_reconcile_scan_limit(reconcile_scan_limit)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )
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
        history: bool | HistoryConfig | None = None,
        reconcile_scan_limit: int = DEFAULT_RECONCILE_SCAN_LIMIT,
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
            CockroachDBStore instance. The connection or pool closes on exit.

        With ``pool_config``, ``pipeline`` is ignored, as in PostgresStore: a
        pipeline needs a single connection.

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

        def make(conn: _psycopg_conn.Conn, pipe: Pipeline | None = None) -> CockroachDBStore:
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
        are recorded with ON CONFLICT DO NOTHING. A setup() that finds another
        process still applying migration 2 waits for it, for up to
        SCHEMA_CHANGE_SETTLE_SECONDS, before it records the migration, so no
        store writes the new columns before they exist. When no job applies
        the migration, as after a peer's migration failed, it applies the
        migration itself. With an index, it creates
        the vectors table and its C-SPANN index (unless ``flat``) on every
        call, then checks an existing index against the configuration. With
        history, it creates the history table, and with ``ttl_days`` turns on
        CockroachDB row-level TTL there. With a TTL configuration, it then
        turns on CockroachDB row-level TTL on ``expires_at``.

        It reads a table's TTL settings first and changes them only when
        they differ from the configuration. On CockroachDB v25.4 a TTL
        change fails while another schema change runs on the same table,
        such as a vector index that another setup() builds. That error is
        retried with the budget that schema_change_policy gives it, up to
        ten minutes by default for each table, even if the policy's
        ``retry_on`` would refuse it; ``max_attempts=1`` turns this off.

        An application role that did not run setup() needs SELECT, INSERT,
        UPDATE and DELETE on the store's items and vectors tables, and on
        the history table when history is on. With history off, it still
        needs SELECT and DELETE on a history table that exists, left from
        when history was on, for delete_namespace. Without SELECT on the
        migrations table, or without a migrations table, as when the tables
        were created without setup(), the store finds migration 2 through the
        items table's columns instead. On a multi-node cluster, one write can
        then fail in the seconds after setup() applies migration 2. Running
        setup() before rolling out 0.6.0, or granting SELECT on the migrations
        table, avoids that. The first store call inside a transaction the
        caller opened gets the permission error once. Only the SELECT grant
        avoids it.

        Raises:
            ValueError: If the vectors table exists with other dims, or its
                index uses another distance. A different partition size, or
                an index while ``flat`` is set, logs a WARNING instead.
            TimeoutError: If another process is still applying migration 2
                after SCHEMA_CHANGE_SETTLE_SECONDS. Run setup() again.
        """
        migrations = self.sql.migrations()
        version_sql = self.sql.migration_version()
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
            label = f"store migration {v}"
            self._run_ddl(migrations[v], label)
            if v == MIGRATION_RECONCILE:
                # Another setup() may be applying this migration. The statement
                # then returns at once, before the columns exist. A future ALTER
                # migration needs the same wait, and the same takeover
                self._wait_for_migration(migrations[v])

            def record_migration(migration_v: int = v, label: str = label) -> None:
                with self._cursor(
                    timeout=self.retry_policy.ddl_attempt_timeout, label=label
                ) as cur:
                    cur.execute(record_sql, (migration_v,))
                    self._sync_pipeline()

            self._retry(record_migration, label=label)

        # A write before setup() may have cached the schema without
        # migration 2. The next write reads it again.
        self._columns_ready = None

        for statement in self.sql.vector_statements():
            self._run_ddl(statement, "store vector setup")
        if self.index is not None:
            self._check_index()
        # A TTL change waits out a schema change that another process runs
        # on the same table, such as its migration 2
        policy = schema_change_policy(self.retry_policy)
        if self._history is not None:
            self._run_ddl(self.sql.create_history_table(), "store history setup", policy=policy)
            if self._history.ttl_days is not None:
                self._set_ttl(self.sql.history_ttl_change(), "store history setup", policy)
            # A write before setup() may have found no history table. The
            # next write reads the schema again.
            self._history_ready = None
        if self._ttl_given:
            for change in self.sql.ttl_changes(self.ttl.cron):
                self._set_ttl(change, "store TTL setup", policy)

    def _wait_for_migration(self, migration: str) -> None:
        """Wait until migration 2 is in place, when another setup() is still applying it.

        CockroachDB runs the migration as a schema change job that backfills
        the items table. The same statement sent while that job runs returns
        at once, before the columns exist, and recording migration 2 then
        would let other stores write columns that are not there. So this
        returns once the columns are public and no schema change job on the
        items table is left, the job's last step being to wait until every
        node uses the columns. For the setup() whose statement ran the job,
        that is one read of the columns and one of SHOW JOBS. Otherwise it
        checks again, with backoff up to SCHEMA_CHANGE_BUSY_MAX_BACKOFF
        seconds, and logs its progress (see _settle_pause). When the columns
        are not public and no job applies the migration, it sends
        ``migration`` again (see _migration_left).

        A job is found by its description. SHOW JOBS cuts a long description
        short, which can cut a long qualified table name, and a role without
        admin sees only its own jobs. Then no job is found, and the column
        check alone guards. An error from SHOW JOBS that the retry policy
        would not retry also leaves the column check alone, logged once.

        This waits for another process and retries no failed statement, so
        the retry policy's max_attempts does not shorten it. Each check
        retries under the policy.

        Raises:
            TimeoutError: If the migration is not in place after
                SCHEMA_CHANGE_SETTLE_SECONDS, saying what remained
        """
        wait = _MigrationWait()
        check = 0
        while (left := self._migration_left(migration, wait)) is not None:
            if wait.started is None:
                self._start_migration_wait(wait, left)
            self._sleep(self._settle_pause(wait, check, left))
            check += 1
        if wait.started is not None:
            self._end_migration_wait(wait)

    def _migration_left(self, migration: str, wait: _MigrationWait) -> str | None:
        """What still keeps migration 2 from being in place, or None once nothing does.

        It reads the columns, then SHOW JOBS: on every check once the
        columns are public, and otherwise as _jobs_due allows, keeping the
        last count in between. When a check that read SHOW JOBS finds the
        columns not public and no job, it sends ``migration`` again, then
        reads both again, so a takeover that applied the migration ends the
        wait.
        """
        columns = [self.sql.reconcile_columns_exist()]
        public = self._columns_public(self._read_all(columns, "store migration wait"))
        if not public and not self._jobs_due(wait):
            return self._migration_condition(public, wait.jobs)
        running = self._note_jobs(wait, self._jobs_on_items())
        if not public and not running:
            # No job that SHOW JOBS shows applies the migration: a peer's
            # migration failed or was reverted, or the job is another role's or
            # its long qualified name was cut from its description. The
            # statement is idempotent: it takes the migration over, or returns
            # at once while another job runs it
            self._run_ddl(migration, "store migration 2")
            public = self._columns_public(self._read_all(columns, "store migration wait"))
            running = self._note_jobs(wait, self._jobs_on_items())
            if public:
                self._log_takeover()
        return self._migration_condition(public, running)

    def _jobs_on_items(self) -> int | None:
        """The schema change jobs that SHOW JOBS shows on the items table, or None without it.

        A job is found by the start of its description. A long qualified
        name may be cut from it, and another role's job is not shown: the
        column check then guards alone. So does an error the retry policy
        would not retry, after _skip_job_check logs it.
        """
        if self._jobs_unreadable:
            return None
        try:
            found = self._read_all([self.sql.schema_changes_running()], "store migration wait")
        except Exception as error:
            if not self._skip_job_check(error):
                raise
            return None
        return self._running_jobs(found)

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Sleep between the checks of _wait_for_migration."""
        time.sleep(seconds)

    def _set_ttl(self, change: TTLChange, label: str, policy: RetryPolicy) -> None:
        """Turn on a table's row-level TTL, unless it already has those settings.

        Every attempt reads the table's settings before it changes them. So
        a setup() that finds them in place changes nothing, and neither
        does a retry after another process set them. On v25.4 a change waits
        for any other schema change on the table.
        """

        def apply() -> None:
            with self._cursor(timeout=policy.ddl_attempt_timeout, label=label) as cur:
                cur.execute(*change.applied)
                self._sync_pipeline()
                row = cur.fetchone()
                if row is not None and row["applied"]:
                    return
                cur.execute(change.statement)
                self._sync_pipeline()

        self._retry(apply, label=label, policy=policy)

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
        exact = plan.exact_statements()
        if exact:
            plan.fill_exact(self._read_all(exact, "store search exact"))
        if plan.puts is not None:
            self._run_puts(plan.puts)
        return plan.results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Run ``batch`` in a worker thread, as PostgresStore does."""
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, list(ops))

    @contextmanager
    def _cursor(
        self,
        *,
        pipeline: bool = False,
        transaction: bool = False,
        timeout: float | None = None,
        label: str | None = None,
    ) -> Iterator[Cursor[DictRow]]:
        """PsycopgOps._cursor, refused while this thread runs a reconcile policy of the store.

        Every connection checkout of the store comes through here. Inside a
        policy the reconcile holds the connection: see _refuse_inside_user_callable.
        """
        self._refuse_inside_user_callable()
        with super()._cursor(
            pipeline=pipeline, transaction=transaction, timeout=timeout, label=label
        ) as cur:
            yield cur

    @staticmethod
    def _send(
        conn: Connection[DictRow],
        statements: Sequence[tuple[str, list]],
        cursors: list[Cursor[DictRow]],
    ) -> list[Cursor[DictRow]]:
        """Execute each statement on a cursor of its own, without fetching, and return the cursors.

        They are added to ``cursors`` as well, for the caller to close. In a
        pipeline, statements sent together take one round trip, and fetching
        their rows flushes without a sync, so an open transaction stays open.
        """
        sent = []
        for statement in statements:
            sent.append(conn.cursor(binary=True, row_factory=dict_row))
            cursors.append(sent[-1])
            sent[-1].execute(*statement)
        return sent

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
        queries and documents differently get the query form. Inside a
        reconcile policy it refuses before calling the model.
        """
        assert self.index is not None
        self._refuse_inside_user_callable()
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

    def _schema(self, *, refresh: bool = False) -> SchemaCheck:
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
            schema = self._cache_schema(self._read_schema())
        return schema

    def _read_schema(self) -> list[list[DictRow]]:
        """Send the schema check, with the column read if the migrations read fails.

        The migrations read falls back when the store's role has no grant on
        the migrations table (42501), or there is no migrations table
        (42P01), as when the tables were created without setup(). Only the
        schema check's own round trip is caught: either error from any other
        statement still raises.
        """
        read_migrations = not self._migrations_unreadable
        try:
            return self._read_all(
                self._schema_statements(read_migrations=read_migrations), "store schema check"
            )
        except (psycopg.errors.InsufficientPrivilege, psycopg.errors.UndefinedTable) as error:
            if not self._fall_back_to_columns(error, read_migrations=read_migrations):
                raise
        return self._read_all(self._schema_statements(read_migrations=False), "store schema check")

    def _reconcile_columns(self, *, refresh: bool = False) -> bool:
        """Whether migration 2's columns exist, as the schema check last read it.

        It reads the schema the first time, and again when it last found the
        columns missing and that answer is older than _SCHEMA_RECHECK_SECONDS,
        or ``refresh`` is set. Methods that need the columns pass ``refresh``,
        so they see a setup() that another process ran since. Until setup()
        applies migration 2, puts write with the 0.5 statements and those
        methods raise ValueError.
        """
        return self._schema(refresh=refresh).columns

    def _history_schema(self) -> SchemaCheck:
        """The schema check for an operation that needs the history table.

        It runs with ``refresh``: what it found present it trusts for good,
        and what it found missing it reads again, once, so it sees a setup()
        that another process ran since before it refuses.

        Raises:
            ValueError: If the store keeps no history, or setup() has not
                created the history table yet.
        """
        self._require_history()
        schema = self._schema(refresh=True)
        self._require_history_table(schema)
        return schema

    def _history_table_exists(self) -> bool:
        """Whether the history table exists, for delete_namespace on a store without history.

        True is trusted for good. Until then every call reads. Only
        delete_namespace asks, and it is a rare call: plain writes never do.
        """
        if self._history_ready is True:
            return True
        [rows] = self._read_all([self.sql.history_exists()], "store history check")
        return self._cache_history(rows)

    def _embed_put(self, plan: PutPlan) -> list[VectorRow]:
        """Embed the texts a put plan writes, once, and pair the vectors with their rows.

        Puts, find_conflicts and reconcile call it before their first
        attempt, so a retry reuses the vectors. All embed documents, with
        embed_documents: the vectors are the ones the write stores. Inside a
        reconcile policy it refuses before calling the model.
        """
        self._refuse_inside_user_callable()
        embeddings: list[list[float]] = []
        if self.index is not None and plan.embed_requests:
            embeddings = self.index.embeddings.embed_documents(
                [text for _, _, _, text in plan.embed_requests]
            )
        return self._vector_rows(plan, embeddings)

    def _run_puts(self, plan: PutPlan) -> None:
        # The schema check comes first. With history, a write refuses to run
        # until setup() has created the history table, since it could not
        # keep what it replaces, and a refused batch costs no embedding call
        schema = self._history_schema() if self._history is not None else self._schema()
        vector_rows = self._embed_put(plan)
        # The write id too: a retry after a commit whose result was lost
        # finds it on the rows and changes nothing
        if schema.columns:
            plan.write_id = str(uuid.uuid4())
        statements = self._write_statements(plan, vector_rows)
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
        not ``("user", "ab")``. One statement deletes the items, their
        vectors and their history. A store without history deletes history
        too when the history table exists, as when the store kept history
        before, so nothing of the namespace is left. Its role then needs
        SELECT and DELETE on that table: without them the whole call fails
        with a permission error and deletes nothing, rather than forget only
        part of the namespace.

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
        prefix = _namespace_to_delete(namespace_prefix, "delete_namespace", "list_namespaces")
        if self._history is not None:
            self._history_schema()
            history = True
        else:
            history = self._history_table_exists()
        statement = self.sql.delete_namespace(prefix, history=history)

        def write() -> int:
            # One statement, so items, vectors and history commit together. pipeline=True
            # makes a component-wide pipeline sync, which commits it
            with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store delete namespace",
            ) as cur:
                cur.execute(*statement)
                return len(cur.fetchall())

        return self._retry(write, label="store delete namespace")

    def history(
        self, namespace: tuple[str, ...], key: str, *, limit: int = 20
    ) -> list[MemoryVersion]:
        """The old versions of a memory, newest first.

        A store with history keeps every version that a write ends: one a
        put overwrites (reason "overwritten"), one a delete ends
        ("deleted"), and one that reconcile or resolve supersedes
        ("superseded", with ``superseded_by`` naming the memory that
        replaced it). A new memory that reconcile or resolve records as
        already superseded by a newer fact is kept there too, as
        "superseded". The live value is not one of them; get() returns it.

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
        prefix = _memory_prefix(namespace, "history")
        _check_limit(limit)
        self._history_schema()
        statement = self.sql.history_versions(prefix, str(key), limit=limit)
        [rows] = self._read_all([statement], "store history")
        return [_row_to_version(tuple(namespace), str(key), row) for row in rows]

    def purge(self, namespace: tuple[str, ...], key: str) -> bool:
        """Forget a memory: delete the item, its vectors and its history.

        On a store with history, delete() keeps the version it ends, so
        purge is how to remove a memory for good. One statement deletes all
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
                history=True and run setup(); without history, delete()
                already removes a memory and its vectors), or setup() has
                not created the history table
            InvalidNamespaceError: If the namespace is empty or a label is invalid
        """
        prefix = _memory_prefix(namespace, "purge")
        self._history_schema()
        statement = self.sql.purge(prefix, str(key))

        def write() -> bool:
            # One statement, so the item, its vectors and its history commit
            # together. pipeline=True makes a component-wide pipeline sync
            with self._cursor(
                pipeline=True,
                timeout=self.retry_policy.attempt_timeout,
                label="store purge",
            ) as cur:
                cur.execute(*statement)
                row = cur.fetchone()
                return row is not None and bool(row["removed"])

        return self._retry(write, label="store purge")

    def find_conflicts(
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

        Candidates are live memories in ``namespace`` itself, not in its
        descendants, for one of three reasons, strongest first:

        - ``"key"``: the memory under ``key``, whatever ``filter`` says.
        - ``"slot"``: with ``slot``, the memories whose value holds the same
          value in that field, compared as JSON.
        - ``"similar"``: on a store with an index, the memories whose vectors
          score at or above ``threshold`` against the new memory's, with the
          store's score function: 1 minus the distance on a cosine store,
          the negated distance on an l2 or inner_product store. The new
          memory is embedded as a document (``embed_documents``), while
          search embeds its query (``embed_query``), so with a model that
          embeds the two differently the scores differ from search's.

        Key and slot matches carry their best score when they have vectors,
        and similar ones always do. Candidates come strongest reason first,
        then best score, then key. Expired memories are never candidates,
        and no read pushes back an expiry.

        The index fields of ``value`` are embedded once, and the embedding is
        kept for the write. Then two round trips read the candidates: first
        the database's clock, the key and slot matches and the namespace's
        vector count, then the similar memories and the scores. A namespace
        with at most ``reconcile_scan_limit`` vectors has all of them ranked.
        A larger one takes its similar candidates from a window of the
        C-SPANN index. When that window holds nothing below ``threshold``
        and gave fewer than ``limit`` candidates, matches may lie beyond it,
        and one more round trip ranks every vector of the namespace.

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
                candidate's updated_at if that is later, so a correction
                never looks older than the memory it corrects. A memory
                written without one is valid from its updated_at.

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
            method="find_conflicts",
        )
        self._require_columns(self._reconcile_columns(refresh=True))
        vector_rows = self._embed_put(plan.put)
        read = self._key_slot_read(
            plan, self._read_all(self._key_slot_statements(plan), "store find conflicts")
        )
        similar = self._similar_read(plan, vector_rows, read)
        statements = similar.statements()
        found: Sequence[Sequence[dict[str, Any]]] = (
            self._read_all(statements, "store find conflicts similar", similar.session)
            if statements
            else []
        )
        exact = similar.exact_statements(found)
        if exact:
            found = similar.fill_exact(found, self._read_all(exact, "store find conflicts exact"))
        return self._conflicts(plan, vector_rows, read, similar, found)

    def record(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None:
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
        prefix = _memory_prefix(namespace, "record")
        self._require_columns(self._reconcile_columns(refresh=True))
        [rows] = self._read_all([self.sql.record(prefix, str(key))], "store record")
        return _row_to_record(tuple(namespace), rows[0]) if rows else None

    def confirm(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        ttl: float | None | NotProvided | Callable[[MemoryRecord], float | None] = NOT_PROVIDED,
    ) -> bool:
        """Mark a live memory as confirmed, renewing its expiry and counting confirmations.

        This counts the confirmation and renews the expiry. That way expiry can
        follow how often a fact is confirmed, not how often it is read.
        Confirming does not change the version or updated_at, so it never
        invalidates a pending find_conflicts. Confirmations belong to the
        memory's value: a write that changes it starts them again from none.

        Each call picks a confirmation id before its first attempt and stores
        it on the memory, so a retry after a commit whose result was lost
        counts nothing again, unless another confirmation of the same memory
        lands in between, in which case the retry counts again. The memory's
        write id stays as it is, so a write retried after its own lost commit
        still finds it.

        Args:
            namespace: The memory's namespace.
            key: The memory's key.
            ttl: How long the memory stays after confirmation. NOT_PROVIDED
                renews from the row's own TTL. None removes the expiry. A
                number is minutes. A callable takes the MemoryRecord read in
                the same transaction and returns a number of minutes or None.
                Keep it fast and deterministic: it runs again on each retry,
                and must not call the store. It must be a plain function:
                an async one is refused before any I/O.

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
        prefix = _memory_prefix(namespace, "confirm")
        _validate_confirm_ttl(ttl)
        self._require_columns(self._reconcile_columns(refresh=True))

        # Choose the confirm id before the first attempt
        confirm_id = str(uuid.uuid4())
        key_str = str(key)

        def attempt() -> bool:
            cursors: list[Cursor[DictRow]] = []
            try:
                with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label="store confirm",
                ) as cur:
                    conn = cur.connection

                    # If ttl is a callable, read the record first
                    resolved_ttl: float | None | NotProvided
                    if callable(ttl):
                        read_sent = self._send(conn, [self.sql.record(prefix, key_str)], cursors)
                        rows = read_sent[0].fetchall()
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
                    sent = self._send(
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
                check_rows = sent[1].fetchall()
                return self._confirm_result(rowcount, check_rows)
            finally:
                for cursor in cursors:
                    cursor.close()

        return self._retry(attempt, label="store confirm")

    def reconcile(
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

        The candidates are the ones find_conflicts finds: the live memory
        under ``key``, the memories in the same slot and, on a store with an
        index, those that score at or above ``threshold``. They are read in
        the transaction, ``policy`` decides what to do with the new memory
        and with each candidate, and the decision is applied before the
        transaction commits. Serializable isolation covers all of it: if
        another writer changes what the policy saw, one of the two
        transactions retries, and its policy sees the other's write. Two
        agents that reconcile contradicting memories for one user cannot
        both win.

        ``policy`` runs inside the transaction, and runs again on every
        retry, so it must be a plain function, fast and deterministic: an
        async function is refused before any I/O. A slow policy holds the
        transaction open and can run past the attempt timeout. An LLM judge,
        or a person, belongs in find_conflicts and resolve, which hold no
        transaction while the decision is made. A policy must not call the
        store either: the store holds its connection while the policy runs,
        so such a call raises RuntimeError at once. A policy that raises
        rolls the transaction back, and its exception propagates unchanged,
        without a retry. So does the ValueError for an invalid decision.

        The decision writes the new memory, as put does, with its provenance
        and valid time; skips it, as when it repeats a memory the decision
        confirms; or records it as superseded by a candidate whose fact is
        newer. A store with history records such a late fact straight into
        history; a store without history skips it. Each candidate is kept,
        superseded, deleted or confirmed. On a store with history, every
        version the decision ends is kept in history: the memory the new one
        overwrites and each candidate it supersedes end at the ``valid_at``
        given, or at the write without one, and deleted ones end at the
        write. Without history, superseding and deleting remove the memory.
        Confirming sets confirmed_at, counts the confirmation and renews the
        memory's expiry from its own TTL. It changes neither the version nor
        updated_at.

        The index fields of ``value`` are embedded once, before the first
        attempt. Each attempt reads the candidates in two round trips, as
        find_conflicts does, or three when the index's window may have left
        matches out. Then the policy runs, and the writes and the COMMIT go
        out together.

        If an attempt commits and its result is lost, as when the connection
        drops while the COMMIT's answer travels, the retry finds the rows
        that carry its write id and writes nothing. It returns the result
        rebuilt from those rows, with ``already_applied`` set. That result
        lists only what the rows prove: ``kept`` is empty, and on a store
        without history so are ``superseded`` and ``deleted``. A
        confirmation of the new memory in between leaves the write id on it.
        If another writer replaced the new memory before the retry, the
        history row of the version it overwrote, or of a memory it
        superseded, still proves the write: the result says it was written,
        with no version.

        Args:
            namespace: The new memory's namespace. It cannot be empty.
            key: The new memory's key, or None for a new uuid4.
            value: The new memory's value.
            policy: Decides, from the new memory as a Proposal and its
                candidates, what to do with each: a function that returns a
                Decision. langchain_cockroachdb.store has built-in ones.
            slot: A top-level field of ``value`` naming the fact it states,
                as in find_conflicts.
            threshold: The lowest score of a similar candidate, as in
                find_conflicts: 0.8 by default on a cosine store, required on
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
            ValueError: If another argument is invalid, as in find_conflicts;
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
            method="reconcile",
        )
        _check_policy(policy)
        # The schema check comes first, so a store that cannot write costs no
        # embedding call. With history, the write also needs the history table
        if self._history is not None:
            self._history_schema()
        else:
            self._require_columns(self._reconcile_columns(refresh=True))
        vector_rows = self._embed_put(plan.put)
        # The keys that the decisions of earlier attempts named. A retry
        # looks for its write id under them only: an attempt that committed
        # wrote nowhere else, and one that reached no decision wrote nothing
        named: set[str] = set()

        def attempt() -> ReconcileResult:
            cursors: list[Cursor[DictRow]] = []
            try:
                with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label="store reconcile",
                ) as cur:
                    conn = cur.connection

                    def fetch(statements: Sequence[tuple[str, list]]) -> list[list[DictRow]]:
                        # In a pipeline the fetch flushes without a sync, so the
                        # transaction stays open
                        return [
                            statement_cursor.fetchall()
                            for statement_cursor in self._send(conn, statements, cursors)
                        ]

                    # A retry first looks for the rows its write id is on, in
                    # the same round trip as the first candidate read. If an
                    # earlier attempt committed, it returns that result
                    reads = self._key_slot_statements(plan)
                    checks = self._applied_statements(plan, named)
                    found = fetch([*reads, *checks])
                    applied = self._applied_result(plan, found[len(reads) :]) if checks else None
                    if applied is not None:
                        return applied
                    read = self._key_slot_read(plan, found[: len(reads)])
                    similar = self._similar_read(plan, vector_rows, read)
                    for setting in similar.session:
                        cur.execute(setting, prepare=False)
                    scored: Sequence[Sequence[dict[str, Any]]] = fetch(similar.statements())
                    exact = similar.exact_statements(scored)
                    if exact:
                        scored = similar.fill_exact(scored, fetch(exact))
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
                    writes = self._send(conn, statements, cursors)
                # Committed. Only the item upsert returns rows: the version written
                returned = [
                    statement_cursor.fetchall()
                    for statement_cursor in writes
                    if statement_cursor.description is not None
                ]
                return self._reconcile_result(plan, candidates, decision, returned)
            finally:
                for statement_cursor in cursors:
                    statement_cursor.close()

        return self._retry(attempt, label="store reconcile")

    def resolve(self, conflicts: Conflicts, decision: Decision) -> ReconcileResult:
        """Apply a decision about what find_conflicts found, in one transaction.

        This is the second step of the two-step flow: find_conflicts reads
        the candidates, an LLM or a person decides while no transaction is
        open, and resolve applies the decision as reconcile applies a
        policy's. It writes the new memory that find_conflicts planned, with
        its provenance, valid time and write id, and the vectors it embedded
        then, so nothing is embedded again. For the same reason it refuses a
        proposal edited after find_conflicts, rather than drop the edit: to
        change the memory, call find_conflicts again with the edited one.

        In the same transaction, before it writes, resolve checks that
        nothing the decision depends on has changed. The memory under the
        new memory's key, and every candidate the decision supersedes,
        deletes or confirms, or names as superseding the new memory, must
        still have the version and updated_at that find_conflicts read. A
        memory written since counts as changed, also when a 0.5.x store
        wrote it, which sets updated_at without bumping the version. So does
        one deleted or expired since, and a live memory under the new key
        where find_conflicts found none. resolve then raises ConflictsChanged
        with those keys, writes nothing and does not retry: run
        find_conflicts again and decide again. Candidates the decision keeps
        are not checked, and confirming a memory changes neither its version
        nor updated_at, so a confirmation in between never counts.

        The write id comes from find_conflicts. If a resolve of the same
        Conflicts already committed, as when an error left its outcome
        unknown and the caller calls again, resolve writes nothing and
        returns that write's result rebuilt from its rows, with
        ``already_applied`` set, whatever decision this call passes. As with
        reconcile, a rebuilt result lists only what the rows prove: ``kept``
        is empty, and on a store without history so are ``superseded`` and
        ``deleted``. Such a store keeps no row of a decision that only
        supersedes or deletes. When the commit of such a resolve is lost, its
        retry finds those memories gone, as the decision asked. So once an
        attempt of the call may have committed (it sent its COMMIT and failed
        with an error other than a serialization failure or a deadlock, which
        roll the attempt back), a memory the decision supersedes or deletes,
        the one under the new key included, counts as unchanged when it is
        gone. The retry applies the decision again, which changes nothing
        there, and returns its result without ``already_applied``, since no
        row proves the earlier commit. After only serialization failures or
        deadlocks (40001, 40P01), which roll attempts back, the retry checks
        as strictly as the first attempt: a memory gone since was removed by
        another writer, and is a change. Once an attempt may have committed,
        the retries after it stay lenient, whatever error ends them next. A
        new call checks strictly again too: if an error escaped an earlier
        call, as when its retries ran out or the hang guard stopped it, and
        the caller calls resolve again, such a decision raises
        ConflictsChanged naming the keys that the earlier call removed.
        find_conflicts then shows the current state.

        Without a ``valid_at`` given to find_conflicts, the new memory
        stores no valid time, so it is valid from its write, and the
        versions the decision ends end at that write. A Conflicts and a
        Decision can wait in a LangGraph checkpoint while a person decides:
        resolve takes them as the checkpoint gives them back.

        Args:
            conflicts: What find_conflicts returned, on a store with the same
                table and index.
            decision: What to do with the new memory and each candidate.

        Returns:
            What happened, as reconcile reports it.

        Raises:
            ConflictsChanged: If a memory the decision depends on changed
                since find_conflicts. The attempt that found it wrote nothing.
            ValueError: Before any I/O: if ``conflicts`` holds no plan from
                find_conflicts, as when it was built by hand; if a store
                with another table or index made it; if its proposal's
                namespace, key, value, slot, slot value, provenance or valid
                time changed since find_conflicts, the default valid time it
                worked out included; if its candidates' keys, versions or
                updated_at changed; or if ``decision`` is not a valid
                Decision for its proposal and candidates. After a schema
                read: if setup() has not added the columns this method needs
                or, on a store with history, created the history table.
        """
        resolving = self._plan_resolve(conflicts, decision, method="resolve")
        # As in reconcile: with history, the write also needs the history table
        if self._history is not None:
            self._history_schema()
        else:
            self._require_columns(self._reconcile_columns(refresh=True))
        # Whether an earlier attempt of this call may have committed: it sent
        # its COMMIT and failed with an error that is not a definite abort
        committed = False

        def attempt() -> ReconcileResult:
            nonlocal committed
            committing = False
            cursors: list[Cursor[DictRow]] = []
            try:
                with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label="store resolve",
                ) as cur:
                    conn = cur.connection
                    # The check and the reads of an earlier commit go out as one
                    # round trip. In a pipeline the fetch flushes without a sync,
                    # so the transaction stays open
                    reads = self._send(conn, resolving.reads, cursors)
                    found = [statement_cursor.fetchall() for statement_cursor in reads]
                    applied = self._check_resolve(resolving, found, may_have_committed=committed)
                    if applied is not None:
                        return applied
                    writes = self._send(conn, resolving.writes, cursors)
                    # The COMMIT goes out as the block ends
                    committing = True
                # Committed. Only the item upsert returns rows: the version written
                returned = [
                    statement_cursor.fetchall()
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
                    statement_cursor.close()

        return self._retry(attempt, label="store resolve")

    def put_if(
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

        Two threads that read the same memory with record() and both write
        it back with put would lose one update without a word. put_if
        writes only if the live memory under ``key`` still has the version
        and updated_at of ``expected``, the check resolve makes: a put
        moves the version, and a 0.5.x store, which does not, still moves
        updated_at. With ``expected=None`` it writes only if no live memory
        has the key, so a memory is created once. An expired memory counts
        as absent.

        On a match it writes as put does, in the transaction that checked:
        the write id goes on the row, the version it replaces is kept in
        history on a store with history, ending at ``valid_at`` when given
        or else at the write, and the memory's vectors are replaced. The
        result has ``applied`` set and the version written.

        On a mismatch it writes nothing and returns ``applied=False`` with
        ``current``, the live memory as the same transaction read it, or
        None if there is none. A conflict is an expected outcome here, so it
        is returned, not raised: merge and call put_if again with
        ``current``, keep the stored memory, or ask the user.

        The index fields of ``value`` are embedded once, before the first
        attempt. If an attempt commits and its result is lost, as when the
        connection drops while the COMMIT's answer travels, the retry finds
        the write id on the row and returns ``applied=True`` with
        ``already_applied``. If another writer replaced the memory before the
        retry, a 0.5.x store included, the retry reports a conflict with that
        writer's memory, which is the current state.

        Args:
            namespace: The memory's namespace. It cannot be empty.
            key: The memory's key.
            value: The new value.
            expected: The MemoryRecord that record() returned for this
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
            method="put_if",
        )
        # The schema check comes first, so a store that cannot write costs no
        # embedding call. With history, the write also needs the history table
        if self._history is not None:
            self._history_schema()
        else:
            self._require_columns(self._reconcile_columns(refresh=True))
        vector_rows = self._embed_put(plan.put)
        return self._run_conditional(
            plan, self._conditional_writes(plan, vector_rows), label="store put_if"
        )

    def delete_if(
        self, namespace: tuple[str, ...], key: str, *, expected: MemoryRecord | None
    ) -> ConditionalWrite:
        """Delete a memory only if it is still the one the caller read, in one transaction.

        delete_if deletes only if the live memory under ``key`` still has
        the version and updated_at of ``expected``, as put_if checks them.
        On a match it deletes as delete does: on a store with history the
        memory is kept in history as deleted, and its vectors go. With
        ``expected=None`` it applies only while no live memory has the key,
        so it deletes nothing live.

        ``applied=True`` means the memory matched ``expected`` and this call
        deleted it, or, after an earlier attempt of the call that may have
        committed, that the memory is gone and was this call's to delete: it
        matched ``expected`` when that attempt checked it. On a mismatch it
        deletes nothing and returns ``applied=False`` with ``current``, the
        live memory as the same transaction read it, or None if there is
        none, as when another writer deleted it or it expired since. A
        conflict is returned, not raised.

        If an attempt sends its COMMIT and the result is lost, as when the
        connection drops while the COMMIT's answer travels, that attempt may
        have deleted the memory, so a retry that finds it gone reports it
        applied. On a store with history the retry also finds its own history
        row, which proves the earlier attempt deleted the memory, and the
        result has ``already_applied``. A store without history keeps no row
        that proves which attempt deleted it, so ``already_applied`` stays
        False. Serialization failures and deadlocks (40001, 40P01) roll
        attempts back, so after only those a memory gone since is a conflict:
        another writer removed it. Once an attempt may have committed, the
        retries after it stay lenient, whatever error ends them next. If
        another writer put the key again before the retry, the retry reports
        a conflict with that memory.

        Args:
            namespace: The memory's namespace. It cannot be empty.
            key: The memory's key.
            expected: The MemoryRecord that record() returned for this
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
        plan = self._plan_delete_if(namespace, key, expected=expected, method="delete_if")
        # As in put_if: with history, the delete also needs the history table
        if self._history is not None:
            self._history_schema()
        else:
            self._require_columns(self._reconcile_columns(refresh=True))
        return self._run_conditional(plan, self._conditional_writes(plan), label="store delete_if")

    def _run_conditional(
        self, plan: ConditionalPlan, writes: Sequence[tuple[str, list]], *, label: str
    ) -> ConditionalWrite:
        """Run the attempts of put_if or delete_if: read and check, then write and commit.

        Each attempt sends the reads of ``plan`` as one round trip and checks
        their rows while the transaction stays open. A conflict, or a write
        an earlier attempt already committed, returns at once and writes
        nothing. Otherwise ``writes`` go out with the COMMIT.
        """
        # Whether an earlier attempt of this call may have committed: it sent
        # its COMMIT and failed with an error that is not a definite abort.
        # Until one may have, every attempt checks as strictly as the first
        committed = False

        def attempt() -> ConditionalWrite:
            nonlocal committed
            committing = False
            cursors: list[Cursor[DictRow]] = []
            try:
                with self._cursor(
                    pipeline=True,
                    transaction=True,
                    timeout=self.retry_policy.attempt_timeout,
                    label=label,
                ) as cur:
                    conn = cur.connection
                    # In a pipeline the fetch flushes without a sync, so the
                    # transaction stays open
                    reads = self._send(conn, plan.reads(may_have_committed=committed), cursors)
                    found = [statement_cursor.fetchall() for statement_cursor in reads]
                    decided = self._check_conditional(plan, found, may_have_committed=committed)
                    if decided is not None:
                        return decided
                    sent = self._send(conn, writes, cursors)
                    # The COMMIT goes out as the block ends
                    committing = True
                # Committed. Only put_if's item upsert returns rows: the version written
                returned = [
                    statement_cursor.fetchall()
                    for statement_cursor in sent
                    if statement_cursor.description is not None
                ]
                return self._conditional_result(plan, returned)
            except Exception as error:
                committed = committed or (committing and may_have_committed(error))
                raise
            finally:
                for statement_cursor in cursors:
                    statement_cursor.close()

        return self._retry(attempt, label=label)

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
            # Cancelling the future is how a caller stops the sweeper, and a
            # cancelled future takes no result, so both calls check it first
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
                if not future.done():
                    future.set_result(None)
            except Exception as error:
                if not future.done():
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
        Safe to run from several processes at once. On CockroachDB v25.4 a
        TTL change fails while another schema change runs on the same table;
        that error is retried with the budget that schema_change_policy
        gives it, up to ten minutes by default, even if the policy's
        ``retry_on`` would refuse it. ``max_attempts=1`` turns this off.
        """
        policy = schema_change_policy(self.retry_policy)
        for statement in self.sql.disable_ttl_statements():
            self._run_ddl(statement, "store disable TTL", policy=policy)

    def __del__(self) -> None:
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)
