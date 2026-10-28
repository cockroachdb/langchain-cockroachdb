"""Synchronous CockroachDB store for LangGraph long-term memory.

Provides the interface of langgraph-checkpoint-postgres's PostgresStore on
CockroachDB. Every operation retries transient errors under the store's
RetryPolicy. Every write is an UPSERT, a DELETE, an UPDATE that pushes back
an expiry, or an INSERT into history that skips versions already there, so
a retry after an ambiguous result is safe.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    InvalidNamespaceError,
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
from langchain_cockroachdb.retry import RetryPolicy, schema_change_policy
from langchain_cockroachdb.store.base import (
    BaseCockroachDBStore,
    BatchPlan,
    ProposalPlan,
    PutPlan,
    SchemaCheck,
    _check_paging,
    _memory_prefix,
    _row_to_record,
    _row_to_version,
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
from langchain_cockroachdb.store.namespaces import validate_labels
from langchain_cockroachdb.store.reconcile import Conflicts, MemoryRecord, MemoryVersion
from langchain_cockroachdb.store.sql import VectorRow

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
                candidates through the C-SPANN index, which may miss some.
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
        """
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

        # A write before setup() may have cached the schema without
        # migration 2. The next write reads it again.
        self._columns_ready = None

        for statement in self.sql.vector_statements():
            self._run_ddl(statement, "store vector setup")
        if self.index is not None:
            self._check_index()
        if self._history is not None:
            policy = schema_change_policy(self.retry_policy)
            for statement in self.sql.history_statements(self._history.ttl_days):
                self._run_ddl(statement, "store history setup", policy=policy)
            # A write before setup() may have found no history table. The
            # next write reads the schema again.
            self._history_ready = None
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
        exact = plan.exact_statements()
        if exact:
            plan.fill_exact(self._read_all(exact, "store search exact"))
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

    def _schema(self, *, refresh: bool = False) -> SchemaCheck:
        """What the schema check knows: migration 2's columns and the history table.

        It reads the latest migration, and on a store with history whether
        the history table exists, in one round trip. It reads them the first
        time, and again when an answer it found missing is older than
        _SCHEMA_RECHECK_SECONDS, or ``refresh`` is set. What it found present
        it trusts for good.
        """
        schema = self._cached_schema(refresh=refresh)
        if schema is None:
            schema = self._cache_schema(
                self._read_all(self._schema_statements(), "store schema check")
            )
        return schema

    def _reconcile_columns(self, *, refresh: bool = False) -> bool:
        """Whether migration 2's columns exist, as the schema check last read it.

        It reads the latest migration the first time, and again when it last
        found none and that answer is older than _SCHEMA_RECHECK_SECONDS, or
        ``refresh`` is set. Methods that need the columns pass ``refresh``, so
        they see a setup() that another process ran since. Until setup()
        records migration 2, puts write with the 0.5 statements and those
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

    def _run_puts(self, plan: PutPlan) -> None:
        # The schema check comes first. With history, a write refuses to run
        # until setup() has created the history table, since it could not
        # keep what it replaces, and a refused batch costs no embedding call
        schema = self._history_schema() if self._history is not None else self._schema()
        # Embed once, before the first attempt: a retry reuses the vectors
        embeddings: list[list[float]] = []
        if self.index is not None and plan.embed_requests:
            embeddings = self.index.embeddings.embed_documents(
                [text for _, _, _, text in plan.embed_requests]
            )
        vector_rows = self._vector_rows(plan, embeddings)
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
        prefix = validate_labels(namespace_prefix)
        if not prefix:
            raise InvalidNamespaceError(
                "delete_namespace needs a non-empty namespace prefix. To clear the whole "
                "store on purpose, delete each top-level namespace from list_namespaces("
                "max_depth=1), or drop the store's tables."
            )
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

        A store with history keeps every version that a put overwrites
        (reason "overwritten") or a delete ends ("deleted"). The live value
        is not one of them; get() returns it.

        Args:
            namespace: The memory's namespace.
            key: The memory's key.
            limit: Most versions to return (default 20).

        Returns:
            Up to ``limit`` versions, highest version first. Versions past
            the history's ``ttl_days`` are left out.

        Raises:
            ValueError: If the store keeps no history (create it with
                history=True and run setup()), setup() has not created the
                history table, or ``limit`` is negative
            InvalidNamespaceError: If the namespace is empty or a label is invalid
        """
        prefix = _memory_prefix(namespace, "history")
        _check_paging(limit, 0)
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
          scores search returns.

        Key and slot matches carry their best score when they have vectors,
        and similar ones always do. Candidates come strongest reason first,
        then best score, then key. Expired memories are never candidates,
        and no read pushes back an expiry.

        The index fields of ``value`` are embedded once, and the embedding is
        kept for the write. Then two round trips read the candidates: first
        the database's clock, the key and slot matches and the namespace's
        vector count, then the similar memories and the scores. A namespace
        with at most ``reconcile_scan_limit`` vectors has all of them ranked.
        A larger one takes its similar candidates from the C-SPANN index,
        which may miss some.

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
        vector_rows = self._embed_proposal(plan)
        read = self._key_slot_read(
            plan, self._read_all(self._key_slot_statements(plan), "store find conflicts")
        )
        similar = self._similar_read(plan, vector_rows, read)
        statements = similar.statements()
        found = (
            self._read_all(statements, "store find conflicts similar", similar.session)
            if statements
            else []
        )
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

    def _embed_proposal(self, plan: ProposalPlan) -> list[VectorRow]:
        """Embed the new memory's index fields once, and pair the vectors with their rows.

        Runs before the first attempt of any read, so a retry reuses the
        vectors. embed_documents, not embed_query: the vectors are also the
        ones its write stores.
        """
        embeddings: list[list[float]] = []
        if self.index is not None and plan.embed_requests:
            embeddings = self.index.embeddings.embed_documents(
                [text for _, _, _, text in plan.embed_requests]
            )
        return self._vector_rows(plan.put, embeddings)

    def reconcile(
        self,
        namespace: tuple[str, ...],
        key: str | None,
        value: dict[str, Any],
        *,
        policy: Callable[..., Any],
        slot: str | None = None,
        threshold: float | None = None,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        index: Literal[False] | list[str] | None = None,
        ttl: float | None | NotProvided = NOT_PROVIDED,
        provenance: dict[str, Any] | None = None,
        valid_at: datetime | None = None,
    ) -> Any:
        """Write a memory and settle the memories it may conflict with.

        Not available yet: it raises NotImplementedError once the table has
        migration 2's columns.

        Raises:
            ValueError: If setup() has not applied migration 2.
        """
        self._require_columns(self._reconcile_columns(refresh=True))
        raise NotImplementedError("reconcile arrives in a later task")

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
        that error is retried under the store's retry policy, even if its
        ``retry_on`` would refuse it. ``max_attempts=1`` turns this off.
        """
        policy = schema_change_policy(self.retry_policy)
        for statement in self.sql.disable_ttl_statements():
            self._run_ddl(statement, "store disable TTL", policy=policy)

    def __del__(self) -> None:
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)
