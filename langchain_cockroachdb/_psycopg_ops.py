"""Connection, retry and transaction machinery shared by psycopg components.

The LangGraph checkpointers and the store open psycopg connections
themselves, not through SQLAlchemy. They share the code here: retries under
a RetryPolicy, the hang guard, one operation at a time on a single
connection, and writes with several statements sent as one
``BEGIN ... COMMIT`` batch. A host class names itself in ``component``
(``"checkpointer"`` or ``"store"``), which log and error messages use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, ClassVar, TypeVar

from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Connection, Cursor, Pipeline
from psycopg.pq import TransactionStatus
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb._guard import aabandon_after, abandon_after
from langchain_cockroachdb._loop import LoopMessages, run_on_external_loop
from langchain_cockroachdb.checkpointer import _ainternal, _internal
from langchain_cockroachdb.retry import RetryPolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _check_component(cls: type) -> None:
    """Raise TypeError unless a host class names itself in ``component``."""
    component = getattr(cls, "component", None)
    if not isinstance(component, str) or not component:
        raise TypeError(
            f"{cls.__name__} must set the class attribute component, a non-empty "
            'string such as "checkpointer" or "store", used in log and error messages'
        )


def _operation_label(component: str) -> str:
    """The default hang-guard label: "checkpoint operation" keeps the 0.4.1 text."""
    noun = "checkpoint" if component == "checkpointer" else component
    return f"{noun} operation"


class PsycopgOps:
    """Sync mixin: retries, the hang guard and one-batch writes on psycopg.

    The host sets ``conn`` (a connection or a pool), ``pipe``, ``lock``,
    ``supports_pipeline``, ``retry_policy`` and ``_batch_open`` in its
    __init__, and ``component`` on the class.
    """

    # Names the host in log and error messages: "checkpointer" or "store". A
    # subclass that does not set it fails with TypeError when it is defined.
    component: ClassVar[str]
    # Where _can_retry logs. The default is this module's logger; a host can
    # point it at its own module's logger
    _log: ClassVar[logging.Logger] = logger

    conn: _internal.Conn
    pipe: Pipeline | None
    # Runs operations one at a time on a single connection; a pool needs no lock
    lock: threading.Lock
    supports_pipeline: bool
    retry_policy: RetryPolicy
    # Set while a write batch of ours is between BEGIN and COMMIT on a single connection
    _batch_open: bool

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _check_component(cls)

    def _retry(self, fn: Callable[[], T], *, label: str) -> T:
        """Run ``fn`` under the retry policy."""
        return self.retry_policy.run(fn, label=label, can_retry=self._can_retry)

    def _sync_pipeline(self) -> None:
        """Sync pipeline if one is in use. Each statement runs in its own transaction."""
        if self.pipe:
            self.pipe.sync()

    def _can_retry(self, error: BaseException) -> bool:
        if isinstance(self.conn, Connection) and self.conn.closed:
            self._log.warning(
                "Not retrying: the %s's only connection is closed and cannot be "
                "reopened. Pass a psycopg_pool.ConnectionPool so broken connections are replaced.",
                self.component,
            )
            return False
        # The caller owns its transaction, so it decides whether to retry all of it
        return not _internal.in_caller_transaction(error)

    def _run_ddl(self, statement: str, label: str) -> None:
        """Run a single DDL statement with retry and timeout."""

        def execute() -> None:
            with self._cursor(timeout=self.retry_policy.ddl_attempt_timeout, label=label) as cur:
                cur.execute(statement)
                self._sync_pipeline()

        self._retry(execute, label=label)

    def _operation_lock(self) -> contextlib.AbstractContextManager[object]:
        """The lock an operation holds while it uses the connection.

        A single connection runs one operation at a time. A pool gives every
        operation its own connection, so pooled operations take no lock and run
        concurrently. What they share is the pool, which is thread-safe, and
        attributes that are only set in __init__.
        """
        if isinstance(self.conn, ConnectionPool):
            return contextlib.nullcontext()
        return self.lock

    @staticmethod
    def _transaction(
        conn: Connection[DictRow], transaction: bool
    ) -> contextlib.AbstractContextManager[object]:
        """``conn.transaction()`` if the statements must commit together, else nothing."""
        return conn.transaction() if transaction else contextlib.nullcontext()

    def _roll_back_batch(self, conn: Connection[DictRow], *, synced: bool) -> bool:
        """Roll back a write batch of ours that stopped between BEGIN and COMMIT.

        ``synced`` says whether libpq's status is current. On a
        component-wide pipeline it is only current once a sync has
        completed: until then it can read IDLE while the batch is open. So
        this syncs first if needed, then rolls back an open transaction;
        psycopg's rollback() syncs a pipeline before and after the ROLLBACK.
        Errors are ignored, each step on its own: results still on their way
        often carry the batch's error, and a connection that broke cannot
        roll back. Returns whether the status is now current.
        """
        if self.pipe and not synced:
            with contextlib.suppress(Exception):
                self.pipe.sync()
                synced = True
        if conn.info.transaction_status in _internal.OPEN_TRANSACTION:
            with contextlib.suppress(Exception):
                conn.rollback()
                synced = True
        return synced

    def _settle_batch_flag(self, conn: Connection[DictRow], *, synced: bool) -> None:
        """Clear the batch flag once no transaction of ours can be open on ``conn``.

        Called in a finally, so an interruption cannot skip it. UNKNOWN means
        the connection is gone. IDLE counts only when the status is current
        (``synced``): then the batch committed or rolled back. Otherwise the
        flag stays set, and the next operation looks again after a sync.
        """
        status = conn.info.transaction_status
        if status == TransactionStatus.UNKNOWN or (synced and status == TransactionStatus.IDLE):
            self._batch_open = False

    def _prepare_batch(
        self, conn: Connection[DictRow], *, pipeline: bool, transaction: bool
    ) -> bool:
        """Finish rolling back a stopped batch of ours, then say whether to batch.

        If that batch is still open and cannot be rolled back, this raises
        instead of letting the operation nest in it. A write is batched when
        its statements must commit together, it is pipelined, the connection
        is in autocommit mode, and libpq reports it idle, so no transaction
        the caller opened is open on it: BEGIN there would fail and abort the
        caller's transaction. On a component-wide pipeline the component's
        own last read can still hold an implicit transaction while the status
        reads idle; the batch's BEGIN then joins it.
        """
        if self._batch_open:
            # Outside pipeline mode libpq's status is current
            synced = not self.pipe
            try:
                synced = self._roll_back_batch(conn, synced=synced)
            finally:
                self._settle_batch_flag(conn, synced=synced)
            if self._batch_open and conn.info.transaction_status in _internal.OPEN_TRANSACTION:
                raise _internal.stuck_batch_error(self.component)
        return (
            transaction
            and pipeline
            and conn.autocommit
            and conn.info.transaction_status == TransactionStatus.IDLE
            and (bool(self.pipe) or self.supports_pipeline)
        )

    @contextmanager
    def _on_component_pipeline(
        self,
        conn: Connection[DictRow],
        pipe: Pipeline,
        *,
        pipeline: bool,
        transaction: bool,
        batched: bool,
    ) -> Iterator[Cursor[DictRow]]:
        """Run the block on the component-wide pipeline, resyncing it after an error."""
        # Whether a sync completed, so that libpq's status is current
        synced = False
        try:
            with conn.cursor() as tx, conn.cursor(binary=True, row_factory=dict_row) as cur:
                if batched:
                    self._batch_open = True
                    tx.execute("BEGIN", prepare=False)
                with self._transaction(conn, transaction and not batched):
                    yield cur
                if batched:
                    tx.execute("COMMIT", prepare=False)
                # psycopg's transaction syncs the pipeline itself as it ends
                if batched or (pipeline and not transaction):
                    pipe.sync()
                    synced = True
                if batched:
                    _internal.check_committed(tx)
        except BaseException:
            # A pipeline only exists on a single connection, so the lock is held
            # here. Resync before releasing it, so other threads' queued statements
            # are untouched, and roll back what a stopped batch left open
            with contextlib.suppress(Exception):
                pipe.sync()
                synced = True
            if batched:
                synced = self._roll_back_batch(conn, synced=synced)
            raise
        finally:
            if batched:
                self._settle_batch_flag(conn, synced=synced)

    @contextmanager
    def _in_one_batch(self, conn: Connection[DictRow]) -> Iterator[Cursor[DictRow]]:
        """Send BEGIN, the block's statements and COMMIT with one pipeline sync."""
        own_connection = not isinstance(self.conn, ConnectionPool)
        with conn.cursor() as tx, conn.cursor(binary=True, row_factory=dict_row) as cur:
            try:
                with conn.pipeline():
                    if own_connection:
                        self._batch_open = True
                    tx.execute("BEGIN", prepare=False)
                    yield cur
                    tx.execute("COMMIT", prepare=False)
            except BaseException:
                # The pipeline's exit has synced, so the status is current
                self._roll_back_batch(conn, synced=True)
                raise
            finally:
                if own_connection:
                    self._settle_batch_flag(conn, synced=True)
            _internal.check_committed(tx)

    @contextmanager
    def _cursor(
        self,
        *,
        pipeline: bool = False,
        transaction: bool = False,
        timeout: float | None = None,
        label: str | None = None,
    ) -> Iterator[Cursor[DictRow]]:
        """Create a database cursor as a context manager, with the hang guard.

        ``pipeline`` sends the statements together, in pipeline mode when libpq
        has it. ``transaction`` makes them commit together: without it, the sync
        that ends a pipeline commits whatever was sent, even when the block
        stopped partway through. On an idle autocommit connection in pipeline
        mode, BEGIN, the statements and COMMIT go to the server as one batch.
        That takes one round trip, and CockroachDB can retry the whole batch
        itself. If the block stops between BEGIN and COMMIT, the transaction is
        rolled back before the connection is used again. Inside a transaction
        the caller opened, and on other connections, psycopg's
        ``conn.transaction()`` runs the statements; in an open transaction it
        nests a savepoint. A single statement commits on its own and needs no
        transaction. An error inside a transaction the caller opened is marked,
        so the component raises it at once instead of retrying. Without a
        ``label``, the hang guard names the operation after ``component``.
        """
        if label is None:
            label = _operation_label(self.component)
        with self._operation_lock(), _internal.get_connection(self.conn) as conn:
            fileno = conn.fileno() if timeout is not None else None
            with abandon_after(fileno, timeout, label=label):
                batched = self._prepare_batch(conn, pipeline=pipeline, transaction=transaction)
                # Open now and not a stopped batch of ours: the caller's transaction
                in_callers_transaction = (
                    not self._batch_open
                    and conn.info.transaction_status in _internal.OPEN_TRANSACTION
                )
                try:
                    if self.pipe:
                        with self._on_component_pipeline(
                            conn,
                            self.pipe,
                            pipeline=pipeline,
                            transaction=transaction,
                            batched=batched,
                        ) as cur:
                            yield cur
                    elif batched:
                        with self._in_one_batch(conn) as cur:
                            yield cur
                    elif pipeline and self.supports_pipeline:
                        # psycopg's transaction goes around the pipeline, which saves a
                        # round trip
                        with (
                            self._transaction(conn, transaction),
                            conn.pipeline(),
                            conn.cursor(binary=True, row_factory=dict_row) as cur,
                        ):
                            yield cur
                    else:
                        with (
                            self._transaction(conn, transaction),
                            conn.cursor(binary=True, row_factory=dict_row) as cur,
                        ):
                            yield cur
                except Exception as error:
                    if in_callers_transaction:
                        _internal.mark_in_caller_transaction(error)
                    raise


class AsyncPsycopgOps:
    """Async mixin: retries, the hang guard and one-batch writes on psycopg.

    The host sets ``conn`` (a connection or a pool), ``pipe``, ``lock``,
    ``supports_pipeline``, ``retry_policy`` and ``_batch_open`` in its
    __init__, and ``component`` on the class. For ``_run_sync`` it also sets
    ``loop`` and ``_pid``, and ``_own_loop_message`` and ``_loop_messages``
    on the class.
    """

    # Names the host in log and error messages: "checkpointer" or "store". A
    # subclass that does not set it fails with TypeError when it is defined.
    component: ClassVar[str]
    # Where _can_retry logs. The default is this module's logger; a host can
    # point it at its own module's logger
    _log: ClassVar[logging.Logger] = logger
    # What _run_sync raises from the host's own loop, and when it gives up on it
    _own_loop_message: ClassVar[str]
    _loop_messages: ClassVar[LoopMessages]

    conn: _ainternal.Conn
    pipe: AsyncPipeline | None
    # Runs operations one at a time on a single connection; a pool needs no lock
    lock: asyncio.Lock
    supports_pipeline: bool
    retry_policy: RetryPolicy
    # Set while a write batch of ours is between BEGIN and COMMIT on a single connection
    _batch_open: bool
    # The event loop the host was created on, and the process that created it
    loop: asyncio.AbstractEventLoop
    _pid: int

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _check_component(cls)

    async def _retry(self, fn: Callable[[], Awaitable[T]], *, label: str) -> T:
        """Run ``fn`` under the retry policy."""
        return await self.retry_policy.arun(fn, label=label, can_retry=self._can_retry)

    async def _sync_pipeline(self) -> None:
        """Sync pipeline if one is in use. Each statement runs in its own transaction."""
        if self.pipe:
            await self.pipe.sync()

    def _can_retry(self, error: BaseException) -> bool:
        if isinstance(self.conn, AsyncConnection) and self.conn.closed:
            self._log.warning(
                "Not retrying: the %s's only connection is closed and cannot be "
                "reopened. Pass a psycopg_pool.AsyncConnectionPool so broken connections "
                "are replaced.",
                self.component,
            )
            return False
        # The caller owns its transaction, so it decides whether to retry all of it
        return not _internal.in_caller_transaction(error)

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the component's loop and wait for the result.

        Raises:
            asyncio.InvalidStateError: If called from that loop, where waiting
                would deadlock
            RuntimeError: If that loop is closed, is not running or stops
                before the work finishes, or if this is a forked child of the
                process that created the component
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            coro.close()
            raise asyncio.InvalidStateError(self._own_loop_message)
        return run_on_external_loop(
            coro, self.loop, owner_pid=self._pid, messages=self._loop_messages
        )

    def _operation_lock(self) -> contextlib.AbstractAsyncContextManager[object]:
        """The lock an operation holds while it uses the connection.

        A single connection runs one operation at a time. A pool gives every
        operation its own connection, so pooled operations take no lock and run
        concurrently. What they share is the pool, which is safe for concurrent
        tasks, and attributes that are only set in __init__.
        """
        if isinstance(self.conn, AsyncConnectionPool):
            return contextlib.nullcontext()
        return self.lock

    @staticmethod
    def _transaction(
        conn: AsyncConnection[DictRow], transaction: bool
    ) -> contextlib.AbstractAsyncContextManager[object]:
        """``conn.transaction()`` if the statements must commit together, else nothing."""
        return conn.transaction() if transaction else contextlib.nullcontext()

    async def _roll_back_batch(self, conn: AsyncConnection[DictRow], *, synced: bool) -> bool:
        """Roll back a write batch of ours that stopped between BEGIN and COMMIT.

        ``synced`` says whether libpq's status is current. On a
        component-wide pipeline it is only current once a sync has
        completed: until then it can read IDLE while the batch is open. So
        this syncs first if needed, then rolls back an open transaction;
        psycopg's rollback() syncs a pipeline before and after the ROLLBACK.
        Errors are ignored, each step on its own: results still on their way
        often carry the batch's error, and a connection that broke cannot
        roll back. Returns whether the status is now current.
        """
        if self.pipe and not synced:
            with contextlib.suppress(Exception):
                await self.pipe.sync()
                synced = True
        if conn.info.transaction_status in _internal.OPEN_TRANSACTION:
            with contextlib.suppress(Exception):
                await conn.rollback()
                synced = True
        return synced

    def _settle_batch_flag(self, conn: AsyncConnection[DictRow], *, synced: bool) -> None:
        """Clear the batch flag once no transaction of ours can be open on ``conn``.

        Called in a finally, so a cancellation cannot skip it. UNKNOWN means
        the connection is gone. IDLE counts only when the status is current
        (``synced``): then the batch committed or rolled back. Otherwise the
        flag stays set, and the next operation looks again after a sync.
        """
        status = conn.info.transaction_status
        if status == TransactionStatus.UNKNOWN or (synced and status == TransactionStatus.IDLE):
            self._batch_open = False

    async def _prepare_batch(
        self, conn: AsyncConnection[DictRow], *, pipeline: bool, transaction: bool
    ) -> bool:
        """Finish rolling back a stopped batch of ours, then say whether to batch.

        If that batch is still open and cannot be rolled back, this raises
        instead of letting the operation nest in it. A write is batched when
        its statements must commit together, it is pipelined, the connection
        is in autocommit mode, and libpq reports it idle, so no transaction
        the caller opened is open on it: BEGIN there would fail and abort the
        caller's transaction. On a component-wide pipeline the component's
        own last read can still hold an implicit transaction while the status
        reads idle; the batch's BEGIN then joins it.
        """
        if self._batch_open:
            # Outside pipeline mode libpq's status is current
            synced = not self.pipe
            try:
                synced = await self._roll_back_batch(conn, synced=synced)
            finally:
                self._settle_batch_flag(conn, synced=synced)
            if self._batch_open and conn.info.transaction_status in _internal.OPEN_TRANSACTION:
                raise _internal.stuck_batch_error(self.component)
        return (
            transaction
            and pipeline
            and conn.autocommit
            and conn.info.transaction_status == TransactionStatus.IDLE
            and (bool(self.pipe) or self.supports_pipeline)
        )

    @asynccontextmanager
    async def _on_component_pipeline(
        self,
        conn: AsyncConnection[DictRow],
        pipe: AsyncPipeline,
        *,
        pipeline: bool,
        transaction: bool,
        batched: bool,
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Run the block on the component-wide pipeline, resyncing it after an error."""
        # Whether a sync completed, so that libpq's status is current
        synced = False
        try:
            async with (
                conn.cursor() as tx,
                conn.cursor(binary=True, row_factory=dict_row) as cur,
            ):
                if batched:
                    self._batch_open = True
                    await tx.execute("BEGIN", prepare=False)
                async with self._transaction(conn, transaction and not batched):
                    yield cur
                if batched:
                    await tx.execute("COMMIT", prepare=False)
                # psycopg's transaction syncs the pipeline itself as it ends
                if batched or (pipeline and not transaction):
                    await pipe.sync()
                    synced = True
                if batched:
                    _internal.check_committed(tx)
        except BaseException:
            # A pipeline only exists on a single connection, so the lock is held
            # here. Resync before releasing it, so other tasks' queued statements
            # are untouched, and roll back what a stopped batch left open
            with contextlib.suppress(Exception):
                await pipe.sync()
                synced = True
            if batched:
                synced = await self._roll_back_batch(conn, synced=synced)
            raise
        finally:
            if batched:
                self._settle_batch_flag(conn, synced=synced)

    @asynccontextmanager
    async def _in_one_batch(
        self, conn: AsyncConnection[DictRow]
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Send BEGIN, the block's statements and COMMIT with one pipeline sync."""
        own_connection = not isinstance(self.conn, AsyncConnectionPool)
        async with (
            conn.cursor() as tx,
            conn.cursor(binary=True, row_factory=dict_row) as cur,
        ):
            try:
                async with conn.pipeline():
                    if own_connection:
                        self._batch_open = True
                    await tx.execute("BEGIN", prepare=False)
                    yield cur
                    await tx.execute("COMMIT", prepare=False)
            except BaseException:
                # The pipeline's exit has synced, so the status is current
                await self._roll_back_batch(conn, synced=True)
                raise
            finally:
                if own_connection:
                    self._settle_batch_flag(conn, synced=True)
            _internal.check_committed(tx)

    @asynccontextmanager
    async def _cursor(
        self,
        *,
        pipeline: bool = False,
        transaction: bool = False,
        timeout: float | None = None,
        label: str | None = None,
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager, with the hang guard.

        ``pipeline`` sends the statements together, in pipeline mode when libpq
        has it. ``transaction`` makes them commit together: without it, the sync
        that ends a pipeline commits whatever was sent, even when the block
        stopped partway through. On an idle autocommit connection in pipeline
        mode, BEGIN, the statements and COMMIT go to the server as one batch.
        That takes one round trip, and CockroachDB can retry the whole batch
        itself. If the block stops between BEGIN and COMMIT, a cancellation
        included, the transaction is rolled back before the connection is used
        again. Inside a transaction the caller opened, and on other
        connections, psycopg's ``conn.transaction()`` runs the statements; in
        an open transaction it nests a savepoint. A single statement commits on
        its own and needs no transaction. An error inside a transaction the
        caller opened is marked, so the component raises it at once instead
        of retrying. Without a ``label``, the hang guard names the operation
        after ``component``.
        """
        if label is None:
            label = _operation_label(self.component)
        async with self._operation_lock(), _ainternal.get_connection(self.conn) as conn:
            fileno = conn.fileno() if timeout is not None else None
            async with aabandon_after(fileno, timeout, label=label):
                batched = await self._prepare_batch(
                    conn, pipeline=pipeline, transaction=transaction
                )
                # Open now and not a stopped batch of ours: the caller's transaction
                in_callers_transaction = (
                    not self._batch_open
                    and conn.info.transaction_status in _internal.OPEN_TRANSACTION
                )
                try:
                    if self.pipe:
                        async with self._on_component_pipeline(
                            conn,
                            self.pipe,
                            pipeline=pipeline,
                            transaction=transaction,
                            batched=batched,
                        ) as cur:
                            yield cur
                    elif batched:
                        async with self._in_one_batch(conn) as cur:
                            yield cur
                    elif pipeline and self.supports_pipeline:
                        # psycopg's transaction goes around the pipeline, which saves a
                        # round trip
                        async with (
                            self._transaction(conn, transaction),
                            conn.pipeline(),
                            conn.cursor(binary=True, row_factory=dict_row) as cur,
                        ):
                            yield cur
                    else:
                        async with (
                            self._transaction(conn, transaction),
                            conn.cursor(binary=True, row_factory=dict_row) as cur,
                        ):
                            yield cur
                except Exception as error:
                    if in_callers_transaction:
                        _internal.mark_in_caller_transaction(error)
                    raise

    async def _run_ddl(self, statement: str, label: str) -> None:
        """Run a single DDL statement with retry and timeout."""

        async def execute() -> None:
            async with self._cursor(
                timeout=self.retry_policy.ddl_attempt_timeout, label=label
            ) as cur:
                await cur.execute(statement)
                await self._sync_pipeline()

        await self._retry(execute, label=label)
