"""Fake psycopg connections, cursors and pools for the checkpointer unit tests.

The fakes stand in for what libpq does when it loses a connection: the call
fails, and psycopg reports the connection as broken. Over TLS, OpenSSL 3 words
that failure "unexpected eof while reading", which CockroachDB Cloud showed
after the hang guard shut the socket of a stuck write.

Test modules that use the fakes call ``close_sockets()`` after each test.
The unit tests' conftest has psycopg's Python send() drain them: its C
version only takes a real libpq connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Sequence
from typing import Any, TypeVar

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.waiting import Ready
from psycopg_pool import AsyncConnectionPool, ConnectionPool

T = TypeVar("T")

_sockets: list[socket.socket] = []


def close_sockets() -> None:
    """Close the socket pairs the fakes opened for the hang guard."""
    while _sockets:
        _sockets.pop().close()


class Block:
    """A pipeline or transaction block that records when it opens and closes.

    The close event names the error that left the block, or is None when the
    block finished cleanly. A transaction that closes with an error is one
    psycopg rolls back.
    """

    def __init__(self, events: list[tuple[str, ...]], name: str) -> None:
        self.events = events
        self.name = name

    def __enter__(self) -> None:
        self.events.append((self.name, "open"))

    def __exit__(self, exc_type: type[BaseException] | None, *rest: object) -> None:
        self.events.append((self.name, "close", exc_type.__name__ if exc_type else None))

    async def __aenter__(self) -> None:
        self.__enter__()

    async def __aexit__(self, exc_type: type[BaseException] | None, *rest: object) -> None:
        self.__exit__(exc_type)


class FakeInfo:
    """The part of psycopg's ConnectionInfo the savers read."""

    def __init__(self) -> None:
        self.transaction_status = TransactionStatus.IDLE


class FakePGconn:
    """The part of libpq's connection that psycopg's send() uses.

    flush() returns the next of ``flushes``, then 0: nothing is left to send.
    ``io`` records each flush() and each wait for the socket, in order. Pass
    a connection's ``events`` as ``io`` to see them among its statements.
    """

    def __init__(
        self, flushes: Sequence[int] = (), io: list[tuple[Any, ...]] | None = None
    ) -> None:
        self.flushes = list(flushes)
        self.io: list[tuple[Any, ...]] = [] if io is None else io

    def flush(self) -> int:
        result = self.flushes.pop(0) if self.flushes else 0
        self.io.append(("flush", result))
        return result

    def consume_input(self) -> None:
        self.io.append(("consume_input",))


class FakeConnection:
    """A psycopg connection with only the parts the savers use.

    Statement number ``fail_at`` (the first by default) fails with ``error``
    when one is given. With ``stuck`` it first waits until the hang guard
    shuts down the connection's socket. With ``breaks``, psycopg's broken flag
    is set as the error is raised, the way libpq marks a connection it lost.
    ``on_statement`` is called with each statement as it runs (awaited for the
    async fakes), so a test can hold an operation inside its connection.
    ``events`` records statements, rollbacks and the pipeline and transaction
    blocks around them, in order.

    BEGIN, COMMIT, a failing statement and rollback() move
    ``info.transaction_status`` the way they move libpq's. The server answers
    COMMIT with ``commit_answer``. With ``in_pipeline_mode`` set, rollback()
    syncs the way psycopg's does. The first ``rollback_interruptions``
    rollbacks are interrupted before they start, the way a second cancellation
    or Ctrl-C would stop them. The first ``interrupted_after_rollback``
    rollbacks finish and then raise the interruption, the way psycopg's wait()
    lets an interrupted rollback finish before it re-raises. Every rollback
    fails with ``rollback_error`` when one is given.
    """

    autocommit = True
    # psycopg's default. Set as a class attribute, it also hides psycopg's own
    # property in the subclasses below, which never run psycopg's __init__
    prepare_threshold: int | None = 5

    def __init__(
        self,
        error: BaseException | None = None,
        *,
        fail_at: int = 1,
        breaks: bool = False,
        stuck: bool = False,
        on_statement: Callable[[str], Any] | None = None,
        commit_answer: str = "COMMIT",
        rollback_interruptions: int = 0,
        interrupted_after_rollback: int = 0,
        rollback_error: BaseException | None = None,
    ) -> None:
        self.error = error
        self.fail_at = fail_at
        self.breaks = breaks
        self.stuck = stuck
        self.on_statement = on_statement
        self.commit_answer = commit_answer
        self.rollback_interruptions = rollback_interruptions
        self.interrupted_after_rollback = interrupted_after_rollback
        self.rollback_error = rollback_error
        self.in_pipeline_mode = False
        self.statements: list[str] = []
        self.events: list[tuple[str, ...]] = []
        self._info = FakeInfo()
        self._broken = False
        self._socket: socket.socket | None = None
        self.pgconn = FakePGconn()

    def __del__(self) -> None:
        """Nothing to close, so psycopg's warning about open connections does not apply."""

    @property
    def info(self) -> FakeInfo:
        return self._info

    def wait(self, gen: Generator[Any, Any, T]) -> T:
        """Run a psycopg generator. The socket is ready to write at every wait."""
        try:
            state = next(gen)
            while True:
                self.pgconn.io.append(("wait", state))
                state = gen.send(Ready.W)
        except StopIteration as stop:
            return stop.value

    @property
    def broken(self) -> bool:
        return self._broken

    @property
    def closed(self) -> bool:
        return self._broken

    def fileno(self) -> int:
        if self._socket is None:
            self._socket, peer = socket.socketpair()
            _sockets.extend((self._socket, peer))
        return self._socket.fileno()

    def pipeline(self) -> Block:
        return Block(self.events, "pipeline")

    def transaction(self) -> Block:
        return Block(self.events, "transaction")

    def cursor(self, **kwargs: Any) -> FakeCursor:
        return FakeCursor(self)

    def _record(self, query: Any) -> bool:
        """Record a statement; True if it is the one that fails."""
        self.statements.append(str(query))
        self.events.append(("statement", str(query)))
        if str(query) == "BEGIN":
            self._info.transaction_status = TransactionStatus.INTRANS
        elif str(query) == "COMMIT":
            self._info.transaction_status = TransactionStatus.IDLE
        return self.error is not None and len(self.statements) == self.fail_at

    def _fail(self) -> None:
        self._broken = self.breaks
        if self._info.transaction_status == TransactionStatus.INTRANS:
            self._info.transaction_status = TransactionStatus.INERROR
        assert self.error is not None
        raise self.error

    def _roll_back(self, interruption: type[BaseException]) -> None:
        """What psycopg's rollback() sends: a pipeline is synced before and after."""
        if self.rollback_interruptions:
            self.rollback_interruptions -= 1
            raise interruption()
        if self.rollback_error is not None:
            raise self.rollback_error
        if self.in_pipeline_mode:
            self.events.append(("sync",))
        if self._info.transaction_status == TransactionStatus.IDLE:
            return
        self.events.append(("rollback",))
        self._info.transaction_status = TransactionStatus.IDLE
        if self.in_pipeline_mode:
            self.events.append(("sync",))
        if self.interrupted_after_rollback:
            self.interrupted_after_rollback -= 1
            raise interruption()

    def rollback(self) -> None:
        self._roll_back(KeyboardInterrupt)

    def run(self, query: Any) -> None:
        fails = self._record(query)
        if self.on_statement is not None:
            self.on_statement(str(query))
        if fails:
            if self.stuck:
                # Blocks until the hang guard shuts the socket down
                assert self._guarded_socket().recv(1) == b""
            self._fail()

    def _guarded_socket(self) -> socket.socket:
        assert self._socket is not None, "a stuck connection needs an attempt timeout"
        return self._socket


class FakeAsyncConnection(FakeConnection):
    """FakeConnection for the async saver."""

    def cursor(self, **kwargs: Any) -> FakeAsyncCursor:
        return FakeAsyncCursor(self)

    async def wait(self, gen: Generator[Any, Any, T]) -> T:  # type: ignore[override]
        return FakeConnection.wait(self, gen)

    async def rollback(self) -> None:
        self._roll_back(asyncio.CancelledError)

    async def arun(self, query: Any) -> None:
        fails = self._record(query)
        if self.on_statement is not None:
            await self.on_statement(str(query))
        if fails:
            if self.stuck:
                sock = self._guarded_socket()
                sock.setblocking(False)
                assert await asyncio.get_running_loop().sock_recv(sock, 1) == b""
            self._fail()


class FakeSingleConnection(FakeConnection, psycopg.Connection):
    """FakeConnection that the sync saver takes for its only connection."""


class FakeSingleAsyncConnection(FakeAsyncConnection, psycopg.AsyncConnection):
    """FakeAsyncConnection that the async saver takes for its only connection."""


class FakeCursor:
    """Runs statements on a fake connection. Reads find an empty database.

    ``statusmessage`` is the command tag of the last statement it ran.
    """

    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn
        self._last: str | None = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    @property
    def statusmessage(self) -> str | None:
        if self._last is None:
            return None
        if self._last == "COMMIT":
            return self._conn.commit_answer
        return self._last.split()[0].upper()

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> FakeCursor:
        self._last = str(query).strip()
        self._conn.run(query)
        return self

    def executemany(self, query: Any, params_seq: Any, **kwargs: Any) -> None:
        self._last = str(query).strip()
        self._conn.run(query)

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class FakeAsyncCursor(FakeCursor):
    """FakeCursor for the async saver."""

    def __init__(self, conn: FakeAsyncConnection) -> None:
        super().__init__(conn)
        self._aconn = conn

    async def __aenter__(self) -> FakeAsyncCursor:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def execute(self, query: Any, params: Any = None, **kwargs: Any) -> FakeAsyncCursor:
        self._last = str(query).strip()
        await self._aconn.arun(query)
        return self

    async def executemany(self, query: Any, params_seq: Any, **kwargs: Any) -> None:
        self._last = str(query).strip()
        await self._aconn.arun(query)

    async def fetchone(self) -> None:
        return None

    async def fetchall(self) -> list[Any]:
        return []


class FakePool(ConnectionPool):
    """A pool that hands out the given connections in order, then ones from ``factory``."""

    def __init__(
        self, *connections: FakeConnection, factory: Callable[[], FakeConnection] = FakeConnection
    ) -> None:
        super().__init__("", open=False)
        self.scripted = list(connections)
        self.factory = factory
        self.handed_out: list[FakeConnection] = []

    @contextlib.contextmanager
    def connection(self, timeout: float | None = None) -> Iterator[Any]:
        conn = self.scripted.pop(0) if self.scripted else self.factory()
        self.handed_out.append(conn)
        yield conn


class FakeAsyncPool(AsyncConnectionPool):
    """FakePool for the async saver."""

    def __init__(
        self,
        *connections: FakeAsyncConnection,
        factory: Callable[[], FakeAsyncConnection] = FakeAsyncConnection,
    ) -> None:
        super().__init__("", open=False)
        self.scripted = list(connections)
        self.factory = factory
        self.handed_out: list[FakeAsyncConnection] = []

    @contextlib.asynccontextmanager
    async def connection(self, timeout: float | None = None) -> AsyncIterator[Any]:
        conn = self.scripted.pop(0) if self.scripted else self.factory()
        self.handed_out.append(conn)
        yield conn


class FakePipe:
    """The saver-wide pipeline of a saver made with ``pipeline=True``.

    Counts syncs, adds them to ``events`` when given, and calls ``on_sync``.
    """

    def __init__(
        self,
        events: list[tuple[str, ...]] | None = None,
        on_sync: Callable[[], None] | None = None,
    ) -> None:
        self.events = events
        self.on_sync = on_sync
        self.syncs = 0

    def sync(self) -> None:
        self.syncs += 1
        if self.events is not None:
            self.events.append(("sync",))
        if self.on_sync is not None:
            self.on_sync()


class FakeAsyncPipe:
    """FakePipe for the async saver."""

    def __init__(
        self,
        events: list[tuple[str, ...]] | None = None,
        on_sync: Callable[[], None] | None = None,
    ) -> None:
        self._pipe = FakePipe(events, on_sync)

    @property
    def syncs(self) -> int:
        return self._pipe.syncs

    async def sync(self) -> None:
        self._pipe.sync()


def connection(kind: str, error: BaseException | None = None, **kwargs: Any) -> FakeConnection:
    cls = FakeConnection if kind == "sync" else FakeAsyncConnection
    return cls(error, **kwargs)


def single_connection(kind: str, error: BaseException, **kwargs: Any) -> FakeConnection:
    cls = FakeSingleConnection if kind == "sync" else FakeSingleAsyncConnection
    return cls(error, **kwargs)


def pool(kind: str, *connections: Any) -> Any:
    return FakePool(*connections) if kind == "sync" else FakeAsyncPool(*connections)
