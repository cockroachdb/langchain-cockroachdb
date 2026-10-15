"""Fake psycopg connections, cursors and pools for the checkpointer unit tests.

The fakes stand in for what libpq does when it loses a connection: the call
fails, and psycopg reports the connection as broken. Over TLS, OpenSSL 3 words
that failure "unexpected eof while reading", which CockroachDB Cloud showed
after the hang guard shut the socket of a stuck write.

Test modules that use the fakes call ``close_sockets()`` after each test.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import psycopg
from psycopg_pool import AsyncConnectionPool, ConnectionPool

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


class FakeConnection:
    """A psycopg connection with only the parts the savers use.

    Statement number ``fail_at`` (the first by default) fails with ``error``
    when one is given. With ``stuck`` it first waits until the hang guard
    shuts down the connection's socket. With ``breaks``, psycopg's broken flag
    is set as the error is raised, the way libpq marks a connection it lost.
    ``on_statement`` is called with each statement as it runs (awaited for the
    async fakes), so a test can hold an operation inside its connection.
    ``events`` records statements and the pipeline and transaction blocks
    around them, in order.
    """

    def __init__(
        self,
        error: BaseException | None = None,
        *,
        fail_at: int = 1,
        breaks: bool = False,
        stuck: bool = False,
        on_statement: Callable[[str], Any] | None = None,
    ) -> None:
        self.error = error
        self.fail_at = fail_at
        self.breaks = breaks
        self.stuck = stuck
        self.on_statement = on_statement
        self.statements: list[str] = []
        self.events: list[tuple[str, ...]] = []
        self._broken = False
        self._socket: socket.socket | None = None

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
        return self.error is not None and len(self.statements) == self.fail_at

    def _fail(self) -> None:
        self._broken = self.breaks
        assert self.error is not None
        raise self.error

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
    """Runs statements on a fake connection. Reads find an empty database."""

    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> FakeCursor:
        self._conn.run(query)
        return self

    def executemany(self, query: Any, params_seq: Any, **kwargs: Any) -> None:
        self._conn.run(query)

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class FakeAsyncCursor:
    """FakeCursor for the async saver."""

    def __init__(self, conn: FakeAsyncConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeAsyncCursor:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def execute(self, query: Any, params: Any = None, **kwargs: Any) -> FakeAsyncCursor:
        await self._conn.arun(query)
        return self

    async def executemany(self, query: Any, params_seq: Any, **kwargs: Any) -> None:
        await self._conn.arun(query)

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
