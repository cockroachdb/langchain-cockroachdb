"""How the checkpointers retry, on fake psycopg connections and pools.

The fakes stand in for what libpq does when it loses a connection: the call
fails, and psycopg reports the connection as broken. Over TLS, OpenSSL 3 words
that failure "unexpected eof while reading", which CockroachDB Cloud showed
after the hang guard shut the socket of a stuck write.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import AsyncIterator, Iterator
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import errors as pg_errors
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver, RetryPolicy

TLS_EOF = "consuming input failed: SSL error: unexpected eof while reading"
SERVER_CLOSED = "consuming input failed: server closed the connection unexpectedly"
UNKNOWN = "lost contact with the server in a way no phrase list knows"
FAST = {"initial_backoff": 0.01, "max_backoff": 0.05}
CONFIG = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
KINDS = pytest.mark.parametrize("kind", ["sync", "async"])

_sockets: list[socket.socket] = []


@pytest.fixture(autouse=True)
def _close_sockets() -> Iterator[None]:
    yield
    while _sockets:
        _sockets.pop().close()


class FakeConnection:
    """A psycopg connection with only the parts the savers use.

    The first statement fails with ``error`` when one is given. With ``stuck``
    it first waits until the hang guard shuts down the connection's socket.
    With ``breaks``, psycopg's broken flag is set as the error is raised, the
    way libpq marks a connection it lost.
    """

    def __init__(
        self, error: BaseException | None = None, *, breaks: bool = False, stuck: bool = False
    ) -> None:
        self.error = error
        self.breaks = breaks
        self.stuck = stuck
        self.statements: list[str] = []
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

    def pipeline(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()

    def transaction(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()

    def cursor(self, **kwargs: Any) -> FakeCursor:
        return FakeCursor(self)

    def _record(self, query: Any) -> bool:
        """Record a statement; True if it is the one that fails."""
        self.statements.append(str(query))
        return self.error is not None and len(self.statements) == 1

    def _fail(self) -> None:
        self._broken = self.breaks
        assert self.error is not None
        raise self.error

    def run(self, query: Any) -> None:
        if self._record(query):
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
        if self._record(query):
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
    """A pool that hands out the given connections in order, then healthy ones."""

    def __init__(self, *connections: FakeConnection) -> None:
        super().__init__("", open=False)
        self.scripted = list(connections)
        self.handed_out: list[FakeConnection] = []

    @contextlib.contextmanager
    def connection(self, timeout: float | None = None) -> Iterator[Any]:
        conn = self.scripted.pop(0) if self.scripted else FakeConnection()
        self.handed_out.append(conn)
        yield conn


class FakeAsyncPool(AsyncConnectionPool):
    """FakePool for the async saver."""

    def __init__(self, *connections: FakeAsyncConnection) -> None:
        super().__init__("", open=False)
        self.scripted = list(connections)
        self.handed_out: list[FakeAsyncConnection] = []

    @contextlib.asynccontextmanager
    async def connection(self, timeout: float | None = None) -> AsyncIterator[Any]:
        conn = self.scripted.pop(0) if self.scripted else FakeAsyncConnection()
        self.handed_out.append(conn)
        yield conn


def connection(kind: str, error: BaseException | None = None, **kwargs: Any) -> FakeConnection:
    cls = FakeConnection if kind == "sync" else FakeAsyncConnection
    return cls(error, **kwargs)


def single_connection(kind: str, error: BaseException, **kwargs: Any) -> FakeConnection:
    cls = FakeSingleConnection if kind == "sync" else FakeSingleAsyncConnection
    return cls(error, **kwargs)


def pool(kind: str, *connections: Any) -> Any:
    return FakePool(*connections) if kind == "sync" else FakeAsyncPool(*connections)


async def put(kind: str, conn: Any, policy: RetryPolicy) -> dict[str, Any]:
    """Save an empty checkpoint, which takes one statement per attempt."""
    if kind == "sync":
        return CockroachDBSaver(conn, retry_policy=policy).put(CONFIG, empty_checkpoint(), {}, {})
    saver = AsyncCockroachDBSaver(conn, retry_policy=policy)
    return await saver.aput(CONFIG, empty_checkpoint(), {}, {})


async def setup(kind: str, conn: Any, policy: RetryPolicy) -> None:
    if kind == "sync":
        CockroachDBSaver(conn, retry_policy=policy).setup()
    else:
        await AsyncCockroachDBSaver(conn, retry_policy=policy).setup()


def server_error(code: str, message: str) -> psycopg.Error:
    return pg_errors.lookup(code)(message)


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


@pytest.mark.timeout(10)
@KINDS
@pytest.mark.parametrize("phrases", ["default", "none"])
async def test_pooled_saver_retries_a_write_whose_tls_connection_the_guard_shut(
    kind: str, phrases: str, caplog: pytest.LogCaptureFixture
) -> None:
    # What CockroachDB Cloud showed: the hang guard abandoned a stuck write, and
    # the TLS connection failed with OpenSSL 3's wording. With no transient
    # phrases at all, only psycopg's broken flag can make the error retryable.
    policy = RetryPolicy(attempt_timeout=0.2, **FAST)
    if phrases == "none":
        policy = RetryPolicy(attempt_timeout=0.2, transient_messages=(), **FAST)
    stuck = connection(kind, psycopg.OperationalError(TLS_EOF), breaks=True, stuck=True)
    connections = pool(kind, stuck)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        saved = await put(kind, connections, policy)
    assert saved["configurable"]["thread_id"] == "thread-1"
    assert len(connections.handed_out) == 2
    assert "INSERT INTO checkpoints" in connections.handed_out[1].statements[0]
    logged = messages(caplog)
    assert any("put checkpoint ran longer than its 0.2s attempt timeout" in m for m in logged)
    assert any(
        m.startswith(f"Retrying put checkpoint after OperationalError: {TLS_EOF}") for m in logged
    )


@KINDS
async def test_pooled_saver_retries_any_driver_error_after_its_connection_broke(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    broke = connection(kind, psycopg.OperationalError(UNKNOWN), breaks=True)
    connections = pool(kind, broke)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        await put(kind, connections, RetryPolicy(**FAST))
    assert len(connections.handed_out) == 2
    assert any(
        m.startswith(f"Retrying put checkpoint after OperationalError: {UNKNOWN}")
        for m in messages(caplog)
    )


@KINDS
async def test_pooled_saver_does_not_retry_an_unknown_error_on_a_healthy_connection(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    healthy = connection(kind, psycopg.OperationalError(UNKNOWN))
    connections = pool(kind, healthy)
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(psycopg.OperationalError, match="no phrase"),
    ):
        await put(kind, connections, RetryPolicy(**FAST))
    assert len(connections.handed_out) == 1
    assert not any("Retrying" in m for m in messages(caplog))


@KINDS
@pytest.mark.parametrize(
    "make_error",
    [
        # What CockroachDB raises when CANCEL SESSION ends a running statement
        lambda: server_error("57014", "query execution canceled"),
        lambda: psycopg.OperationalError("SSL error: certificate verify failed"),
    ],
    ids=["definite_sqlstate", "configuration_error"],
)
async def test_broken_connection_does_not_overrule_a_final_error(
    kind: str, make_error: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Like SQLAlchemy's connection_invalidated: a usable SQLSTATE and the
    # non-transient phrases still decide first
    error = make_error()
    connections = pool(kind, connection(kind, error, breaks=True))
    with caplog.at_level(logging.WARNING), pytest.raises(type(error)):
        await put(kind, connections, RetryPolicy(**FAST))
    assert len(connections.handed_out) == 1
    assert not any("Retrying" in m for m in messages(caplog))


@KINDS
@pytest.mark.parametrize("wording", [TLS_EOF, UNKNOWN], ids=["tls_eof", "unknown"])
async def test_single_connection_saver_does_not_retry_a_broken_connection(
    kind: str, wording: str, caplog: pytest.LogCaptureFixture
) -> None:
    only = single_connection(kind, psycopg.OperationalError(wording), breaks=True)
    with caplog.at_level(logging.WARNING), pytest.raises(psycopg.OperationalError):
        await put(kind, only, RetryPolicy(**FAST))
    assert len(only.statements) == 1
    logged = messages(caplog)
    assert not any("Retrying" in m for m in logged)
    not_retrying = [m for m in logged if m.startswith("Not retrying")]
    if wording == TLS_EOF:
        # A known lost connection gets the advice to use a pool
        assert len(not_retrying) == 1
        assert "the checkpointer's only connection is closed" in not_retrying[0]
    else:
        # The broken flag only counts for a pool, so this error is not retryable
        assert not_retrying == []


@pytest.mark.timeout(10)
@KINDS
async def test_setup_labels_the_bootstrap_apart_from_migration_0(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The bootstrap (create the migrations table, read the version) gets stuck
    # once, then migration 0 fails once
    stuck = connection(kind, psycopg.OperationalError(SERVER_CLOSED), breaks=True, stuck=True)
    bootstrap = connection(kind)
    migration_0 = connection(kind, server_error("40001", "restart transaction"))
    connections = pool(kind, stuck, bootstrap, migration_0)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
        await setup(kind, connections, RetryPolicy(ddl_attempt_timeout=0.2, **FAST))
    assert "SELECT v FROM checkpoint_migrations" in bootstrap.statements[1]
    assert migration_0.statements == [CockroachDBSaver.MIGRATIONS[0]]
    logged = messages(caplog)
    assert any("checkpointer setup ran longer than its 0.2s attempt timeout" in m for m in logged)
    assert any(
        m.startswith(f"Retrying checkpointer setup after OperationalError: {SERVER_CLOSED}")
        for m in logged
    )
    migration_0_retries = [m for m in logged if m.startswith("Retrying checkpointer migration 0")]
    assert len(migration_0_retries) == 1
    assert "SerializationFailure (SQLSTATE 40001)" in migration_0_retries[0]
    assert not any("checkpointer migration 0 ran longer" in m for m in logged)
