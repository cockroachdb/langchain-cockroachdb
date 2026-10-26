"""Real CockroachDB faults for integration tests.

The errors come from the server, so psycopg, SQLAlchemy and the retry policy
see exactly what they would see in production:

- crdb_internal.force_error raises any SQLSTATE we ask for
- CANCEL SESSIONS kills pooled connections, like a node restart or a load
  balancer reset
- an open transaction holding a row's write intent makes writers wait, the way
  a stuck range does

Interruption is a client-side fault: it stops a write between statements,
the way a cancelled task or a crashed caller does. SlowProxy is the other
one: it slows the link from the client to the server, the way a congested
network or a TLS link to a distant cluster does.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

FORCE_ERROR = "SELECT crdb_internal.force_error(%(fault_code)s, %(fault_message)s)"
FORCE_RETRY = "SELECT crdb_internal.force_retry(CAST(%(interval)s AS INTERVAL))"


def supports_unsafe_internals(libpq_url: str) -> bool:
    """True if the server has the allow_unsafe_internals setting (v26 and later)."""
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        try:
            conn.execute("SET allow_unsafe_internals = true")
        except psycopg.Error:
            return False
    return True


def _fault_params(code: str) -> dict[str, str]:
    return {"fault_code": code, "fault_message": f"injected {code} by the test suite"}


class _Faults:
    """Pending faults and every statement seen, shared by both injectors."""

    def __init__(self, unsafe_internals: bool) -> None:
        self.unsafe_internals = unsafe_internals
        self._lock = threading.Lock()
        self._pending: list[tuple[str, str | None]] = []
        self._retries: list[tuple[str, str]] = []
        self.statements: list[str] = []
        self.injected: list[str] = []

    def server_retry_next(self, match: str, *, interval: str = "500ms") -> None:
        """Make CockroachDB retry the transaction of the next statement containing ``match``.

        crdb_internal.force_retry runs just before that statement, in the same
        batch, and fails the transaction with retry errors until it is
        ``interval`` old. When CockroachDB can retry the batch itself, the
        client sees no error and the statement takes at least ``interval``.
        Only PsycopgFaults injects these.
        """
        with self._lock:
            self._retries.append((match.lower(), interval))

    def _take_retry(self, statement: str) -> str | None:
        lowered = statement.lower()
        with self._lock:
            for index, (match, interval) in enumerate(self._retries):
                if match in lowered:
                    del self._retries[index]
                    return interval
        return None

    def fail_next(self, sqlstate: str, *, times: int = 1, match: str | None = None) -> None:
        """Make the next ``times`` statements containing ``match`` fail with ``sqlstate``."""
        with self._lock:
            self._pending.extend([(sqlstate, match.lower() if match else None)] * times)

    def count(self, match: str) -> int:
        """Statements sent that contain ``match`` (case-insensitive), failed ones included."""
        needle = match.lower()
        with self._lock:
            return sum(needle in statement for statement in self.statements)

    def _take(self, statement: str) -> str | None:
        lowered = statement.lower()
        with self._lock:
            self.statements.append(lowered)
            if lowered.lstrip().startswith(("set ", "show ")):
                return None
            for index, (code, match) in enumerate(self._pending):
                if match is None or match in lowered:
                    del self._pending[index]
                    self.injected.append(code)
                    return code
        return None


class FaultInjector(_Faults):
    """Fault injection for a SQLAlchemy async engine (vector store, chat history).

    The faulted statement is swapped for crdb_internal.force_error, so the server
    raises the error during normal execution and SQLAlchemy wraps it.
    """

    def __init__(self, engine: AsyncEngine, *, unsafe_internals: bool) -> None:
        super().__init__(unsafe_internals)
        self._sync_engine = engine.sync_engine
        event.listen(self._sync_engine, "before_cursor_execute", self._before_execute, retval=True)

    def close(self) -> None:
        event.remove(self._sync_engine, "before_cursor_execute", self._before_execute)

    def _before_execute(self, conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        code = None if executemany else self._take(statement)
        if code is None:
            return statement, parameters
        if self.unsafe_internals:
            cursor.execute("SET allow_unsafe_internals = true")
        return FORCE_ERROR, _fault_params(code)


class PsycopgFaults(_Faults):
    """Fault injection for raw psycopg connections (checkpointers).

    Pass ``cursor_factory=faults.cursor_class()`` (or the async variant) to the
    connection or pool so every cursor checks for a pending fault, or a
    pending server retry, first.
    """

    def cursor_class(self) -> type[psycopg.Cursor[Any]]:
        faults = self

        class FaultyCursor(psycopg.Cursor):  # type: ignore[type-arg]
            def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                faults._retry_if_due(self, query)
                faults._raise_if_due(self, query)
                return super().execute(query, params, **kwargs)

            def executemany(self, query, params_seq, **kwargs):  # type: ignore[no-untyped-def,override]
                faults._retry_if_due(self, query)
                faults._raise_if_due(self, query)
                return super().executemany(query, params_seq, **kwargs)

        return FaultyCursor

    def async_cursor_class(self) -> type[psycopg.AsyncCursor[Any]]:
        faults = self

        class FaultyAsyncCursor(psycopg.AsyncCursor):  # type: ignore[type-arg]
            async def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                await faults._aretry_if_due(self, query)
                await faults._araise_if_due(self, query)
                return await super().execute(query, params, **kwargs)

            async def executemany(self, query, params_seq, **kwargs):  # type: ignore[no-untyped-def,override]
                await faults._aretry_if_due(self, query)
                await faults._araise_if_due(self, query)
                return await super().executemany(query, params_seq, **kwargs)

        return FaultyAsyncCursor

    def _retry_if_due(self, cursor: psycopg.Cursor[Any], query: Any) -> None:
        interval = self._take_retry(query if isinstance(query, str) else str(query))
        if interval is None:
            return
        if self.unsafe_internals:
            psycopg.Cursor.execute(cursor, "SET allow_unsafe_internals = true")
        psycopg.Cursor.execute(cursor, FORCE_RETRY, {"interval": interval})

    async def _aretry_if_due(self, cursor: psycopg.AsyncCursor[Any], query: Any) -> None:
        interval = self._take_retry(query if isinstance(query, str) else str(query))
        if interval is None:
            return
        if self.unsafe_internals:
            await psycopg.AsyncCursor.execute(cursor, "SET allow_unsafe_internals = true")
        await psycopg.AsyncCursor.execute(cursor, FORCE_RETRY, {"interval": interval})

    def _raise_if_due(self, cursor: psycopg.Cursor[Any], query: Any) -> None:
        code = self._take(query if isinstance(query, str) else str(query))
        if code is None:
            return
        if self.unsafe_internals:
            psycopg.Cursor.execute(cursor, "SET allow_unsafe_internals = true")
        psycopg.Cursor.execute(cursor, FORCE_ERROR, _fault_params(code))

    async def _araise_if_due(self, cursor: psycopg.AsyncCursor[Any], query: Any) -> None:
        code = self._take(query if isinstance(query, str) else str(query))
        if code is None:
            return
        if self.unsafe_internals:
            await psycopg.AsyncCursor.execute(cursor, "SET allow_unsafe_internals = true")
        await psycopg.AsyncCursor.execute(cursor, FORCE_ERROR, _fault_params(code))


class Interruption:
    """Stops the client partway through a write, after earlier statements were queued.

    Once armed, the next statement containing ``match`` (case-insensitive) is
    interrupted; executemany sends its first row first. Pass
    ``cursor_factory=interruption.cursor_class()`` (or the async variant) to the
    connection or pool. The sync cursor raises RuntimeError("interrupted"). The
    async cursor sets ``reached`` and waits until the test cancels its task.
    """

    def __init__(self, match: str) -> None:
        self.match = match.lower()
        self.armed = False
        self.reached: asyncio.Event | None = None

    def _due(self, query: Any) -> bool:
        if self.armed and self.match in str(query).lower():
            self.armed = False
            return True
        return False

    async def _wait_for_cancel(self) -> None:
        assert self.reached is not None, "set reached to an asyncio.Event first"
        self.reached.set()
        await asyncio.Event().wait()

    def cursor_class(self) -> type[psycopg.Cursor[Any]]:
        interruption = self

        class InterruptingCursor(psycopg.Cursor):  # type: ignore[type-arg]
            def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                if interruption._due(query):
                    raise RuntimeError("interrupted")
                return super().execute(query, params, **kwargs)

            def executemany(self, query, params_seq, **kwargs):  # type: ignore[no-untyped-def,override]
                if interruption._due(query):
                    super().executemany(query, list(params_seq)[:1], **kwargs)
                    raise RuntimeError("interrupted")
                return super().executemany(query, params_seq, **kwargs)

        return InterruptingCursor

    def async_cursor_class(self) -> type[psycopg.AsyncCursor[Any]]:
        interruption = self

        class InterruptingAsyncCursor(psycopg.AsyncCursor):  # type: ignore[type-arg]
            async def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                if interruption._due(query):
                    await interruption._wait_for_cancel()
                return await super().execute(query, params, **kwargs)

            async def executemany(self, query, params_seq, **kwargs):  # type: ignore[no-untyped-def,override]
                if interruption._due(query):
                    await super().executemany(query, list(params_seq)[:1], **kwargs)
                    await interruption._wait_for_cancel()
                return await super().executemany(query, params_seq, **kwargs)

        return InterruptingAsyncCursor


class SlowProxy:
    """A TCP proxy that passes the client's bytes to the server slowly.

    It listens on 127.0.0.1 on a free port and forwards each connection to
    the server. From client to server it moves at most ``chunk`` bytes at a
    time and pauses ``pause`` seconds after each, behind a small receive
    buffer. From server to client it runs at full speed. Once the client's
    socket send buffer is full, libpq's flush() returns 1 with bytes still
    to send; limit_send_buffer() keeps that buffer small. Connect with
    ``connect_kwargs``: libpq reaches the proxy but still checks TLS against
    the host name in the URL. Use it as a context manager; on exit it cuts
    every connection it forwards.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        chunk: int = 4096,
        pause: float = 0.002,
        receive_buffer: int = 16384,
    ) -> None:
        self._upstream = (host, port)
        self._chunk = chunk
        self._pause = pause
        self._lock = threading.Lock()
        self._closed = False
        self._sockets: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self.connections = 0
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set before listen(), so that accepted sockets advertise a small window
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.05)
        self._listener = listener
        self.port: int = listener.getsockname()[1]
        self._start(self._accept)

    @classmethod
    def to(cls, libpq_url: str, **kwargs: Any) -> SlowProxy:
        """A proxy in front of the server that ``libpq_url`` names."""
        params = conninfo_to_dict(libpq_url)
        host = params.get("hostaddr") or params.get("host") or "localhost"
        return cls(str(host), int(params.get("port") or 26257), **kwargs)

    @property
    def connect_kwargs(self) -> dict[str, Any]:
        """psycopg connection options that send a connection through the proxy."""
        return {"hostaddr": "127.0.0.1", "port": self.port}

    def _start(self, target: Callable[..., None], *args: Any) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                client, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                server = socket.create_connection(self._upstream, timeout=10)
            except OSError:
                client.close()
                continue
            server.settimeout(None)
            for sock in (client, server):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                if self._closed:
                    client.close()
                    server.close()
                    return
                self._sockets += [client, server]
                self.connections += 1
            self._start(self._pump, client, server, self._chunk, self._pause)
            self._start(self._pump, server, client, 65536, 0.0)

    @staticmethod
    def _pump(source: socket.socket, target: socket.socket, chunk: int, pause: float) -> None:
        with contextlib.suppress(OSError):
            while data := source.recv(chunk):
                target.sendall(data)
                if pause:
                    time.sleep(pause)
        # Pass the end of the stream on, so each side sees the other close
        with contextlib.suppress(OSError):
            target.shutdown(socket.SHUT_WR)

    def close(self) -> None:
        """Stop accepting and cut every forwarded connection."""
        with self._lock:
            self._closed = True
            sockets, self._sockets = self._sockets, []
            threads = list(self._threads)
        self._listener.close()
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        for thread in threads:
            thread.join(timeout=5)
        for sock in sockets:
            sock.close()

    def __enter__(self) -> SlowProxy:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def limit_send_buffer(
    conn: psycopg.Connection[Any] | psycopg.AsyncConnection[Any], size: int
) -> None:
    """Give the socket of ``conn`` a fixed send buffer of ``size`` bytes.

    The operating system grows a socket's send buffer by itself, to several
    megabytes, so a local test would need a huge write before libpq is left
    holding bytes it could not send. A fixed size stops that growth.
    """
    sock = socket.socket(fileno=os.dup(conn.fileno()))
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, size)
    finally:
        sock.close()


def kill_sessions(libpq_url: str, application_name: str) -> int:
    """Cancel every session opened with ``application_name``; returns how many there were."""
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        rows = admin.execute(
            "SELECT session_id FROM [SHOW CLUSTER SESSIONS] WHERE application_name = %s",
            (application_name,),
        ).fetchall()
        if rows:
            admin.execute(
                sql.SQL(
                    "CANCEL SESSIONS (SELECT session_id FROM [SHOW CLUSTER SESSIONS] "
                    "WHERE application_name = {})"
                ).format(sql.Literal(application_name))
            )
    return len(rows)


def wait_for_statement(
    libpq_url: str, application_name: str, match: str, *, timeout: float = 10.0
) -> None:
    """Wait until a session opened with ``application_name`` runs a statement containing ``match``.

    The match is case-insensitive. Raises AssertionError after ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    with psycopg.connect(libpq_url, autocommit=True) as admin:
        while True:
            row = admin.execute(
                "SELECT count(*) FROM [SHOW CLUSTER STATEMENTS] "
                "WHERE application_name = %s AND query ILIKE %s",
                (application_name, f"%{match}%"),
            ).fetchone()
            if row is not None and row[0] > 0:
                return
            if time.monotonic() > deadline:
                raise AssertionError(f"no {application_name} statement matching {match!r} ran")
            time.sleep(0.05)


@contextlib.contextmanager
def hold_row_lock(
    libpq_url: str, update_sql: str, params: tuple[Any, ...], *, release_after: float | None = None
) -> Iterator[None]:
    """Keep ``update_sql`` uncommitted so writes to its row wait.

    With ``release_after`` the transaction is rolled back from a timer thread
    while the caller is still waiting, which ends the outage.
    """
    conn = psycopg.connect(libpq_url)
    conn.execute(update_sql, params)
    lock = threading.Lock()
    released = False

    def release() -> None:
        nonlocal released
        with lock:
            if not released:
                released = True
                conn.rollback()

    timer = threading.Timer(release_after, release) if release_after is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        if timer is not None:
            timer.cancel()
        release()
        conn.close()
