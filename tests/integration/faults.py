"""Real CockroachDB faults for integration tests.

The errors come from the server, so psycopg, SQLAlchemy and the retry policy
see exactly what they would see in production:

- crdb_internal.force_error raises any SQLSTATE we ask for
- CANCEL SESSIONS kills pooled connections, like a node restart or a load
  balancer reset
- an open transaction holding a row's write intent makes writers wait, the way
  a stuck range does
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from typing import Any

import psycopg
from psycopg import sql
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

FORCE_ERROR = "SELECT crdb_internal.force_error(%(fault_code)s, %(fault_message)s)"


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
        self.statements: list[str] = []
        self.injected: list[str] = []

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
    connection or pool so every cursor checks for a pending fault first.
    """

    def cursor_class(self) -> type[psycopg.Cursor[Any]]:
        faults = self

        class FaultyCursor(psycopg.Cursor):  # type: ignore[type-arg]
            def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                faults._raise_if_due(self, query)
                return super().execute(query, params, **kwargs)

            def executemany(self, query, params_seq, **kwargs):  # type: ignore[no-untyped-def,override]
                faults._raise_if_due(self, query)
                return super().executemany(query, params_seq, **kwargs)

        return FaultyCursor

    def async_cursor_class(self) -> type[psycopg.AsyncCursor[Any]]:
        faults = self

        class FaultyAsyncCursor(psycopg.AsyncCursor):  # type: ignore[type-arg]
            async def execute(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def,override]
                await faults._araise_if_due(self, query)
                return await super().execute(query, params, **kwargs)

            async def executemany(self, query, params_seq, **kwargs):  # type: ignore[no-untyped-def,override]
                await faults._araise_if_due(self, query)
                return await super().executemany(query, params_seq, **kwargs)

        return FaultyAsyncCursor

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
