"""Setup shared by the unit tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from psycopg import generators
from psycopg.waiting import Ready, Wait

# psycopg's own send(), before the fixture below replaces it in each test
PSYCOPG_SEND = getattr(generators, "send", None)


def _send_loop(pgconn: Any) -> Generator[Wait, Ready | None, None]:
    """A copy of psycopg's Python send(), for a psycopg without generators._send."""
    while pgconn.flush() != 0:
        while not (ready := (yield Wait.RW)):
            continue
        if ready & Ready.R:
            pgconn.consume_input()


@pytest.fixture(autouse=True)
def _python_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drain the fake connections with psycopg's Python send().

    As each operation on a component-wide pipeline starts, and before each
    sync of it, the checkpointers and the store drain libpq's output buffer
    with psycopg's send(), which they import from psycopg.generators when
    they need it. Its C version only takes a real libpq connection, and the
    fakes have none. The Python version keeps the same contract: flush until
    nothing is left, waiting for the socket in between.
    """
    python_send = getattr(generators, "_send", None) or _send_loop
    monkeypatch.setattr(generators, "send", python_send, raising=False)


@pytest.fixture
def psycopg_send() -> object:
    """psycopg's own send(), or None if psycopg no longer has one."""
    return PSYCOPG_SEND
