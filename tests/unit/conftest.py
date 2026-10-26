"""Setup shared by the unit tests."""

from __future__ import annotations

import pytest
from psycopg import generators


@pytest.fixture(autouse=True)
def _python_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drain the fake connections with psycopg's Python send().

    Before each sync of a component-wide pipeline, the checkpointers and the
    store drain libpq's output buffer with psycopg's send(). Its C version
    only takes a real libpq connection, and the fakes have none. The Python
    version keeps the same contract: flush until nothing is left, waiting
    for the socket in between.
    """
    monkeypatch.setattr("langchain_cockroachdb._psycopg_ops.send", generators._send)
