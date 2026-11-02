"""A graph survives connection replacement or resumes after saver retries run out.

The cursor closes only its own checked-out connection before a selected SQL
statement. Psycopg raises the real error, and the pool must replace the closed
connection. This is a client-side connection fault, not a network partition or
a server failover. No server fault-injection privileges are needed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypedDict
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool, ConnectionPool

pytest.importorskip("langgraph.graph")

from langgraph.graph import END, START, StateGraph  # noqa: E402

from langchain_cockroachdb import (  # noqa: E402
    AsyncCockroachDBSaver,
    CockroachDBSaver,
    RetryPolicy,
)
from tests.integration.graph_recovery_worker import POOL_OPTIONS, config  # noqa: E402

ATTEMPTS = 3
POLICY = RetryPolicy(max_attempts=ATTEMPTS, initial_backoff=0, max_backoff=0, jitter=0)


class CheckpointDisconnects:
    """Close the next N checkpoint INSERT connections at a deterministic boundary."""

    def __init__(self) -> None:
        self.remaining = 0
        self.closed_connections: list[Any] = []
        self._lock = threading.Lock()

    def arm(self, count: int) -> None:
        with self._lock:
            self.remaining = count

    def due(self, query: Any, connection: Any) -> bool:
        if "insert into checkpoints (" not in str(query).lower():
            return False
        with self._lock:
            if not self.remaining:
                return False
            self.remaining -= 1
            self.closed_connections.append(connection)
            return True

    def cursor_class(self) -> type[psycopg.Cursor]:
        disconnects = self

        class DisconnectingCursor(psycopg.Cursor):
            def execute(self, query, params=None, **kwargs):
                if disconnects.due(query, self.connection):
                    self.connection.close()
                return super().execute(query, params, **kwargs)

        return DisconnectingCursor

    def async_cursor_class(self) -> type[psycopg.AsyncCursor]:
        disconnects = self

        class DisconnectingAsyncCursor(psycopg.AsyncCursor):
            async def execute(self, query, params=None, **kwargs):
                if disconnects.due(query, self.connection):
                    await self.connection.close()
                return await super().execute(query, params, **kwargs)

        return DisconnectingAsyncCursor


class RecoveryState(TypedDict, total=False):
    agent: str
    prepared: str
    finished: str


def recovery_graph(saver: Any, arm: Callable[[], None], executions: list[str]) -> Any:
    def prepare(state: RecoveryState) -> dict:
        executions.append("prepare")
        # durability='sync' makes the input checkpoint durable before this
        # node starts. Its next checkpoint INSERT must encounter the fault.
        arm()
        return {"prepared": f"prepared:{state['agent']}"}

    def finish(state: RecoveryState) -> dict:
        executions.append("finish")
        return {"finished": f"finished:{state['prepared']}"}

    graph = StateGraph(RecoveryState)
    graph.add_node("prepare", prepare)
    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=saver)


def assert_exhausted(saved: Any, agent: str, executions: list[str]) -> None:
    assert saved is not None
    assert saved.checkpoint["channel_values"]["agent"] == agent
    assert "prepared" not in saved.checkpoint["channel_values"]
    assert [value for _, channel, value in saved.pending_writes if channel == "prepared"] == [
        f"prepared:{agent}"
    ]
    assert executions == ["prepare"]


def assert_recovered(
    result: dict,
    agent: str,
    executions: list[str],
    disconnects: CheckpointDisconnects,
    failures: int,
) -> None:
    assert result == {
        "agent": agent,
        "prepared": f"prepared:{agent}",
        "finished": f"finished:prepared:{agent}",
    }
    # The completed node's pending writes survive retry exhaustion, so resume
    # must not run its external effect again. Effects whose writes did not
    # commit can still repeat; this is not an exactly-once guarantee.
    assert executions == ["prepare", "finish"]
    assert disconnects.remaining == 0
    assert len(disconnects.closed_connections) == failures
    assert all(connection.closed for connection in disconnects.closed_connections)
    assert len({id(connection) for connection in disconnects.closed_connections}) == failures


def assert_exhaustion_log(caplog: pytest.LogCaptureFixture) -> None:
    messages = [record.getMessage() for record in caplog.records]
    assert sum(message.startswith("Retrying put checkpoint after") for message in messages) == 2
    assert (
        sum(
            message.startswith("Giving up on put checkpoint after 3 attempts:")
            for message in messages
        )
        == 1
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("failures", [1, ATTEMPTS], ids=["retry", "exhaust_and_resume"])
def test_sync_graph_recovers_after_checkpoint_connection_loss(
    libpq_url: str, failures: int, caplog: pytest.LogCaptureFixture
) -> None:
    agent = f"connection_recovery_{uuid4().hex}"
    disconnects = CheckpointDisconnects()
    executions: list[str] = []
    options = dict(POOL_OPTIONS, cursor_factory=disconnects.cursor_class(), application_name=agent)
    with ConnectionPool[psycopg.Connection[dict[str, Any]]](
        libpq_url, min_size=1, max_size=1, kwargs=options
    ) as pool:
        saver = CockroachDBSaver(pool, retry_policy=POLICY)
        saver.setup()
        graph = recovery_graph(saver, lambda: disconnects.arm(failures), executions)
        try:
            with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
                if failures == ATTEMPTS:
                    with pytest.raises(psycopg.OperationalError):
                        graph.invoke({"agent": agent}, config(agent), durability="sync")
                    assert len(disconnects.closed_connections) == ATTEMPTS
                    assert_exhaustion_log(caplog)
                    assert_exhausted(saver.get_tuple(config(agent)), agent, executions)
                    result = graph.invoke(None, config(agent), durability="sync")
                else:
                    result = graph.invoke({"agent": agent}, config(agent), durability="sync")
            assert_recovered(result, agent, executions, disconnects, failures)
            saved = saver.get_tuple(config(agent))
            assert saved is not None
            assert saved.checkpoint["channel_values"]["finished"] == result["finished"]
        finally:
            saver.delete_thread(agent)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("failures", [1, ATTEMPTS], ids=["retry", "exhaust_and_resume"])
async def test_async_graph_recovers_after_checkpoint_connection_loss(
    libpq_url: str, failures: int, caplog: pytest.LogCaptureFixture
) -> None:
    agent = f"connection_recovery_{uuid4().hex}"
    disconnects = CheckpointDisconnects()
    executions: list[str] = []
    options = dict(
        POOL_OPTIONS, cursor_factory=disconnects.async_cursor_class(), application_name=agent
    )
    async with AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]](
        libpq_url, min_size=1, max_size=1, kwargs=options, open=False
    ) as pool:
        saver = AsyncCockroachDBSaver(pool, retry_policy=POLICY)
        await saver.setup()
        graph = recovery_graph(saver, lambda: disconnects.arm(failures), executions)
        try:
            with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"):
                if failures == ATTEMPTS:
                    with pytest.raises(psycopg.OperationalError):
                        await graph.ainvoke({"agent": agent}, config(agent), durability="sync")
                    assert len(disconnects.closed_connections) == ATTEMPTS
                    assert_exhaustion_log(caplog)
                    assert_exhausted(await saver.aget_tuple(config(agent)), agent, executions)
                    result = await graph.ainvoke(None, config(agent), durability="sync")
                else:
                    result = await graph.ainvoke({"agent": agent}, config(agent), durability="sync")
            assert_recovered(result, agent, executions, disconnects, failures)
            saved = await saver.aget_tuple(config(agent))
            assert saved is not None
            assert saved.checkpoint["channel_values"]["finished"] == result["finished"]
        finally:
            await saver.adelete_thread(agent)
