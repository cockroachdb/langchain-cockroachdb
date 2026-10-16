"""How the checkpointers run operations side by side, on fake psycopg connections.

A saver on a pool runs each operation on its own pooled connection, so
operations overlap. A saver on a single connection, with or without a
saver-wide pipeline, runs them one at a time under its lock.

Every wait has a timeout, so a test that would hang fails instead.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver, RetryPolicy
from tests.unit.checkpointer_fakes import (
    FakeAsyncConnection,
    FakeAsyncPipe,
    FakeAsyncPool,
    FakeConnection,
    FakePipe,
    FakePool,
    FakeSingleAsyncConnection,
    FakeSingleConnection,
    close_sockets,
)

CONFIG: Any = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
ONCE = RetryPolicy(max_attempts=1)
WAIT = 5.0


@pytest.fixture(autouse=True)
def _close_sockets() -> Iterator[None]:
    yield
    close_sockets()


class AsyncMeeting:
    """Lets ``parties`` tasks wait until all of them have arrived.

    asyncio.Barrier needs Python 3.11, and the package supports 3.10.
    """

    def __init__(self, parties: int, timeout: float) -> None:
        self.parties = parties
        self.timeout = timeout
        self.arrived = 0
        self._everyone = asyncio.Event()

    async def wait(self, query: str) -> None:
        self.arrived += 1
        if self.arrived >= self.parties:
            self._everyone.set()
        await asyncio.wait_for(self._everyone.wait(), self.timeout)


class WatchedLock:
    """A threading.Lock that says when a caller had to wait for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class WatchedAsyncLock(asyncio.Lock):
    """An asyncio.Lock that says when a caller had to wait for it."""

    def __init__(self) -> None:
        super().__init__()
        self.contended = asyncio.Event()

    async def acquire(self) -> bool:
        if self.locked():
            self.contended.set()
        return await super().acquire()


def run_in_threads(*operations: Callable[[], Any]) -> tuple[list[threading.Thread], list[Any]]:
    """Start each operation in its own thread; errors land in the returned list."""
    errors: list[Any] = []

    def run(operation: Callable[[], Any]) -> None:
        try:
            operation()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=run, args=(op,), daemon=True) for op in operations]
    for thread in threads:
        thread.start()
    return threads, errors


def join(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(WAIT * 2)
    assert not any(thread.is_alive() for thread in threads), "an operation did not finish"


@pytest.mark.timeout(15)
def test_pooled_sync_saver_runs_operations_at_the_same_time() -> None:
    # Each statement waits until the other operation is inside its own
    # connection too. Under a saver-wide lock the barrier would time out.
    barrier = threading.Barrier(2, timeout=WAIT)
    connections = FakePool(factory=lambda: FakeConnection(on_statement=lambda q: barrier.wait()))
    saver = CockroachDBSaver(connections, retry_policy=ONCE)
    threads, errors = run_in_threads(
        lambda: saver.put(CONFIG, empty_checkpoint(), {}, {}),
        lambda: saver.get_tuple(CONFIG),
    )
    join(threads)
    assert errors == []
    assert len(connections.handed_out) == 2


@pytest.mark.timeout(15)
async def test_pooled_async_saver_runs_operations_at_the_same_time() -> None:
    meeting = AsyncMeeting(2, timeout=WAIT)
    connections = FakeAsyncPool(factory=lambda: FakeAsyncConnection(on_statement=meeting.wait))
    saver = AsyncCockroachDBSaver(connections, retry_policy=ONCE)
    await asyncio.gather(
        saver.aput(CONFIG, empty_checkpoint(), {}, {}),
        saver.aget_tuple(CONFIG),
    )
    assert meeting.arrived == 2
    assert len(connections.handed_out) == 2


@pytest.mark.timeout(15)
@pytest.mark.parametrize("pipeline", [False, True], ids=["connection", "pipeline"])
def test_single_connection_sync_saver_runs_operations_one_at_a_time(pipeline: bool) -> None:
    first_inside = threading.Event()
    release = threading.Event()

    def hold_the_first(query: str) -> None:
        if not first_inside.is_set():
            first_inside.set()
            assert release.wait(WAIT)

    conn = FakeSingleConnection(on_statement=hold_the_first)
    saver = CockroachDBSaver(conn, FakePipe() if pipeline else None, retry_policy=ONCE)
    lock = saver.lock = WatchedLock()
    first, first_errors = run_in_threads(lambda: saver.get_tuple(CONFIG))
    assert first_inside.wait(WAIT)
    second, second_errors = run_in_threads(lambda: saver.put(CONFIG, empty_checkpoint(), {}, {}))
    # The second operation waits for the lock while the first is inside the connection
    assert lock.contended.wait(WAIT)
    assert len(conn.statements) == 1
    release.set()
    join(first + second)
    assert first_errors == second_errors == []
    assert conn.statements[0].strip().startswith("select")
    assert "INSERT INTO checkpoints" in conn.statements[1]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("pipeline", [False, True], ids=["connection", "pipeline"])
async def test_single_connection_async_saver_runs_operations_one_at_a_time(
    pipeline: bool,
) -> None:
    first_inside = asyncio.Event()
    release = asyncio.Event()

    async def hold_the_first(query: str) -> None:
        if not first_inside.is_set():
            first_inside.set()
            await asyncio.wait_for(release.wait(), WAIT)

    conn = FakeSingleAsyncConnection(on_statement=hold_the_first)
    pipe = FakeAsyncPipe() if pipeline else None
    saver = AsyncCockroachDBSaver(conn, pipe, retry_policy=ONCE)
    lock = saver.lock = WatchedAsyncLock()
    first = asyncio.create_task(saver.aget_tuple(CONFIG))
    await asyncio.wait_for(first_inside.wait(), WAIT)
    second = asyncio.create_task(saver.aput(CONFIG, empty_checkpoint(), {}, {}))
    await asyncio.wait_for(lock.contended.wait(), WAIT)
    assert len(conn.statements) == 1
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), WAIT)
    assert conn.statements[0].strip().startswith("select")
    assert "INSERT INTO checkpoints" in conn.statements[1]
