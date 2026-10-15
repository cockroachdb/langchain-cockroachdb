"""AsyncCockroachDBSaver's sync methods.

They run the async methods on the event loop the saver was created on and wait
for the result, so they are meant for other threads. These tests use real loops
in daemon threads and stub the async methods, so no database is needed. Each
test has a timeout, so a call that hangs fails its test instead of hanging the
suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import threading
import warnings
import weakref
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, _loop

CONFIG: Any = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
SAVED: Any = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "1"}}


class StubSaver(AsyncCockroachDBSaver):
    """The real sync methods in front of async methods that need no database."""

    def __init__(self) -> None:
        super().__init__(AsyncConnectionPool("", open=False))
        self.calls: list[str] = []
        self.started = threading.Event()
        # Seconds every async method waits before it returns, if set
        self.hold: float | None = None

    async def _step(self, name: str) -> None:
        assert asyncio.get_running_loop() is self.loop
        self.calls.append(name)
        self.started.set()
        if self.hold is not None:
            await asyncio.sleep(self.hold)

    async def aget_tuple(self, config: Any) -> Any:
        await self._step("aget_tuple")
        return "checkpoint tuple"

    async def alist(
        self, config: Any, *, filter: Any = None, before: Any = None, limit: Any = None
    ) -> AsyncIterator[Any]:
        await self._step("alist")
        for item in ("first", "second"):
            yield item

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        await self._step("aput")
        return "next config"

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        await self._step("aput_writes")

    async def adelete_thread(self, thread_id: str) -> None:
        await self._step("adelete_thread")


CALLS: dict[str, Callable[[StubSaver], Any]] = {
    "get_tuple": lambda saver: saver.get_tuple(CONFIG),
    "list": lambda saver: [*saver.list(CONFIG)],
    "put": lambda saver: saver.put(CONFIG, empty_checkpoint(), {}, {}),
    "put_writes": lambda saver: saver.put_writes(SAVED, [("channel", "value")], "task-1"),
    "delete_thread": lambda saver: saver.delete_thread("thread-1"),
}
RESULTS: dict[str, Any] = {
    "get_tuple": "checkpoint tuple",
    "list": ["first", "second"],
    "put": "next config",
    "put_writes": None,
    "delete_thread": None,
}
METHODS = pytest.mark.parametrize("method", list(CALLS))


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start a new loop in a daemon thread; return once it runs."""
    loop = asyncio.new_event_loop()
    running = threading.Event()
    loop.call_soon(running.set)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    assert running.wait(5), "the loop did not start"
    return loop, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)


def _close_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    if not loop.is_closed():
        _stop_loop(loop, thread)
        loop.close()


def _saver_on(loop: asyncio.AbstractEventLoop) -> StubSaver:
    """Create a saver on ``loop``, as an application's async code would."""

    async def create() -> StubSaver:
        return StubSaver()

    return asyncio.run_coroutine_threadsafe(create(), loop).result(timeout=5)


def _fast_checks(monkeypatch: pytest.MonkeyPatch, start_timeout: float) -> None:
    """Check the loop often, and give a loop that is not running start_timeout seconds."""
    monkeypatch.setattr(_loop, "LOOP_CHECK_INTERVAL", 0.02)
    monkeypatch.setattr(_loop, "LOOP_START_TIMEOUT", start_timeout)


@contextlib.contextmanager
def _no_unawaited_coroutines() -> Iterator[None]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
        gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_run_on_the_saver_loop_from_other_threads(method: str) -> None:
    loop, thread = _start_loop()
    try:
        saver = _saver_on(loop)
        results: list[Any] = []
        callers = [
            threading.Thread(target=lambda: results.append(CALLS[method](saver)), daemon=True)
            for _ in range(4)
        ]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(5)
        assert results == [RESULTS[method]] * 4
        assert len(saver.calls) == 4
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_refuse_calls_from_the_saver_loop(method: str) -> None:
    # Waiting for the loop from the loop's own thread would deadlock
    loop, thread = _start_loop()
    try:
        saver = _saver_on(loop)

        async def call_from_the_loop() -> BaseException | None:
            try:
                CALLS[method](saver)
            except asyncio.InvalidStateError as error:
                return error
            return None

        with _no_unawaited_coroutines():
            error = asyncio.run_coroutine_threadsafe(call_from_the_loop(), loop).result(timeout=3)
        assert isinstance(error, asyncio.InvalidStateError)
        assert "only allowed from a different thread" in str(error)
        assert saver.calls == []
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_fast_on_a_closed_loop(method: str) -> None:
    loop, thread = _start_loop()
    saver = _saver_on(loop)
    _close_loop(loop, thread)
    with (
        _no_unawaited_coroutines(),
        pytest.raises(RuntimeError, match="AsyncCockroachDBSaver was created on is closed"),
    ):
        CALLS[method](saver)
    assert saver.calls == []


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_fast_when_the_loop_is_not_running(method: str, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    saver = _saver_on(loop)
    _stop_loop(loop, thread)
    try:
        with (
            _no_unawaited_coroutines(),
            pytest.raises(RuntimeError, match="was created on is not running"),
        ):
            CALLS[method](saver)
        # Nothing was left on the loop to run once it runs again
        loop.run_until_complete(asyncio.sleep(0.05))
        assert saver.calls == []
    finally:
        loop.close()


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_fast_in_a_forked_child(method: str, monkeypatch) -> None:
    # In a fork child the copied loop still reports running, but no thread runs
    # it there. A loop whose thread is stuck in a callback looks the same.
    loop, thread = _start_loop()
    saver = _saver_on(loop)
    release = threading.Event()
    loop.call_soon_threadsafe(release.wait, 30)
    parent_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    try:
        assert loop.is_running()
        with (
            _no_unawaited_coroutines(),
            pytest.raises(
                RuntimeError,
                match=rf"created in process {parent_pid}, .* forked child .* Create a new saver",
            ),
        ):
            CALLS[method](saver)
    finally:
        monkeypatch.undo()
        release.set()
        _close_loop(loop, thread)
    assert saver.calls == []


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_when_the_loop_stops_during_a_call(method: str, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    saver = _saver_on(loop)
    saver.hold = 30

    def stop_once_started() -> None:
        if saver.started.wait(5):
            loop.call_soon_threadsafe(loop.stop)

    stopper = threading.Thread(target=stop_once_started, daemon=True)
    stopper.start()
    try:
        with pytest.raises(RuntimeError, match="was created on stopped before the call finished"):
            CALLS[method](saver)
        # The call cancelled its work; let the loop run the cancellation
        thread.join(5)
        loop.run_until_complete(asyncio.sleep(0.1))
    finally:
        stopper.join(5)
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
def test_sync_methods_fail_when_the_loop_closes_during_a_call(monkeypatch, caplog) -> None:
    # A long grace for a loop that is not running, so the call sees it closed
    _fast_checks(monkeypatch, start_timeout=5.0)
    loop, thread = _start_loop()
    saver = _saver_on(loop)
    saver.hold = 30

    def close_once_started() -> None:
        if saver.started.wait(5):
            _stop_loop(loop, thread)
            loop.close()

    closer = threading.Thread(target=close_once_started, daemon=True)
    closer.start()
    try:
        with pytest.raises(
            RuntimeError, match="was created on was closed before the call finished"
        ):
            CALLS["put"](saver)
        # The closed loop's pending task is collected here, not during a later test
        gc.collect()
    finally:
        closer.join(5)
        _close_loop(loop, thread)
    errors = [r.getMessage().splitlines()[0] for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == ["Task was destroyed but it is pending!"]


@pytest.mark.timeout(10)
def test_sync_methods_refuse_a_loop_that_was_running_when_the_process_forked(
    monkeypatch,
) -> None:
    # A real child refuses this saver on its pid first. The pid matches here,
    # so this shows the saver's wording for the forked-loop check on its own.
    loop, thread = _start_loop()
    saver = _saver_on(loop)
    monkeypatch.setattr(_loop, "_forked_loops", weakref.WeakSet())
    _loop.note_forked_loop(loop)
    try:
        with (
            _no_unawaited_coroutines(),
            pytest.raises(
                RuntimeError,
                match=r"AsyncCockroachDBSaver was created on was running in the parent process",
            ),
        ):
            CALLS["get_tuple"](saver)
        assert saver.calls == []
    finally:
        _close_loop(loop, thread)
