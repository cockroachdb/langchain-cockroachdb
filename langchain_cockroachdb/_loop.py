"""Background event loop that runs database work for the sync API.

SQLAlchemy's async engine and psycopg's async connections belong to one event
loop at a time. Sync methods used to call asyncio.run() per call, so every call
and every thread brought its own loop to a shared pool, which fails under
concurrency. All sync calls now run on this one loop instead.

An engine built with ``loop=`` sends its work to that loop. The library does
not own it, so a call checks that it is running, and raises instead of waiting
forever when the loop is closed or stops, or when the call comes from a forked
child.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import os
import threading
import time
from collections.abc import Coroutine, Iterator
from typing import Any, TypeVar

T = TypeVar("T")

# How often a call waiting on a loop passed as loop= checks that the loop is
# still running, in seconds.
LOOP_CHECK_INTERVAL = 0.1

# How long a loop passed as loop= may be not running before calls give up on
# it, in seconds. A loop whose thread was just started is not running yet for a
# moment, and a loop can be stopped and run again.
LOOP_START_TIMEOUT = 1.0

_ADVICE = (
    "Run that loop in its own thread (for example with loop.run_forever()) for as long "
    "as the engine is in use, or omit loop= so the engine uses the library's own "
    "background loop."
)
_CLOSED = f"The event loop passed as loop= is closed. {_ADVICE}"
_NOT_RUNNING = (
    f"The event loop passed as loop= is not running, so the call would wait forever. {_ADVICE}"
)
_CLOSED_WHILE_WAITING = (
    "The event loop passed as loop= was closed before the call finished. The work may "
    f"or may not have taken effect. {_ADVICE}"
)
_STOPPED_WHILE_WAITING = (
    "The event loop passed as loop= stopped before the call finished. The work may or "
    f"may not have taken effect. {_ADVICE}"
)

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def is_background_loop(loop: asyncio.AbstractEventLoop) -> bool:
    """Check if loop is the background loop without creating it."""
    return loop is _loop


def get_background_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide loop, starting its daemon thread on first use."""
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed() or _thread is None or not _thread.is_alive():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="langchain-cockroachdb-loop", daemon=True
            )
            thread.start()
            _loop, _thread = loop, thread
        return _loop


def _refuse_own_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Raise if the caller runs on ``loop``, where waiting for it would deadlock."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError(
            "A sync method was called from the event loop that runs it, which would "
            "deadlock; use the async method instead"
        )


def run_coroutine_sync(coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop) -> T:
    """Run ``coro`` on ``loop`` and block until it finishes.

    Raises:
        RuntimeError: If called from ``loop``'s own thread, which would deadlock
    """
    try:
        _refuse_own_loop(loop)
    except RuntimeError:
        coro.close()
        raise
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise


def _refuse_forked_child(owner_pid: int) -> None:
    """Raise unless this is the process that built the engine.

    A fork child has a copy of the loop that can still report it is running,
    but the thread that ran it exists only in the parent.
    """
    pid = os.getpid()
    if pid != owner_pid:
        raise RuntimeError(
            f"This engine was built with loop= in process {owner_pid}, and this is "
            f"process {pid}, a forked child of it. No thread runs that loop here, so "
            "the call could never finish. Build a new engine in this process."
        )


def _until_running(loop: asyncio.AbstractEventLoop) -> Iterator[float]:
    """Yield how long to sleep while ``loop`` is not running yet.

    Raises:
        RuntimeError: If ``loop`` is closed, or is still not running after
            LOOP_START_TIMEOUT
    """
    deadline = time.monotonic() + LOOP_START_TIMEOUT
    while not loop.is_running():
        if loop.is_closed():
            raise RuntimeError(_CLOSED)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(_NOT_RUNNING)
        yield min(LOOP_CHECK_INTERVAL, remaining)


def _submit(
    coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop
) -> concurrent.futures.Future[T]:
    """Hand ``coro`` to ``loop``, which may have closed since it was checked."""
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        if not loop.is_closed():
            raise
        raise RuntimeError(_CLOSED) from None


class _LoopWatch:
    """Notice when a loop passed as loop= goes away while a call waits on it."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stopped_since: float | None = None

    def problem(self) -> str | None:
        """Return why the call should stop waiting, or None to keep waiting."""
        if self._loop.is_closed():
            return _CLOSED_WHILE_WAITING
        if self._loop.is_running():
            self._stopped_since = None
            return None
        now = time.monotonic()
        if self._stopped_since is None:
            self._stopped_since = now
        elif now - self._stopped_since >= LOOP_START_TIMEOUT:
            return _STOPPED_WHILE_WAITING
        return None


def _give_up(
    future: concurrent.futures.Future[Any],
    coro: Coroutine[Any, Any, Any],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Abandon work handed to ``loop`` when the caller stops waiting for it."""
    if loop.is_closed():
        # A closed loop runs nothing, and cancelling would only make asyncio try
        # to schedule on it and log an error. Close work it never started, so
        # Python does not warn that it was never awaited.
        if inspect.iscoroutine(coro) and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
            coro.close()
    else:
        # Cancels the work on the loop now, or when the loop runs again
        future.cancel()


def run_on_external_loop(
    coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop, *, owner_pid: int
) -> T:
    """Run ``coro`` on a loop passed as loop= and block until it finishes.

    Unlike run_coroutine_sync, this never waits forever. It checks the loop
    before handing over the work, and every LOOP_CHECK_INTERVAL while waiting.

    Args:
        coro: The work to run
        loop: The loop passed as loop=
        owner_pid: The process that built the engine

    Raises:
        RuntimeError: If called from ``loop``'s own thread or from a forked
            child, or if ``loop`` is closed, is not running, or stops before
            the work finishes
    """
    try:
        _refuse_forked_child(owner_pid)
        _refuse_own_loop(loop)
        for delay in _until_running(loop):
            time.sleep(delay)
        future = _submit(coro, loop)
    except BaseException:
        coro.close()
        raise
    watch = _LoopWatch(loop)
    try:
        while True:
            done, _ = concurrent.futures.wait((future,), timeout=LOOP_CHECK_INTERVAL)
            if done:
                break
            problem = watch.problem()
            if problem is not None and not future.done():
                raise RuntimeError(problem)
    except BaseException:
        _give_up(future, coro, loop)
        raise
    return future.result()


async def arun_on_external_loop(
    coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop, *, owner_pid: int
) -> T:
    """Run ``coro`` on a loop passed as loop=, from a coroutine on another loop.

    The async version of run_on_external_loop. Waiting does not block the
    caller's loop, and cancelling the caller cancels the work.
    """
    try:
        _refuse_forked_child(owner_pid)
        for delay in _until_running(loop):
            await asyncio.sleep(delay)
        future = _submit(coro, loop)
    except BaseException:
        coro.close()
        raise
    outcome = asyncio.wrap_future(future)
    watch = _LoopWatch(loop)
    try:
        while True:
            done, _ = await asyncio.wait({outcome}, timeout=LOOP_CHECK_INTERVAL)
            if done:
                break
            problem = watch.problem()
            if problem is not None and not future.done():
                raise RuntimeError(problem)
    except BaseException:
        _give_up(future, coro, loop)
        raise
    return outcome.result()


def _reset_after_fork() -> None:
    """The loop's thread does not survive fork; start a new one on demand."""
    global _lock, _loop, _thread
    _lock = threading.Lock()
    _loop = None
    _thread = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
