"""Background event loop that runs database work for the sync API.

SQLAlchemy's async engine and psycopg's async connections belong to one event
loop at a time. Sync methods used to call asyncio.run() per call, so every call
and every thread brought its own loop to a shared pool, which fails under
concurrency. All sync calls now run on this one loop instead.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

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


def run_coroutine_sync(coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop) -> T:
    """Run ``coro`` on ``loop`` and block until it finishes.

    Raises:
        RuntimeError: If called from ``loop``'s own thread, which would deadlock
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        coro.close()
        raise RuntimeError(
            "A sync method was called from the event loop that runs it, which would "
            "deadlock; use the async method instead"
        )
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise


def _reset_after_fork() -> None:
    """The loop's thread does not survive fork; start a new one on demand."""
    global _lock, _loop, _thread
    _lock = threading.Lock()
    _loop = None
    _thread = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
