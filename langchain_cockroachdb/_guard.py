"""Cut off database attempts that hang.

While a range it wrote to has lost quorum, CockroachDB can block a COMMIT past
statement_timeout, transaction_timeout and cancel requests
(cockroachlabs/cockroach#13692). A blackholed network can stall a read for as
long as the operating system's TCP timeouts allow. Shutting down the
connection's socket makes the blocked call fail as a lost connection, which
the retry policy then handles like any other connection error.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import threading
from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger("langchain_cockroachdb.retry")


def shutdown_socket(fileno: int) -> None:
    """Shut down the socket behind ``fileno`` in both directions.

    Works on a duplicate of the descriptor, so the driver still owns and
    closes the original. Errors are ignored because the socket may be gone.
    """
    try:
        duplicate = os.dup(fileno)
    except OSError:
        return
    try:
        sock = socket.socket(fileno=duplicate)
    except OSError:
        os.close(duplicate)
        return
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    sock.close()


def _abandon(fileno: int, seconds: float, label: str) -> None:
    logger.warning(
        "%s ran longer than its %ss attempt timeout; abandoning its connection", label, seconds
    )
    shutdown_socket(fileno)


@contextlib.contextmanager
def abandon_after(fileno: int | None, seconds: float | None, *, label: str) -> Iterator[None]:
    """Shut down ``fileno`` if the block is still running after ``seconds``.

    For synchronous code; the timer runs on a daemon thread. Does nothing when
    either argument is None. The lock makes sure the socket is never touched
    after the block has finished and the connection may belong to someone else.
    """
    if fileno is None or seconds is None:
        yield
        return
    lock = threading.Lock()
    finished = False

    def fire() -> None:
        with lock:
            if not finished:
                _abandon(fileno, seconds, label)

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        with lock:
            finished = True
        timer.cancel()


@contextlib.asynccontextmanager
async def aabandon_after(
    fileno: int | None, seconds: float | None, *, label: str
) -> AsyncIterator[None]:
    """Async version of :func:`abandon_after`, scheduled on the running loop."""
    if fileno is None or seconds is None:
        yield
        return
    handle = asyncio.get_running_loop().call_later(seconds, _abandon, fileno, seconds, label)
    try:
        yield
    finally:
        handle.cancel()
