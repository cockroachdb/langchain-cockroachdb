"""AsyncCockroachDBStore construction and its sync methods. No database.

The sync methods run the async ones on the event loop the store was created
on and wait for the result, so they are meant for other threads. The loop
tests, ported from the AsyncCockroachDBSaver ones, use real loops in daemon
threads and stub the async methods, so no database is needed. Each has a
timeout, so a call that hangs fails its test instead of hanging the suite.
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
from typing import Any, TypeVar

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import (
    NOT_PROVIDED,
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb import AsyncCockroachDBStore, _loop
from langchain_cockroachdb.store import AsyncCockroachDBStore as StoreExport

S = TypeVar("S", bound=AsyncCockroachDBStore)


def test_construction_outside_a_running_loop_raises() -> None:
    pool = AsyncConnectionPool("", open=False)
    with pytest.raises(RuntimeError):
        AsyncCockroachDBStore(pool)


async def test_construction_inside_a_loop_with_a_pool() -> None:
    assert StoreExport is AsyncCockroachDBStore
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False), table_name="memories")
    assert store.supports_ttl is True
    assert store.loop is asyncio.get_running_loop()
    assert store.sql.migrations_table.endswith("memories_migrations")
    assert store.pipe is None


async def test_pool_with_pipeline_is_rejected_and_index_is_validated() -> None:
    pool = AsyncConnectionPool("", open=False)
    with pytest.raises(ValueError, match="Pipeline"):
        AsyncCockroachDBStore(pool, pipe=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embed"):
        AsyncCockroachDBStore(pool, index={"dims": 3, "embed": None})  # type: ignore[typeddict-item]
    store = AsyncCockroachDBStore(
        pool, index={"dims": 3, "embed": DeterministicFakeEmbedding(size=3)}
    )
    assert store.index is not None and store.index.dims == 3


async def test_sync_batch_from_the_stores_own_loop_raises() -> None:
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False))
    with _no_unawaited_coroutines():
        with pytest.raises(asyncio.InvalidStateError, match="only allowed from a different thread"):
            store.batch([])
        with pytest.raises(asyncio.InvalidStateError):
            store.get(("a",), "k")


class RecordingAsyncCursor:
    """An async cursor that records executes and fetches in a shared log."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.connection = self

    def cursor(self, **kwargs: object) -> RecordingAsyncCursor:
        return RecordingAsyncCursor(self.log)

    async def execute(self, query: str, params: list) -> None:
        self.log.append("execute")

    async def fetchall(self) -> list:
        self.log.append("fetch")
        return []

    async def close(self) -> None:
        pass


async def test_read_group_queues_every_statement_before_fetching() -> None:
    store = AsyncCockroachDBStore(AsyncConnectionPool("", open=False))
    log: list[str] = []

    @contextlib.asynccontextmanager
    async def fake_cursor(**kwargs: object) -> AsyncIterator[RecordingAsyncCursor]:
        log.append("cursor")
        yield RecordingAsyncCursor(log)

    store._cursor = fake_cursor  # type: ignore[method-assign,assignment]
    results = await store.abatch([GetOp(("a",), "k"), GetOp(("b",), "k"), GetOp(("c",), "k")])
    assert results == [None, None, None]
    # One cursor context, three statements, every fetch after every execute
    assert log == ["cursor"] + ["execute"] * 3 + ["fetch"] * 3

    log.clear()
    assert await store.abatch([SearchOp(("a",)), SearchOp(("b",), limit=0)]) == [[], []]
    assert log == ["cursor", "execute", "fetch"]


class StubStore(AsyncCockroachDBStore):
    """The real sync methods in front of async methods that need no database."""

    def __init__(self) -> None:
        super().__init__(AsyncConnectionPool("", open=False))
        self.calls: list[str] = []
        self.started = threading.Event()
        self.cancelled = threading.Event()
        # Seconds every async method waits before it returns, if set
        self.hold: float | None = None

    async def _step(self, name: str) -> None:
        assert asyncio.get_running_loop() is self.loop
        self.calls.append(name)
        self.started.set()
        if self.hold is not None:
            try:
                await asyncio.sleep(self.hold)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def abatch(self, ops: Any) -> Any:
        ops = list(ops)
        await self._step("abatch")
        return ["batch result"] * len(ops)

    async def aget(self, namespace: Any, key: Any, *, refresh_ttl: Any = None) -> Any:
        await self._step("aget")
        return "item"

    async def asearch(
        self,
        namespace_prefix: Any,
        /,
        *,
        query: Any = None,
        filter: Any = None,
        limit: Any = 10,
        offset: Any = 0,
        refresh_ttl: Any = None,
    ) -> Any:
        await self._step("asearch")
        return ["search item"]

    async def aput(
        self, namespace: Any, key: Any, value: Any, index: Any = None, *, ttl: Any = NOT_PROVIDED
    ) -> None:
        await self._step("aput")

    async def adelete(self, namespace: Any, key: Any) -> None:
        await self._step("adelete")

    async def alist_namespaces(
        self,
        *,
        prefix: Any = None,
        suffix: Any = None,
        max_depth: Any = None,
        limit: Any = 100,
        offset: Any = 0,
    ) -> Any:
        await self._step("alist_namespaces")
        return [("a",)]


CALLS: dict[str, Callable[[StubStore], Any]] = {
    "batch": lambda store: store.batch([GetOp(("a",), "k")]),
    "get": lambda store: store.get(("a",), "k"),
    "search": lambda store: store.search(("a",), query="q"),
    "put": lambda store: store.put(("a",), "k", {"v": 1}),
    "delete": lambda store: store.delete(("a",), "k"),
    "list_namespaces": lambda store: store.list_namespaces(prefix=("a",)),
}
RESULTS: dict[str, Any] = {
    "batch": ["batch result"],
    "get": "item",
    "search": ["search item"],
    "put": None,
    "delete": None,
    "list_namespaces": [("a",)],
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


async def _stop_queue(store: AsyncCockroachDBStore) -> None:
    """Stop the task that runs the batching queue, so closing the loop leaves no task pending."""
    task = store._task
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _store_on(loop: asyncio.AbstractEventLoop, cls: type[S], *, queue: bool = False) -> S:
    """Create a store on ``loop``, as an application's async code would.

    The stubs of StubStore never use the batching queue, so its task stops
    unless ``queue`` is set.
    """

    async def create() -> S:
        store = cls()
        if not queue:
            await _stop_queue(store)
        return store

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
def test_sync_methods_run_on_the_store_loop_from_other_threads(method: str) -> None:
    loop, thread = _start_loop()
    try:
        store = _store_on(loop, StubStore)
        results: list[Any] = []
        callers = [
            threading.Thread(target=lambda: results.append(CALLS[method](store)), daemon=True)
            for _ in range(4)
        ]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(5)
        assert results == [RESULTS[method]] * 4
        # Each sync method awaits its own async twin
        assert store.calls == ["a" + method] * 4
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_refuse_calls_from_the_store_loop(method: str) -> None:
    # Waiting for the loop from the loop's own thread would deadlock
    loop, thread = _start_loop()
    try:
        store = _store_on(loop, StubStore)

        async def call_from_the_loop() -> BaseException | None:
            try:
                CALLS[method](store)
            except asyncio.InvalidStateError as error:
                return error
            return None

        with _no_unawaited_coroutines():
            error = asyncio.run_coroutine_threadsafe(call_from_the_loop(), loop).result(timeout=3)
        assert isinstance(error, asyncio.InvalidStateError)
        assert "only allowed from a different thread" in str(error)
        assert store.calls == []
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_fast_on_a_closed_loop(method: str) -> None:
    loop, thread = _start_loop()
    store = _store_on(loop, StubStore)
    _close_loop(loop, thread)
    with (
        _no_unawaited_coroutines(),
        pytest.raises(RuntimeError, match="AsyncCockroachDBStore was created on is closed"),
    ):
        CALLS[method](store)
    assert store.calls == []


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_fast_when_the_loop_is_not_running(method: str, monkeypatch) -> None:
    # The review's case: the loop is open but stopped, after run_until_complete
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    store = _store_on(loop, StubStore)
    _stop_loop(loop, thread)
    try:
        with (
            _no_unawaited_coroutines(),
            pytest.raises(RuntimeError, match="was created on is not running"),
        ):
            CALLS[method](store)
        # Nothing was left on the loop to run once it runs again
        loop.run_until_complete(asyncio.sleep(0.05))
        assert store.calls == []
    finally:
        loop.close()


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_fast_in_a_forked_child(method: str, monkeypatch) -> None:
    # In a fork child the copied loop still reports running, but no thread runs
    # it there. A loop whose thread is stuck in a callback looks the same.
    loop, thread = _start_loop()
    store = _store_on(loop, StubStore)
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
                match=rf"created in process {parent_pid}, .* forked child .* Create a new store",
            ),
        ):
            CALLS[method](store)
    finally:
        monkeypatch.undo()
        release.set()
        _close_loop(loop, thread)
    assert store.calls == []


@pytest.mark.timeout(10)
@METHODS
def test_sync_methods_fail_when_the_loop_stops_during_a_call(method: str, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    store = _store_on(loop, StubStore)
    store.hold = 30

    def stop_once_started() -> None:
        if store.started.wait(5):
            loop.call_soon_threadsafe(loop.stop)

    stopper = threading.Thread(target=stop_once_started, daemon=True)
    stopper.start()
    try:
        with pytest.raises(RuntimeError, match="was created on stopped before the call finished"):
            CALLS[method](store)
        assert not store.cancelled.is_set()
        # The call cancelled its work, so the work does not go on if the loop runs again
        thread.join(5)
        loop.run_until_complete(asyncio.sleep(0.1))
        assert store.cancelled.is_set()
    finally:
        stopper.join(5)
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
def test_sync_methods_fail_when_the_loop_closes_during_a_call(monkeypatch, caplog) -> None:
    # A long grace for a loop that is not running, so the call sees it closed
    _fast_checks(monkeypatch, start_timeout=5.0)
    loop, thread = _start_loop()
    store = _store_on(loop, StubStore)
    store.hold = 30

    def close_once_started() -> None:
        if store.started.wait(5):
            _stop_loop(loop, thread)
            loop.close()

    closer = threading.Thread(target=close_once_started, daemon=True)
    closer.start()
    try:
        with pytest.raises(
            RuntimeError, match="was created on was closed before the call finished"
        ):
            CALLS["put"](store)
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
    # A real child refuses this store on its pid first. The pid matches here,
    # so this shows the store's wording for the forked-loop check on its own.
    loop, thread = _start_loop()
    store = _store_on(loop, StubStore)
    monkeypatch.setattr(_loop, "_forked_loops", weakref.WeakSet())
    _loop.note_forked_loop(loop)
    try:
        with (
            _no_unawaited_coroutines(),
            pytest.raises(
                RuntimeError,
                match=r"AsyncCockroachDBStore was created on belongs to the parent process",
            ),
        ):
            CALLS["get"](store)
        assert store.calls == []
    finally:
        _close_loop(loop, thread)


class QueueStore(AsyncCockroachDBStore):
    """The real async methods and batching queue in front of an abatch that needs no database."""

    def __init__(self) -> None:
        super().__init__(AsyncConnectionPool("", open=False))
        self.batches: list[list[Any]] = []

    async def abatch(self, ops: Any) -> Any:
        ops = list(ops)
        self.batches.append(ops)
        return [[("a",)] if isinstance(op, ListNamespacesOp) else None for op in ops]


@pytest.mark.timeout(10)
def test_sync_methods_go_through_the_batching_queue() -> None:
    # So the awaited calls of other code on the loop still coalesce with them
    loop, thread = _start_loop()
    store = _store_on(loop, QueueStore, queue=True)
    try:
        assert store.get(("a",), "k") is None
        assert store.search(("a",), query="q", limit=3) is None
        store.put(("a",), "k", {"v": 1})
        store.delete(("a",), "k")
        assert store.list_namespaces(prefix=("a",), max_depth=2) == [("a",)]
        assert store.batch([GetOp(("b",), "k")]) == [None]
        ops = [op for ops in store.batches for op in ops]
        assert [type(op) for op in ops] == [GetOp, SearchOp, PutOp, PutOp, ListNamespacesOp, GetOp]
        assert (ops[1].query, ops[1].limit) == ("q", 3)
        assert ops[2].value == {"v": 1} and ops[3].value is None
        assert ops[4].max_depth == 2
        # Errors of the async method reach the caller
        with pytest.raises(InvalidNamespaceError):
            store.put(("a.b",), "k", {"v": 1})
    finally:
        asyncio.run_coroutine_threadsafe(_stop_queue(store), loop).result(timeout=5)
        _close_loop(loop, thread)
