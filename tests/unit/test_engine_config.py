"""Engine configuration and loop dispatch tests that need no database."""

import asyncio
import concurrent.futures
import contextlib
import gc
import logging
import os
import threading
import time
import warnings
import weakref
from collections.abc import Coroutine, Iterator
from typing import Any, TypeVar

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import langchain_cockroachdb.engine as engine_module
from langchain_cockroachdb import CockroachDBEngine, RetryPolicy, _loop
from langchain_cockroachdb._loop import get_background_loop

T = TypeVar("T")

# Nothing connects in these tests; SQLAlchemy engines are lazy.
URL = "cockroachdb://root@localhost:1/defaultdb?sslmode=disable"


def test_url_is_normalized_and_connect_timeout_added(monkeypatch) -> None:
    captured = {}
    real = engine_module.create_async_engine

    def spy(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        return real(url, **kwargs)

    monkeypatch.setattr(engine_module, "create_async_engine", spy)
    engine = CockroachDBEngine.from_connection_string("postgresql://root@localhost:1/db")
    assert captured["url"] == "cockroachdb+psycopg://root@localhost:1/db"
    assert captured["kwargs"]["connect_args"] == {"connect_timeout": 10, "gssencmode": "disable"}
    assert engine.engine.url.drivername == "cockroachdb+psycopg"


def test_connect_timeout_can_be_disabled_or_overridden(monkeypatch) -> None:
    seen = []
    real = engine_module.create_async_engine
    monkeypatch.setattr(
        engine_module, "create_async_engine", lambda url, **kw: seen.append(kw) or real(url, **kw)
    )
    CockroachDBEngine.from_connection_string(URL, connect_timeout=None)
    CockroachDBEngine.from_connection_string(URL, connect_args={"connect_timeout": 3})
    CockroachDBEngine.from_connection_string(URL + "&connect_timeout=4")
    assert [kw["connect_args"] for kw in seen] == [
        {"gssencmode": "disable"},
        {"connect_timeout": 3, "gssencmode": "disable"},
        {"gssencmode": "disable"},
    ]


def test_flat_retry_arguments_build_the_policy() -> None:
    engine = CockroachDBEngine.from_connection_string(
        URL, retry_max_attempts=10, retry_jitter=False
    )
    assert engine.retry_policy.max_attempts == 10
    assert engine.retry_max_attempts == 10
    assert engine.retry_jitter is False
    assert engine.retry_initial_backoff == 0.1


def test_asyncpg_url_gets_no_connect_timeout(monkeypatch) -> None:
    """Asyncpg URLs should not get connect_args (no connect_timeout or gssencmode)."""
    captured = {}
    real = engine_module.create_async_engine

    def spy(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        try:
            return real(url, **kwargs)
        except ImportError:
            pass  # asyncpg not installed, but we captured the kwargs

    monkeypatch.setattr(engine_module, "create_async_engine", spy)
    with contextlib.suppress(ImportError):
        CockroachDBEngine.from_connection_string("cockroachdb+asyncpg://root@localhost:1/db")
    # Asyncpg driver doesn't use libpq, so no connect_args should be added
    assert "connect_args" not in captured["kwargs"]


def test_asyncpg_url_with_psycopg_in_username_gets_no_connect_timeout(monkeypatch) -> None:
    captured = {}
    real = engine_module.create_async_engine

    def spy(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        try:
            return real(url, **kwargs)
        except ImportError:
            pass

    monkeypatch.setattr(engine_module, "create_async_engine", spy)
    with contextlib.suppress(ImportError):
        CockroachDBEngine.from_connection_string(
            "cockroachdb+asyncpg://user+psycopg@localhost:1/db"
        )
    assert "connect_args" not in captured["kwargs"]


def test_psycopg_url_with_asyncpg_in_password_gets_connect_timeout(monkeypatch) -> None:
    captured = {}
    real = engine_module.create_async_engine

    def spy(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        return real(url, **kwargs)

    monkeypatch.setattr(engine_module, "create_async_engine", spy)
    CockroachDBEngine.from_connection_string("cockroachdb://root:pass+asyncpg@localhost:1/db")
    assert captured["kwargs"]["connect_args"] == {"connect_timeout": 10, "gssencmode": "disable"}


def test_policy_and_flat_arguments_conflict() -> None:
    with pytest.raises(ValueError):
        CockroachDBEngine.from_connection_string(
            URL, retry_policy=RetryPolicy(), retry_max_attempts=3
        )


def test_sync_runner_serves_many_threads() -> None:
    engine = CockroachDBEngine.from_connection_string(URL)

    async def work(n: int) -> int:
        await asyncio.sleep(0.01)
        return n

    results: list[int | None] = [None] * 16

    def call(i: int) -> None:
        results[i] = engine._run_as_sync(work(i))

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == list(range(16))


def test_sync_runner_works_inside_a_running_loop() -> None:
    engine = CockroachDBEngine.from_connection_string(URL)

    async def outer() -> str:
        return engine._run_as_sync(asyncio.sleep(0, result="ok"))

    assert asyncio.run(outer()) == "ok"


def test_sync_runner_refuses_to_deadlock_its_own_loop() -> None:
    engine = CockroachDBEngine.from_connection_string(URL)

    async def nested() -> None:
        engine._run_as_sync(asyncio.sleep(0))

    future = asyncio.run_coroutine_threadsafe(nested(), get_background_loop())
    with pytest.raises(RuntimeError, match="deadlock"):
        future.result(timeout=5)


async def test_async_runner_stays_on_the_callers_loop() -> None:
    engine = CockroachDBEngine.from_connection_string(URL)

    async def which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    assert await engine._run_as_async(which_loop()) is asyncio.get_running_loop()


async def test_async_calls_do_not_create_background_loop(monkeypatch) -> None:
    """Async calls must not reach get_background_loop (would create idle thread)."""
    engine = CockroachDBEngine.from_connection_string(URL)

    def raise_if_called():
        raise AssertionError("async call must not create background loop")

    monkeypatch.setattr(engine_module, "get_background_loop", raise_if_called)

    async def work() -> str:
        return "ok"

    result = await engine._run_as_async(work())
    assert result == "ok"


def test_explicit_loop_receives_sync_and_async_work() -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    engine = CockroachDBEngine.from_engine(
        create_async_engine("cockroachdb+psycopg://root@localhost:1/db"), loop=loop
    )

    async def which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    try:
        assert engine._run_as_sync(which_loop()) is loop
        assert asyncio.run(engine._run_as_async(which_loop())) is loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_mixed_sync_and_async_use_warns_once(caplog) -> None:
    engine = CockroachDBEngine.from_connection_string(URL)
    started, release = threading.Event(), threading.Event()

    async def slow() -> None:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)

    worker = threading.Thread(target=lambda: engine._run_as_sync(slow()), daemon=True)
    worker.start()
    try:
        assert started.wait(5)
        with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.engine"):
            asyncio.run(engine._run_as_async(asyncio.sleep(0)))
            asyncio.run(engine._run_as_async(asyncio.sleep(0)))
    finally:
        release.set()
        worker.join(5)
    assert caplog.text.count("one style per engine") == 1


def test_sync_call_to_async_method_does_not_warn(caplog) -> None:
    """Sync call to a public async method (aclose) must not trigger false mixed-use warning."""
    engine = CockroachDBEngine.from_connection_string(URL)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.engine"):
        engine._run_as_sync(engine.aclose())
    assert "one style per engine" not in caplog.text


def test_invalid_retry_max_attempts_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="max_attempts must be an integer of at least 1"):
        CockroachDBEngine.from_connection_string(URL, retry_max_attempts=0)


def test_invalid_retry_jitter_fails_at_construction() -> None:
    with pytest.raises(
        TypeError, match="jitter must be True, False, or None; for a ratio, use RetryPolicy"
    ):
        CockroachDBEngine.from_connection_string(URL, retry_jitter=0.3)  # type: ignore[arg-type]


def test_invalid_retry_initial_backoff_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="initial_backoff must not be negative"):
        CockroachDBEngine.from_connection_string(URL, retry_initial_backoff=-1.0)


def test_interruption_cancels_background_work() -> None:
    from unittest.mock import patch

    from langchain_cockroachdb._loop import run_coroutine_sync

    loop = get_background_loop()
    started = threading.Event()
    cancelled = threading.Event()

    async def long_work() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    original = asyncio.run_coroutine_threadsafe

    def interrupted(coro, target_loop):  # type: ignore[no-untyped-def]
        future = original(coro, target_loop)

        def raising_result(timeout=None):  # type: ignore[no-untyped-def]
            # Interrupt only once the work is running on the background loop
            assert started.wait(5), "background work never started"
            raise KeyboardInterrupt("simulated")

        future.result = raising_result
        return future

    with (
        patch("asyncio.run_coroutine_threadsafe", side_effect=interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        run_coroutine_sync(long_work(), loop)

    assert cancelled.wait(2), "running coroutine did not observe CancelledError"


def test_interruption_before_start_never_completes_the_work() -> None:
    from unittest.mock import patch

    from langchain_cockroachdb._loop import run_coroutine_sync

    loop = get_background_loop()
    completed = threading.Event()

    async def work() -> None:
        await asyncio.sleep(0.2)
        completed.set()

    original = asyncio.run_coroutine_threadsafe

    def interrupted(coro, target_loop):  # type: ignore[no-untyped-def]
        future = original(coro, target_loop)

        def raising_result(timeout=None):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt("simulated")

        future.result = raising_result
        return future

    with (
        patch("asyncio.run_coroutine_threadsafe", side_effect=interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        run_coroutine_sync(work(), loop)

    # Python 3.12 cancels the task after its first step, 3.13 and later before it;
    # either way the work must not finish
    assert not completed.wait(1), "work ran to completion after the caller was interrupted"


def test_reset_after_fork_starts_a_fresh_loop_on_demand(monkeypatch) -> None:
    from langchain_cockroachdb import _loop

    original = _loop.get_background_loop()
    # Restore the module state after the test, whatever the reset does to it
    monkeypatch.setattr(_loop, "_lock", _loop._lock)
    monkeypatch.setattr(_loop, "_loop", _loop._loop)
    monkeypatch.setattr(_loop, "_thread", _loop._thread)

    _loop._reset_after_fork()
    assert _loop._loop is None
    assert _loop._thread is None

    fresh = _loop.get_background_loop()
    thread = _loop._thread
    assert thread is not None
    try:
        assert fresh is not original
        future = asyncio.run_coroutine_threadsafe(asyncio.sleep(0, result=7), fresh)
        assert future.result(5) == 7
    finally:
        fresh.call_soon_threadsafe(fresh.stop)
        thread.join(5)
        fresh.close()


def test_id_type_rejects_trailing_newline() -> None:
    """Test that id_type validation rejects trailing newlines."""
    from langchain_cockroachdb.engine import _ID_TYPE

    # Valid id types should match
    assert _ID_TYPE.fullmatch("UUID")
    assert _ID_TYPE.fullmatch("VARCHAR(36)")

    # Trailing newline should not match
    assert _ID_TYPE.fullmatch("UUID\n") is None


# Engines built with loop= send work to a loop the library does not own. These
# tests close, stop and block real loops. Each one has a timeout, so a call
# that hangs fails its test instead of hanging the suite.

BUILDERS = pytest.mark.parametrize(
    "build", [CockroachDBEngine, CockroachDBEngine.from_engine], ids=["init", "from_engine"]
)
MODES = pytest.mark.parametrize("how", ["sync", "async"])


def _lazy_engine() -> AsyncEngine:
    return create_async_engine("cockroachdb+psycopg://root@localhost:1/db")


def _start_loop(
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start a loop (a new one by default) in a daemon thread; return once it runs."""
    loop = loop or asyncio.new_event_loop()
    running = threading.Event()
    loop.call_soon(running.set)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    assert running.wait(5), "the loop did not start"
    return loop, thread


def _close_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    if not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def _fast_checks(monkeypatch: pytest.MonkeyPatch, start_timeout: float) -> None:
    """Check the loop often, and give a loop that is not running start_timeout seconds."""
    monkeypatch.setattr(_loop, "LOOP_CHECK_INTERVAL", 0.02)
    monkeypatch.setattr(_loop, "LOOP_START_TIMEOUT", start_timeout)


def _isolate_fork_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the test its own engine registry and forked-loop record.

    The test can then run the at-fork hook, as a child process would, without
    touching other tests' engines or leaving records behind.
    """
    monkeypatch.setattr(engine_module, "_ENGINES", weakref.WeakSet())
    monkeypatch.setattr(_loop, "_forked_loops", weakref.WeakSet())


def _run(engine: CockroachDBEngine, coro: Coroutine[Any, Any, T], how: str) -> T:
    """Run coro through the sync dispatch, or the async one from another loop."""
    if how == "sync":
        return engine._run_as_sync(coro)
    return asyncio.run(engine._run_as_async(coro))


def _close(engine: CockroachDBEngine, how: str) -> None:
    """Call the public close() or aclose()."""
    if how == "sync":
        engine.close()
    else:
        asyncio.run(engine.aclose())


@contextlib.contextmanager
def _no_unawaited_coroutines() -> Iterator[None]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
        gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]


@pytest.mark.timeout(10)
@MODES
@BUILDERS
def test_loop_engine_fails_fast_on_a_closed_loop(build, how) -> None:
    loop = asyncio.new_event_loop()
    loop.close()
    engine = build(_lazy_engine(), loop=loop)
    with _no_unawaited_coroutines(), pytest.raises(RuntimeError, match="loop= is closed"):
        _close(engine, how)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_fails_fast_when_the_loop_is_not_running(how, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop = asyncio.new_event_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    ran = threading.Event()

    async def work() -> None:
        ran.set()

    try:
        with (
            _no_unawaited_coroutines(),
            pytest.raises(RuntimeError, match="loop= is not running"),
        ):
            _run(engine, work(), how)
        # Nothing was left on the loop to run once it starts
        loop.run_until_complete(asyncio.sleep(0.05))
        assert not ran.is_set()
    finally:
        loop.close()


@pytest.mark.timeout(10)
@MODES
@BUILDERS
def test_loop_engine_fails_fast_in_a_forked_child(build, how, monkeypatch) -> None:
    # In a fork child the copied loop still reports running, but no thread runs
    # it there. A loop whose thread is stuck in a callback looks the same.
    loop, thread = _start_loop()
    release = threading.Event()
    loop.call_soon_threadsafe(release.wait, 30)
    engine = build(_lazy_engine(), loop=loop)
    parent_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    try:
        assert loop.is_running()
        with (
            _no_unawaited_coroutines(),
            pytest.raises(RuntimeError, match="forked child.*a loop started in this process"),
        ):
            _close(engine, how)
    finally:
        monkeypatch.undo()
        release.set()
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@BUILDERS
def test_loop_engine_refuses_a_loop_that_was_running_when_the_process_forked(
    build, monkeypatch
) -> None:
    # A child runs the at-fork hook before anything else. Its copy of a loop that
    # was running in the parent reports running but can never run there, so a new
    # engine built on it in the child would hang on its first call.
    _isolate_fork_state(monkeypatch)
    loop, thread = _start_loop()
    built_before_fork = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    built_before_fork._active = {"sync": 1, "async": 1}
    pool_before_fork = built_before_fork.engine.sync_engine.pool
    try:
        engine_module._after_fork_in_child()
        # The hook still does its earlier work: fresh counters and a fresh pool
        assert built_before_fork._active == {"sync": 0, "async": 0}
        assert built_before_fork.engine.sync_engine.pool is not pool_before_fork
        with pytest.raises(
            RuntimeError,
            match=r"running in the parent process.*a loop started in this process, or without loop=",
        ):
            build(_lazy_engine(), loop=loop)
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_refuses_calls_on_a_loop_that_was_running_when_the_process_forked(
    how, monkeypatch
) -> None:
    # In a real child the pid check refuses this engine first. The pid matches in
    # this test, so this shows the forked-loop check in the call path on its own.
    _isolate_fork_state(monkeypatch)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    try:
        engine_module._after_fork_in_child()
        with (
            _no_unawaited_coroutines(),
            pytest.raises(RuntimeError, match="running in the parent process"),
        ):
            _close(engine, how)
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_accepts_a_loop_that_was_not_running_when_the_process_forked(
    how, monkeypatch
) -> None:
    # Only loops that were running at the fork are recorded; this one never ran
    _isolate_fork_state(monkeypatch)
    loop = asyncio.new_event_loop()
    _built_before_fork = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    engine_module._after_fork_in_child()
    _, thread = _start_loop(loop)
    try:
        engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
        assert _run(engine, asyncio.sleep(0, result="ok"), how) == "ok"
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
def test_forked_loop_record_does_not_keep_loops_alive(monkeypatch) -> None:
    _isolate_fork_state(monkeypatch)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    engine_module._after_fork_in_child()
    assert loop in _loop._forked_loops
    _close_loop(loop, thread)
    gone = weakref.ref(loop)
    del engine, loop
    gc.collect()
    assert gone() is None
    assert len(_loop._forked_loops) == 0


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_reports_a_loop_that_closed_just_before_the_handoff(how, monkeypatch) -> None:
    # The loop closes after the running check passed but before the work is
    # handed over. Skipping the check reproduces that order.
    loop = asyncio.new_event_loop()
    loop.close()
    monkeypatch.setattr(_loop, "_until_running", lambda loop: iter(()))
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    with _no_unawaited_coroutines(), pytest.raises(RuntimeError, match="loop= is closed"):
        _close(engine, how)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_fails_when_the_loop_stops_during_a_call(how, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    started, cancelled = threading.Event(), threading.Event()

    async def slow() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def stop_once_started() -> None:
        if started.wait(5):
            loop.call_soon_threadsafe(loop.stop)

    stopper = threading.Thread(target=stop_once_started, daemon=True)
    stopper.start()
    try:
        with pytest.raises(RuntimeError, match="stopped before the call finished"):
            _run(engine, slow(), how)
        # The call cancelled its work, so the work does not go on if the loop runs again
        thread.join(5)
        loop.run_until_complete(asyncio.sleep(0.1))
        assert cancelled.is_set()
    finally:
        stopper.join(5)
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_fails_when_the_loop_closes_during_a_call(how, monkeypatch, caplog) -> None:
    # A long grace for a loop that is not running, so the call sees it closed
    _fast_checks(monkeypatch, start_timeout=5.0)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    started = threading.Event()

    async def slow() -> None:
        started.set()
        await asyncio.sleep(30)

    def close_once_started() -> None:
        if started.wait(5):
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()

    closer = threading.Thread(target=close_once_started, daemon=True)
    closer.start()
    try:
        with pytest.raises(RuntimeError, match="was closed before the call finished"):
            _run(engine, slow(), how)
        # The closed loop's pending task is collected here, not during a later test
        gc.collect()
    finally:
        closer.join(5)
        _close_loop(loop, thread)
    # asyncio reports the task the closed loop left pending, and nothing else may
    # log an error. Cancelling that task on the closed loop would make
    # concurrent.futures log one.
    errors = [r.getMessage().splitlines()[0] for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == ["Task was destroyed but it is pending!"]


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_closes_work_that_a_closed_loop_never_started(how, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=5.0)
    loop = asyncio.new_event_loop()
    stuck, submitted, release = threading.Event(), threading.Event(), threading.Event()

    def stuck_then_stop() -> None:
        # Hold the loop until the call has handed over its work, then stop the
        # loop before that work gets a turn
        stuck.set()
        release.wait(5)
        loop.stop()

    loop.call_soon(stuck_then_stop)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)

    real_submit = asyncio.run_coroutine_threadsafe

    def submit(coro, target_loop):  # type: ignore[no-untyped-def]
        future = real_submit(coro, target_loop)
        submitted.set()
        return future

    def close_after_submit() -> None:
        if submitted.wait(5):
            release.set()
            thread.join(5)
            loop.close()

    async def work() -> None:
        pass

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    closer = threading.Thread(target=close_after_submit, daemon=True)
    closer.start()
    try:
        assert stuck.wait(5)
        with (
            _no_unawaited_coroutines(),
            pytest.raises(RuntimeError, match="was closed before the call finished"),
        ):
            _run(engine, work(), how)
    finally:
        release.set()
        closer.join(5)
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_waits_for_a_loop_that_is_starting(how, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=5.0)
    loop = asyncio.new_event_loop()

    def start_late() -> None:
        time.sleep(0.1)
        loop.run_forever()

    thread = threading.Thread(target=start_late, daemon=True)
    thread.start()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)

    async def which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    try:
        assert _run(engine, which_loop(), how) is loop
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_does_not_hold_up_a_call_while_its_loop_starts(how, monkeypatch) -> None:
    # Checks are 3 s apart here, so waiting a full interval before looking again
    # would hold up a call to a loop that starts after 20 ms by 3 s
    monkeypatch.setattr(_loop, "LOOP_CHECK_INTERVAL", 3.0)
    monkeypatch.setattr(_loop, "LOOP_START_TIMEOUT", 5.0)
    loop = asyncio.new_event_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)

    def start_soon() -> None:
        time.sleep(0.02)
        loop.run_forever()

    thread = threading.Thread(target=start_soon, daemon=True)
    thread.start()
    try:
        began = time.monotonic()
        assert _run(engine, asyncio.sleep(0, result="ok"), how) == "ok"
        assert time.monotonic() - began < 1.0
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
@MODES
def test_loop_engine_keeps_waiting_while_a_stopped_loop_starts_again(how, monkeypatch) -> None:
    _fast_checks(monkeypatch, start_timeout=5.0)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    started = threading.Event()
    threads = [thread]

    async def slow() -> str:
        started.set()
        await asyncio.sleep(0.3)
        return "done"

    def stop_and_restart() -> None:
        if started.wait(5):
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            time.sleep(0.2)  # several checks see the loop stopped
            again = threading.Thread(target=loop.run_forever, daemon=True)
            threads.append(again)
            again.start()

    restarter = threading.Thread(target=stop_and_restart, daemon=True)
    restarter.start()
    try:
        assert _run(engine, slow(), how) == "done"
    finally:
        restarter.join(5)
        _close_loop(loop, threads[-1])


@pytest.mark.timeout(10)
def test_loop_engine_waits_out_a_busy_loop(monkeypatch) -> None:
    # Busy for longer than a stopped loop's grace, which must not count as stopped
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    loop.call_soon_threadsafe(time.sleep, 0.5)
    try:
        assert engine._run_as_sync(asyncio.sleep(0, result="done")) == "done"
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
def test_loop_engine_refuses_a_sync_call_from_its_own_loop() -> None:
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)

    async def nested() -> None:
        engine.close()

    try:
        with _no_unawaited_coroutines(), pytest.raises(RuntimeError, match="deadlock"):
            asyncio.run_coroutine_threadsafe(nested(), loop).result(timeout=5)
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(30)
def test_loop_engine_serves_many_threads_without_false_alarms(monkeypatch) -> None:
    # Frequent checks and a short grace would expose a check that misjudges a
    # healthy loop
    _fast_checks(monkeypatch, start_timeout=0.2)
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)

    async def work(n: int) -> tuple[int, asyncio.AbstractEventLoop]:
        await asyncio.sleep(0.1)  # several check intervals
        return n, asyncio.get_running_loop()

    results: dict[int, asyncio.AbstractEventLoop] = {}
    errors: list[BaseException] = []

    def sync_caller(i: int) -> None:
        try:
            for j in range(3):
                n, ran_on = engine._run_as_sync(work(i * 10 + j))
                results[n] = ran_on
        except BaseException as exc:
            errors.append(exc)

    def async_caller(i: int) -> None:
        async def calls() -> list[tuple[int, asyncio.AbstractEventLoop]]:
            return await asyncio.gather(*(engine._run_as_async(work(i * 10 + j)) for j in range(3)))

        try:
            results.update(asyncio.run(calls()))
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=sync_caller if i % 2 else async_caller, args=(i,), daemon=True)
        for i in range(16)
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert errors == []
        assert sorted(results) == [i * 10 + j for i in range(16) for j in range(3)]
        assert all(ran_on is loop for ran_on in results.values())
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
def test_loop_engine_interrupted_sync_call_cancels_its_work(monkeypatch) -> None:
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    started, cancelled = threading.Event(), threading.Event()

    async def slow() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def interrupted_wait(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert started.wait(5), "work never started"
        raise KeyboardInterrupt("simulated")

    monkeypatch.setattr(concurrent.futures, "wait", interrupted_wait)
    try:
        with pytest.raises(KeyboardInterrupt):
            engine._run_as_sync(slow())
        assert cancelled.wait(5), "the work kept running after the caller was interrupted"
    finally:
        _close_loop(loop, thread)


@pytest.mark.timeout(10)
def test_loop_engine_cancelled_async_call_cancels_its_work() -> None:
    loop, thread = _start_loop()
    engine = CockroachDBEngine.from_engine(_lazy_engine(), loop=loop)
    started, cancelled = threading.Event(), threading.Event()

    async def slow() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def cancel_once_started() -> None:
        call = asyncio.ensure_future(engine._run_as_async(slow()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    try:
        asyncio.run(cancel_once_started())
        assert cancelled.wait(5), "the work kept running after the caller was cancelled"
    finally:
        _close_loop(loop, thread)
