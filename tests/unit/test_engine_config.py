"""Engine configuration and loop dispatch tests that need no database."""

import asyncio
import contextlib
import logging
import threading

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

import langchain_cockroachdb.engine as engine_module
from langchain_cockroachdb import CockroachDBEngine, RetryPolicy
from langchain_cockroachdb._loop import get_background_loop

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
