"""Engine configuration and loop dispatch tests that need no database."""

import asyncio
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
    assert captured["kwargs"]["connect_args"] == {"connect_timeout": 10}
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
    assert [kw["connect_args"] for kw in seen] == [{}, {"connect_timeout": 3}, {}]


def test_flat_retry_arguments_build_the_policy() -> None:
    engine = CockroachDBEngine.from_connection_string(URL, retry_max_attempts=10, retry_jitter=False)
    assert engine.retry_policy.max_attempts == 10
    assert engine.retry_max_attempts == 10
    assert engine.retry_jitter is False
    assert engine.retry_initial_backoff == 0.1


def test_policy_and_flat_arguments_conflict() -> None:
    with pytest.raises(ValueError):
        CockroachDBEngine.from_connection_string(URL, retry_policy=RetryPolicy(), retry_max_attempts=3)


def test_sync_runner_serves_many_threads() -> None:
    engine = CockroachDBEngine.from_connection_string(URL)

    async def work(n: int) -> int:
        await asyncio.sleep(0.01)
        return n

    results: list[int | None] = [None] * 16

    def call(i: int) -> None:
        results[i] = engine._run_as_sync(work(i))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(16)]
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


def test_explicit_loop_receives_sync_and_async_work() -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    engine = CockroachDBEngine.from_engine(create_async_engine("cockroachdb+psycopg://root@localhost:1/db"), loop=loop)

    async def which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    try:
        assert engine._run_as_sync(which_loop()) is loop
        assert asyncio.run(engine._run_as_async(which_loop())) is loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def test_mixed_sync_and_async_use_warns_once(caplog) -> None:
    engine = CockroachDBEngine.from_connection_string(URL)
    started, release = threading.Event(), threading.Event()

    async def slow() -> None:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)

    worker = threading.Thread(target=lambda: engine._run_as_sync(slow()))
    worker.start()
    assert started.wait(5)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.engine"):
        asyncio.run(engine._run_as_async(asyncio.sleep(0)))
        asyncio.run(engine._run_as_async(asyncio.sleep(0)))
    release.set()
    worker.join(5)
    assert caplog.text.count("one style per engine") == 1
