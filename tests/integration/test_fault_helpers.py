"""The fault helpers themselves must produce real server errors."""

import asyncio

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from langchain_cockroachdb import get_sqlstate
from tests.integration.faults import FaultInjector, PsycopgFaults, hold_row_lock, kill_sessions


async def test_fault_injector_raises_the_requested_sqlstate(
    engine_factory, unsafe_internals
) -> None:
    engine = engine_factory()
    faults = FaultInjector(engine.engine, unsafe_internals=unsafe_internals)
    faults.fail_next("58C01", match="select 42")
    with pytest.raises(DBAPIError) as info:
        async with engine.engine.connect() as conn:
            await conn.execute(text("SELECT 42"))
    assert get_sqlstate(info.value) == "58C01"
    async with engine.engine.connect() as conn:
        assert (await conn.execute(text("SELECT 42"))).scalar() == 42
    assert faults.count("select 42") == 2
    faults.close()


def test_psycopg_faults_raise_the_requested_sqlstate(libpq_url, unsafe_internals) -> None:
    faults = PsycopgFaults(unsafe_internals)
    with psycopg.connect(libpq_url, autocommit=True, cursor_factory=faults.cursor_class()) as conn:
        faults.fail_next("40003")
        with pytest.raises(psycopg.Error) as info:
            conn.execute("SELECT 1")
        assert get_sqlstate(info.value) == "40003"
        assert conn.execute("SELECT 1").fetchone()[0] == 1


async def test_kill_sessions_breaks_idle_pooled_connections(engine_factory, libpq_url) -> None:
    engine = engine_factory("helper_kill", pool_pre_ping=False, pool_size=1, max_overflow=0)
    async with engine.engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    assert kill_sessions(libpq_url, "helper_kill") >= 1
    await asyncio.sleep(0.3)
    with pytest.raises(DBAPIError) as info:
        async with engine.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    assert info.value.connection_invalidated


async def test_hold_row_lock_makes_writers_wait(engine_factory, libpq_url) -> None:
    engine = engine_factory()
    async with engine.engine.begin() as conn:
        await conn.execute(
            text("CREATE TABLE IF NOT EXISTS lock_probe (id INT PRIMARY KEY, v TEXT)")
        )
        await conn.execute(text("UPSERT INTO lock_probe VALUES (1, 'a')"))
    with hold_row_lock(libpq_url, "UPDATE lock_probe SET v = 'locked' WHERE id = %s", (1,)):
        with pytest.raises(DBAPIError) as info:
            async with engine.engine.connect() as conn:
                await conn.execute(text("SET statement_timeout = '300ms'"))
                await conn.execute(text("UPSERT INTO lock_probe VALUES (1, 'b')"))
        assert get_sqlstate(info.value) == "57014"
