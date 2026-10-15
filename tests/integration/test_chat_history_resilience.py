"""Chat history: engine sharing, atomic ordered writes, idempotency and real faults."""

import asyncio
import threading
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_cockroachdb import CockroachDBChatMessageHistory, RetryPolicy
from tests.integration.faults import FaultInjector, kill_sessions

FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)
DDL_OFF = {"options": "-c autocommit_before_ddl=off -c create_table_with_schema_locked=on"}


def new_table() -> str:
    return f"chat_{uuid.uuid4().hex[:8]}"


def test_shares_the_engine_and_its_policy(engine_factory) -> None:
    engine = engine_factory(retry_policy=RetryPolicy(max_attempts=7))
    history = CockroachDBChatMessageHistory("s1", engine=engine, table_name=new_table())
    assert history.engine is engine.engine
    assert history.retry_policy is engine.retry_policy
    custom = CockroachDBChatMessageHistory(
        "s1", engine=engine, table_name=new_table(), retry_policy=FAST
    )
    assert custom.retry_policy is FAST


async def test_accepts_a_raw_async_engine(engine_factory) -> None:
    engine = engine_factory()
    history = CockroachDBChatMessageHistory("s1", engine=engine.engine, table_name=new_table())
    await history.acreate_table_if_not_exists()
    await history.aadd_messages([HumanMessage("hi")])
    assert [m.content for m in await history.aget_messages()] == ["hi"]


async def test_add_messages_is_atomic_and_ordered(engine_factory, unsafe_internals) -> None:
    engine = engine_factory(retry_policy=FAST)
    history = CockroachDBChatMessageHistory("s1", engine=engine, table_name=new_table())
    await history.acreate_table_if_not_exists()
    messages = [HumanMessage(f"m{i}") if i % 2 == 0 else AIMessage(f"m{i}") for i in range(50)]
    await history.aadd_messages(messages)
    assert [m.content for m in await history.aget_messages()] == [f"m{i}" for i in range(50)]
    faults = FaultInjector(engine.engine, unsafe_internals=unsafe_internals)
    try:
        faults.fail_next("XXA00", match="insert into")
        with pytest.raises(Exception):  # noqa: B017
            await history.aadd_messages([HumanMessage("x"), AIMessage("y")])
    finally:
        faults.close()
    assert len(await history.aget_messages()) == 50


async def test_a_replayed_insert_is_a_no_op(engine_factory) -> None:
    engine = engine_factory()
    history = CockroachDBChatMessageHistory("s1", engine=engine, table_name=new_table())
    await history.acreate_table_if_not_exists()
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    batch = [HumanMessage("q"), AIMessage("a")]
    await history._insert_messages(batch, ids)
    await history._insert_messages(batch, ids)  # what a retry after an ambiguous commit does
    assert [m.content for m in await history.aget_messages()] == ["q", "a"]


@pytest.mark.parametrize("code", ["40001", "40003", "57P01", "08006"])
async def test_writes_reads_and_clears_recover(engine_factory, unsafe_internals, code: str) -> None:
    engine = engine_factory(retry_policy=FAST)
    history = CockroachDBChatMessageHistory("s1", engine=engine, table_name=new_table())
    await history.acreate_table_if_not_exists()
    faults = FaultInjector(engine.engine, unsafe_internals=unsafe_internals)
    try:
        faults.fail_next(code, match="insert into")
        await history.aadd_messages([HumanMessage("q"), AIMessage("a")])
        faults.fail_next(code, match="select message from")
        assert [m.content for m in await history.aget_messages()] == ["q", "a"]
        faults.fail_next(code, match="delete from")
        await history.aclear()
        assert await history.aget_messages() == []
        assert faults.injected == [code, code, code]
    finally:
        faults.close()


async def test_recovers_from_killed_connections(engine_factory, libpq_url) -> None:
    app = f"chat_kill_{uuid.uuid4().hex[:6]}"
    engine = engine_factory(app, pool_pre_ping=False, retry_policy=FAST)
    history = CockroachDBChatMessageHistory("s1", engine=engine, table_name=new_table())
    await history.acreate_table_if_not_exists()
    await history.aadd_messages([HumanMessage("before")])
    assert kill_sessions(libpq_url, app) >= 1
    await asyncio.sleep(0.3)
    assert [m.content for m in await history.aget_messages()] == ["before"]


def test_sync_methods_from_many_threads(engine_factory) -> None:
    engine = engine_factory()
    name = new_table()
    CockroachDBChatMessageHistory(
        "setup", engine=engine, table_name=name
    ).create_table_if_not_exists()
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        history = CockroachDBChatMessageHistory(f"s{i}", engine=engine, table_name=name)
        try:
            for j in range(5):
                history.add_messages([HumanMessage(f"{i}-{j}")])
            assert [m.content for m in history.messages] == [f"{i}-{j}" for j in range(5)]
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert not any(thread.is_alive() for thread in threads), "a worker thread did not finish"
    assert errors == []


def test_owned_engine_closes_once(connection_string) -> None:
    history = CockroachDBChatMessageHistory(
        "s1", connection_string=connection_string, table_name=new_table()
    )
    history.create_table_if_not_exists()
    history.add_message(HumanMessage("x"))
    assert [m.content for m in history.messages] == ["x"]
    history.close()
    history.close()


def test_legacy_private_create_still_works(connection_string) -> None:
    history = CockroachDBChatMessageHistory(
        "s1", connection_string=connection_string, table_name=new_table()
    )
    asyncio.run(history._acreate_table_if_not_exists())
    asyncio.run(history.aclose())


def test_unsafe_table_names_are_rejected(connection_string) -> None:
    with pytest.raises(ValueError):
        CockroachDBChatMessageHistory(
            "s1", connection_string=connection_string, table_name="x; drop"
        )


async def test_repeat_table_setup_works_with_autocommit_before_ddl_off(engine_factory) -> None:
    engine = engine_factory(connect_args=DDL_OFF)
    history = CockroachDBChatMessageHistory("s1", engine=engine, table_name=new_table())
    await history.acreate_table_if_not_exists()
    await history.acreate_table_if_not_exists()
    await history.aadd_messages([HumanMessage("hi")])
    assert [m.content for m in await history.aget_messages()] == ["hi"]
