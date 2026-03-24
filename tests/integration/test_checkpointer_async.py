"""Async checkpointer integration tests for CockroachDB.

Modeled on langgraph-checkpoint-postgres test_async.py.
Tests AsyncCockroachDBSaver with connection and pool modes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langchain_cockroachdb.checkpointer.async_saver import (
    AsyncCockroachDBSaver,
    _sanitize_conn_string,
)


def _pg_uri(connection_string: str) -> str:
    return _sanitize_conn_string(connection_string)


@asynccontextmanager
async def _base_saver(connection_string: str):
    """Fixture for regular async connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    pg_uri = _pg_uri(connection_string)
    base_uri = pg_uri.rsplit("/", 1)[0]

    async with await AsyncConnection.connect(pg_uri, autocommit=True) as conn:
        await conn.execute(f"CREATE DATABASE {database}")

    db_uri = (
        f"{base_uri}/{database}?sslmode=disable"
        if "sslmode=disable" in pg_uri
        else f"{base_uri}/{database}?sslmode=verify-full"
    )
    try:
        async with await AsyncConnection.connect(
            db_uri,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncCockroachDBSaver(conn)
            await checkpointer.setup()
            yield checkpointer
    finally:
        async with await AsyncConnection.connect(pg_uri, autocommit=True) as conn:
            await conn.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


@asynccontextmanager
async def _pool_saver(connection_string: str):
    """Fixture for async pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    pg_uri = _pg_uri(connection_string)
    base_uri = pg_uri.rsplit("/", 1)[0]

    async with await AsyncConnection.connect(pg_uri, autocommit=True) as conn:
        await conn.execute(f"CREATE DATABASE {database}")

    db_uri = (
        f"{base_uri}/{database}?sslmode=disable"
        if "sslmode=disable" in pg_uri
        else f"{base_uri}/{database}?sslmode=verify-full"
    )
    try:
        async with AsyncConnectionPool(
            db_uri,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
        ) as pool:
            checkpointer = AsyncCockroachDBSaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        async with await AsyncConnection.connect(pg_uri, autocommit=True) as conn:
            await conn.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


@asynccontextmanager
async def _saver(name: str, connection_string: str):
    if name == "base":
        async with _base_saver(connection_string) as saver:
            yield saver
    elif name == "pool":
        async with _pool_saver(connection_string) as saver:
            yield saver


@asynccontextmanager
async def _from_conn_string_saver(connection_string: str):
    """Test the from_conn_string factory method."""
    database = f"test_{uuid4().hex[:16]}"
    pg_uri = _pg_uri(connection_string)
    base_uri = pg_uri.rsplit("/", 1)[0]

    async with await AsyncConnection.connect(pg_uri, autocommit=True) as conn:
        await conn.execute(f"CREATE DATABASE {database}")

    db_uri = (
        f"{base_uri}/{database}?sslmode=disable"
        if "sslmode=disable" in pg_uri
        else f"{base_uri}/{database}?sslmode=verify-full"
    )
    try:
        crdb_uri = db_uri.replace("postgresql://", "cockroachdb://", 1)
        async with AsyncCockroachDBSaver.from_conn_string(crdb_uri) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        async with await AsyncConnection.connect(pg_uri, autocommit=True) as conn:
            await conn.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


class TestAsyncCockroachDBSaver:
    """Test async AsyncCockroachDBSaver."""

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_put_get(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
            }
            chkpnt = empty_checkpoint()
            config = await saver.aput(config, chkpnt, {"source": "input", "step": 2}, {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == chkpnt["id"]
            assert result.metadata["source"] == "input"
            assert result.metadata["step"] == 2

            # Also test get latest by thread_id
            latest: RunnableConfig = {
                "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
            }
            result_latest = await saver.aget_tuple(latest)
            assert result_latest is not None
            assert result_latest.checkpoint["id"] == chkpnt["id"]

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_search(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            c1: RunnableConfig = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            c2: RunnableConfig = {"configurable": {"thread_id": "thread-2", "checkpoint_ns": ""}}
            c3: RunnableConfig = {
                "configurable": {"thread_id": "thread-2", "checkpoint_ns": "inner"}
            }
            await saver.aput(c1, empty_checkpoint(), {"source": "input", "step": 2}, {})
            await saver.aput(c2, empty_checkpoint(), {"source": "loop", "step": 1}, {})
            await saver.aput(c3, empty_checkpoint(), {}, {})

            results = [c async for c in saver.alist(None, filter={"source": "input"})]
            assert len(results) == 1

            results = [c async for c in saver.alist(None, filter={"step": 1})]
            assert len(results) == 1

            results = [c async for c in saver.alist(None, filter={})]
            assert len(results) == 3

            results = [c async for c in saver.alist(None, filter={"source": "update", "step": 1})]
            assert len(results) == 0

            results = [c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})]
            assert len(results) == 2
            namespaces = {r.config["configurable"]["checkpoint_ns"] for r in results}
            assert namespaces == {"", "inner"}

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_list_limit(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            c1: RunnableConfig = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            c2: RunnableConfig = {"configurable": {"thread_id": "thread-2", "checkpoint_ns": ""}}
            await saver.aput(c1, empty_checkpoint(), {}, {})
            await saver.aput(c2, empty_checkpoint(), {}, {})

            results = [c async for c in saver.alist(None, limit=1)]
            assert len(results) == 1

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_put_writes(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-writes",
                    "checkpoint_ns": "",
                }
            }
            config = await saver.aput(config, empty_checkpoint(), {}, {})
            await saver.aput_writes(config, [("channel1", "value1")], task_id="task-1")

            result = await saver.aget_tuple(config)
            assert result is not None
            assert len(result.pending_writes) == 1
            assert result.pending_writes[0][1] == "channel1"
            assert result.pending_writes[0][2] == "value1"

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_delete_thread(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-delete",
                    "checkpoint_ns": "",
                }
            }
            config = await saver.aput(config, empty_checkpoint(), {"source": "input"}, {})

            result = await saver.aget_tuple(config)
            assert result is not None

            await saver.adelete_thread("thread-delete")

            lookup: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-delete",
                    "checkpoint_ns": "",
                }
            }
            result = await saver.aget_tuple(lookup)
            assert result is None

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_get_nonexistent(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "nonexistent",
                    "checkpoint_ns": "",
                }
            }
            result = await saver.aget_tuple(config)
            assert result is None

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_parent_checkpoint(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-parent",
                    "checkpoint_ns": "",
                }
            }
            chkpnt_1 = empty_checkpoint()
            config = await saver.aput(config, chkpnt_1, {"step": 1}, {})

            chkpnt_2 = create_checkpoint(chkpnt_1, {}, 1)
            config = await saver.aput(config, chkpnt_2, {"step": 2}, {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.parent_config is not None
            assert result.parent_config["configurable"]["checkpoint_id"] == chkpnt_1["id"]

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_null_chars(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-null",
                    "checkpoint_ns": "",
                }
            }
            config = await saver.aput(config, empty_checkpoint(), {"my_key": "\x00abc"}, {})
            result = await saver.aget_tuple(config)
            assert result.metadata["my_key"] == "abc"

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_combined_metadata(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-meta",
                    "checkpoint_ns": "",
                },
                "metadata": {"run_id": "my_run_id"},
            }
            chkpnt = create_checkpoint(empty_checkpoint(), {}, 1)
            await saver.aput(config, chkpnt, {"source": "loop", "step": 1}, {})

            lookup: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-meta",
                    "checkpoint_ns": "",
                }
            }
            result = await saver.aget_tuple(lookup)
            assert result.metadata["source"] == "loop"
            assert result.metadata["run_id"] == "my_run_id"

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_pending_sends_migration(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-sends",
                    "checkpoint_ns": "",
                }
            }
            checkpoint_0 = empty_checkpoint()
            config = await saver.aput(config, checkpoint_0, {}, {})
            await saver.aput_writes(
                config,
                [(TASKS, "send-1"), (TASKS, "send-2")],
                task_id="task-1",
            )
            await saver.aput_writes(config, [(TASKS, "send-3")], task_id="task-2")

            tuple_0 = await saver.aget_tuple(config)
            assert tuple_0.checkpoint["channel_values"] == {}

            checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
            config = await saver.aput(config, checkpoint_1, {}, {})

            tuple_1 = await saver.aget_tuple(config)
            assert tuple_1.checkpoint["channel_values"] == {TASKS: ["send-1", "send-2", "send-3"]}
            assert TASKS in tuple_1.checkpoint["channel_versions"]

    async def test_from_conn_string(self, connection_string: str) -> None:
        async with _from_conn_string_saver(connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-factory",
                    "checkpoint_ns": "",
                }
            }
            chkpnt = empty_checkpoint()
            config = await saver.aput(config, chkpnt, {"source": "input"}, {})
            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == chkpnt["id"]

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_idempotent_setup(self, connection_string: str, saver_name: str) -> None:
        async with _saver(saver_name, connection_string) as saver:
            await saver.setup()
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-idempotent",
                    "checkpoint_ns": "",
                }
            }
            config = await saver.aput(config, empty_checkpoint(), {}, {})
            result = await saver.aget_tuple(config)
            assert result is not None

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_batch_list(self, connection_string: str, saver_name: str) -> None:
        """Test that alist() returns correct blobs/writes for multiple checkpoints."""
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-batch", "checkpoint_ns": ""}
            }
            chkpnt_1 = empty_checkpoint()
            config = await saver.aput(config, chkpnt_1, {"step": 1}, {})
            await saver.aput_writes(config, [("ch1", "val1")], task_id="task-1")

            chkpnt_2 = create_checkpoint(chkpnt_1, {}, 1)
            config = await saver.aput(config, chkpnt_2, {"step": 2}, {})
            await saver.aput_writes(config, [("ch2", "val2")], task_id="task-2")

            chkpnt_3 = create_checkpoint(chkpnt_2, {}, 2)
            config = await saver.aput(config, chkpnt_3, {"step": 3}, {})

            results = [
                c
                async for c in saver.alist(
                    {"configurable": {"thread_id": "thread-batch", "checkpoint_ns": ""}}
                )
            ]
            assert len(results) == 3

            assert results[0].metadata["step"] == 3
            assert results[1].metadata["step"] == 2
            assert results[2].metadata["step"] == 1

            assert len(results[1].pending_writes) == 1
            assert results[1].pending_writes[0][1] == "ch2"
            assert results[1].pending_writes[0][2] == "val2"

            assert len(results[2].pending_writes) == 1
            assert results[2].pending_writes[0][1] == "ch1"
            assert results[2].pending_writes[0][2] == "val1"

            assert len(results[0].pending_writes) == 0

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_enable_disable_ttl(self, connection_string: str, saver_name: str) -> None:
        """Test that aenable_ttl() and adisable_ttl() execute without errors."""
        async with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-ttl", "checkpoint_ns": ""}
            }
            await saver.aput(config, empty_checkpoint(), {"source": "input"}, {})

            await saver.aenable_ttl(ttl_interval="30 days", cron="@daily")

            result = await saver.aget_tuple(
                {"configurable": {"thread_id": "thread-ttl", "checkpoint_ns": ""}}
            )
            assert result is not None
            assert result.metadata["source"] == "input"

            await saver.adisable_ttl()

            result = await saver.aget_tuple(
                {"configurable": {"thread_id": "thread-ttl", "checkpoint_ns": ""}}
            )
            assert result is not None

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    async def test_ttl_idempotent(self, connection_string: str, saver_name: str) -> None:
        """Test that aenable_ttl() can be called multiple times."""
        async with _saver(saver_name, connection_string) as saver:
            await saver.aenable_ttl(ttl_interval="7 days")
            await saver.aenable_ttl(ttl_interval="14 days")
            await saver.adisable_ttl()
            await saver.adisable_ttl()

    async def test_ttl_expiration(self, connection_string: str) -> None:
        """Test that rows are actually deleted after TTL expires."""
        async with _base_saver(connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-ttl-expire", "checkpoint_ns": ""}
            }
            config = await saver.aput(config, empty_checkpoint(), {"source": "ttl-test"}, {})
            await saver.aput_writes(config, [("ch1", "val1")], task_id="task-1")

            result = await saver.aget_tuple(
                {"configurable": {"thread_id": "thread-ttl-expire", "checkpoint_ns": ""}}
            )
            assert result is not None

            # Backdate created_at to make rows immediately expired
            async with saver._cursor() as cur:
                await cur.execute(
                    "UPDATE checkpoints SET created_at = now() - INTERVAL '1 hour' "
                    "WHERE thread_id = 'thread-ttl-expire'"
                )
                await cur.execute(
                    "UPDATE checkpoint_blobs SET created_at = now() - INTERVAL '1 hour' "
                    "WHERE thread_id = 'thread-ttl-expire'"
                )
                await cur.execute(
                    "UPDATE checkpoint_writes SET created_at = now() - INTERVAL '1 hour' "
                    "WHERE thread_id = 'thread-ttl-expire'"
                )

            await saver.aenable_ttl(ttl_interval="1 second", cron="* * * * *")

            deleted = False
            deadline = asyncio.get_event_loop().time() + 300
            while asyncio.get_event_loop().time() < deadline:
                result = await saver.aget_tuple(
                    {"configurable": {"thread_id": "thread-ttl-expire", "checkpoint_ns": ""}}
                )
                if result is None:
                    deleted = True
                    break
                await asyncio.sleep(5)

            assert deleted, (
                "TTL did not delete expired rows within 5 minutes. "
                "The CockroachDB TTL background job may not have run yet."
            )

            await saver.adisable_ttl()
