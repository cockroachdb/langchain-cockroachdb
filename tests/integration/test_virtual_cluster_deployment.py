"""Opt-in CRUD and isolation checks against two provisioned virtual clusters.

Use disposable application targets. The tests do not provision clusters,
change grants or network access, or require a testcontainer.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AsyncExitStack, ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple, empty_checkpoint

from langchain_cockroachdb import (
    AsyncCockroachDBSaver,
    AsyncCockroachDBVectorStore,
    CockroachDBChatMessageHistory,
    CockroachDBEngine,
    CockroachDBSaver,
)
from langchain_cockroachdb.retry import get_sqlstate

pytestmark = pytest.mark.timeout(180)


@dataclass(frozen=True)
class VirtualClusterTargets:
    # Fixture representations in pytest failures must not expose credentials.
    primary: str = field(repr=False)
    peer: str = field(repr=False)


@contextmanager
def database_errors_without_connection_details() -> Iterator[None]:
    """Driver errors can include credential-bearing connection arguments."""
    try:
        yield
    except AssertionError:
        raise
    except Exception as error:
        raise pytest.fail.Exception(
            f"Virtual-cluster operation failed: {type(error).__name__}; "
            f"SQLSTATE={get_sqlstate(error) or 'unavailable'}. "
            "Connection details omitted.",
            pytrace=False,
        ) from None


def hide_connection_errors(test: Callable[..., Any]) -> Callable[..., Any]:
    """Keep live URLs out of driver tracebacks while preserving data assertions."""
    if iscoroutinefunction(test):

        @wraps(test)
        async def async_test(*args: Any, **kwargs: Any) -> Any:
            with database_errors_without_connection_details():
                return await test(*args, **kwargs)

        return async_test

    @wraps(test)
    def sync_test(*args: Any, **kwargs: Any) -> Any:
        with database_errors_without_connection_details():
            return test(*args, **kwargs)

    return sync_test


@pytest.fixture(scope="module")
def virtual_cluster_targets() -> VirtualClusterTargets:
    __tracebackhide__ = True
    primary = os.environ.get("COCKROACHDB_VIRTUAL_CLUSTER_URL")
    peer = os.environ.get("COCKROACHDB_VIRTUAL_CLUSTER_PEER_URL")
    if not primary or not peer:
        pytest.skip(
            "Set COCKROACHDB_VIRTUAL_CLUSTER_URL and COCKROACHDB_VIRTUAL_CLUSTER_PEER_URL "
            "to two disposable application virtual clusters"
        )
    if primary == peer:
        pytest.fail("Virtual-cluster isolation checks require two distinct targets")
    return VirtualClusterTargets(primary, peer)


@pytest_asyncio.fixture
async def virtual_cluster_engines(
    virtual_cluster_targets: VirtualClusterTargets,
) -> AsyncIterator[list[CockroachDBEngine]]:
    with database_errors_without_connection_details():
        engines = [
            CockroachDBEngine.from_connection_string(url, pool_size=1, max_overflow=0)
            for url in (virtual_cluster_targets.primary, virtual_cluster_targets.peer)
        ]
        try:
            yield engines
        finally:
            for engine in engines:
                await engine.aclose()


@hide_connection_errors
async def test_vector_crud_is_scoped_to_the_virtual_cluster(
    virtual_cluster_engines: list[CockroachDBEngine],
) -> None:
    table = f"vc_vectors_{uuid4().hex}"
    document_id = str(uuid4())
    stores = []
    try:
        for index, engine in enumerate(virtual_cluster_engines):
            await engine.ainit_vectorstore_table(table_name=table, vector_dimension=8)
            store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=8), table)
            stores.append(store)
            await store.aadd_texts([f"cluster {index}"], ids=[document_id])

        for index, store in enumerate(stores):
            assert [doc.page_content for doc in await store.aget_by_ids([document_id])] == [
                f"cluster {index}"
            ]
            assert [doc.page_content for doc in await store.asimilarity_search("cluster", k=1)] == [
                f"cluster {index}"
            ]

        await stores[0].adelete([document_id])
        assert await stores[0].aget_by_ids([document_id]) == []
        assert [doc.page_content for doc in await stores[1].aget_by_ids([document_id])] == [
            "cluster 1"
        ]
    finally:
        for engine in virtual_cluster_engines:
            await engine._run_ddl(
                [f"DROP TABLE IF EXISTS {table}"], label="cleanup virtual-cluster test"
            )


@hide_connection_errors
async def test_chat_history_is_scoped_to_the_virtual_cluster(
    virtual_cluster_engines: list[CockroachDBEngine],
) -> None:
    table = f"vc_chat_{uuid4().hex}"
    session_id = f"vc_session_{uuid4().hex}"
    histories = []
    try:
        for index, engine in enumerate(virtual_cluster_engines):
            history = CockroachDBChatMessageHistory(session_id, engine=engine, table_name=table)
            histories.append(history)
            await history.acreate_table_if_not_exists()
            await history.aadd_messages([HumanMessage(f"cluster {index}")])

        for index, history in enumerate(histories):
            assert [message.content for message in await history.aget_messages()] == [
                f"cluster {index}"
            ]

        await histories[0].aclear()
        assert await histories[0].aget_messages() == []
        assert [message.content for message in await histories[1].aget_messages()] == ["cluster 1"]
    finally:
        for engine in virtual_cluster_engines:
            await engine._run_ddl(
                [f"DROP TABLE IF EXISTS {table}"], label="cleanup virtual-cluster test"
            )


def checkpoint_with_value(base: Checkpoint, value: str) -> Checkpoint:
    return {
        **base,
        "channel_values": {"value": value},
        "channel_versions": {"value": "1"},
    }


def assert_checkpoint(saved: CheckpointTuple | None, value: str) -> None:
    assert saved is not None
    assert saved.checkpoint["channel_values"] == {"value": value}
    assert saved.pending_writes == [("shared_task", "note", value)]


@hide_connection_errors
def test_sync_checkpoints_are_scoped_to_the_virtual_cluster(
    virtual_cluster_targets: VirtualClusterTargets,
) -> None:
    thread_id = f"vc_checkpoint_{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    base = empty_checkpoint()
    with ExitStack() as stack:
        savers = [
            stack.enter_context(CockroachDBSaver.from_conn_string(url))
            for url in (virtual_cluster_targets.primary, virtual_cluster_targets.peer)
        ]
        for saver in savers:
            saver.setup()
        try:
            for index, saver in enumerate(savers):
                value = f"cluster {index}"
                checkpoint = checkpoint_with_value(base, value)
                saved = saver.put(config, checkpoint, {"step": 1}, {"value": "1"})
                saver.put_writes(saved, [("note", value)], "shared_task")
            for index, saver in enumerate(savers):
                assert_checkpoint(saver.get_tuple(config), f"cluster {index}")

            savers[0].delete_thread(thread_id)
            assert savers[0].get_tuple(config) is None
            assert_checkpoint(savers[1].get_tuple(config), "cluster 1")
        finally:
            for saver in savers:
                saver.delete_thread(thread_id)


@hide_connection_errors
async def test_async_checkpoints_are_scoped_to_the_virtual_cluster(
    virtual_cluster_targets: VirtualClusterTargets,
) -> None:
    thread_id = f"vc_checkpoint_{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    base = empty_checkpoint()
    async with AsyncExitStack() as stack:
        savers = [
            await stack.enter_async_context(AsyncCockroachDBSaver.from_conn_string(url))
            for url in (virtual_cluster_targets.primary, virtual_cluster_targets.peer)
        ]
        for saver in savers:
            await saver.setup()
        try:
            for index, saver in enumerate(savers):
                value = f"cluster {index}"
                checkpoint = checkpoint_with_value(base, value)
                saved = await saver.aput(config, checkpoint, {"step": 1}, {"value": "1"})
                await saver.aput_writes(saved, [("note", value)], "shared_task")
            for index, saver in enumerate(savers):
                assert_checkpoint(await saver.aget_tuple(config), f"cluster {index}")

            await savers[0].adelete_thread(thread_id)
            assert await savers[0].aget_tuple(config) is None
            assert_checkpoint(await savers[1].aget_tuple(config), "cluster 1")
        finally:
            for saver in savers:
                await saver.adelete_thread(thread_id)
