"""DeltaChannel history matches LangGraph's default and takes few round trips.

LangGraph's default get_delta_channel_history walks the parent chain with one
get_tuple per ancestor. The savers fetch the chain in pages instead. These
tests compare both on every checkpoint of real graphs, including a forked
thread, a channel with no snapshot yet, and pages smaller than the chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import psycopg
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint

from langchain_cockroachdb.checkpointer.async_saver import AsyncCockroachDBSaver
from langchain_cockroachdb.checkpointer.saver import CockroachDBSaver

# The minimum-dependency job installs langgraph-checkpoint without langgraph,
# and DeltaChannel is only in recent langgraph releases.
pytest.importorskip("langgraph.graph")
DeltaChannel = pytest.importorskip("langgraph.channels.delta").DeltaChannel
from langgraph.graph import START, StateGraph  # noqa: E402

CHANNELS = ["items", "tags", "never_written"]


def _extend(state: list[str] | None, writes: Sequence[list[str]]) -> list[str]:
    result = list(state or [])
    for write in writes:
        result.extend(write)
    return result


class State(TypedDict, total=False):
    items: Annotated[list[str], DeltaChannel(_extend, snapshot_frequency=3)]
    tags: Annotated[list[str], DeltaChannel(_extend, snapshot_frequency=50)]


def _graph(saver: Any) -> Any:
    # Two parallel nodes write the same channel, so a step leaves several writes.
    def left(state: State) -> dict:
        return {"items": [f"L{len(state.get('items', []))}"]}

    def right(state: State) -> dict:
        return {"items": [f"R{len(state.get('items', []))}"], "tags": ["t"]}

    builder = StateGraph(State)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    return builder.compile(checkpointer=saver)


def _run(graph: Any, thread_id: str, turns: int) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    for _ in range(turns):
        graph.invoke({"items": ["in"]}, config)
    return config


@pytest.mark.parametrize("page_size", [256, 2])
def test_matches_the_default_on_every_checkpoint(libpq_url: str, page_size: int) -> None:
    thread_id = f"delta-{uuid4()}"
    with CockroachDBSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
        saver.DELTA_HISTORY_PAGE_SIZE = page_size
        try:
            graph = _graph(saver)
            config = _run(graph, thread_id, turns=8)
            history = list(saver.list(config))
            # Page size 2 makes the walk span several pages.
            assert len(history) > 2
            for checkpoint in history:
                expected = BaseCheckpointSaver.get_delta_channel_history(
                    saver, config=checkpoint.config, channels=CHANNELS
                )
                actual = saver.get_delta_channel_history(
                    config=checkpoint.config, channels=CHANNELS
                )
                assert actual == expected
            # The latest state is rebuilt from that history.
            assert len(graph.get_state(config).values["items"]) == 8 * 3
        finally:
            saver.delete_thread(thread_id)


def test_one_get_tuple_per_call_however_long_the_thread(libpq_url: str) -> None:
    thread_id = f"delta-{uuid4()}"
    with CockroachDBSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
        try:
            config = _run(_graph(saver), thread_id, turns=10)
            calls = 0
            get_tuple = saver.get_tuple

            def counting(config: Any) -> Any:
                nonlocal calls
                calls += 1
                return get_tuple(config)

            saver.get_tuple = counting  # type: ignore[method-assign]
            saver.get_delta_channel_history(config=config, channels=CHANNELS)
            # Only the target; the default makes one call per ancestor.
            assert calls == 1
        finally:
            saver.delete_thread(thread_id)


def test_a_fork_only_sees_ancestors_on_its_path(libpq_url: str) -> None:
    thread_id = f"delta-{uuid4()}"
    with CockroachDBSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
        try:
            graph = _graph(saver)
            config = _run(graph, thread_id, turns=4)
            older = list(saver.list(config))[5].config
            forked = graph.update_state(older, {"items": ["fork"]}, as_node="left")
            expected = BaseCheckpointSaver.get_delta_channel_history(
                saver, config=forked, channels=CHANNELS
            )
            assert saver.get_delta_channel_history(config=forked, channels=CHANNELS) == expected
        finally:
            saver.delete_thread(thread_id)


def test_reloaded_state_keeps_the_order_the_run_produced(libpq_url: str) -> None:
    # Each step's two parallel nodes write the same delta channel. LangGraph
    # applies a step's writes in (task_path, task_id, idx) order, so a reload
    # must replay them in that order too, not in task_id (hash) order.
    thread_id = f"delta-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        with CockroachDBSaver.from_conn_string(libpq_url) as saver:
            saver.setup()
            graph = _graph(saver)
            for _ in range(8):
                live = graph.invoke({"items": ["in"]}, config)
        with CockroachDBSaver.from_conn_string(libpq_url) as fresh:
            reloaded = _graph(fresh).get_state(config).values["items"]
        assert reloaded == live["items"]
    finally:
        with CockroachDBSaver.from_conn_string(libpq_url) as saver:
            saver.delete_thread(thread_id)


def test_pending_writes_follow_writes_sort_key(libpq_url: str) -> None:
    thread_id = f"delta-{uuid4()}"
    with CockroachDBSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
        try:
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = empty_checkpoint()
            saved = saver.put(config, checkpoint, {}, {})
            # Task ids sort the other way round from task paths.
            saver.put_writes(saved, [("items", ["b"])], task_id="a-task", task_path="~2")
            saver.put_writes(saved, [("items", ["a"])], task_id="z-task", task_path="~1")
            writes = saver.get_tuple(saved).pending_writes
            assert [w[0] for w in writes] == ["z-task", "a-task"]
            listed = next(iter(saver.list(saved, limit=1))).pending_writes
            assert [w[0] for w in listed] == ["z-task", "a-task"]
        finally:
            saver.delete_thread(thread_id)


def test_fewer_statements_than_the_default(libpq_url: str, monkeypatch: Any) -> None:
    thread_id = f"delta-{uuid4()}"
    with CockroachDBSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
        try:
            config = _run(_graph(saver), thread_id, turns=10)
            target = next(iter(saver.list(config, limit=1))).config
            statements = 0
            execute = psycopg.Cursor.execute

            def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal statements
                statements += 1
                return execute(self, *args, **kwargs)

            monkeypatch.setattr(psycopg.Cursor, "execute", counting)
            BaseCheckpointSaver.get_delta_channel_history(saver, config=target, channels=CHANNELS)
            default = statements
            statements = 0
            saver.get_delta_channel_history(config=target, channels=CHANNELS)
            monkeypatch.undo()
            # "tags" and "never_written" have no snapshot, so both walks reach the
            # root, about 30 ancestors. The default reads them one get_tuple (three
            # statements) at a time. The override reads the target, then two pages
            # (16, then up to 64 ancestors), four statements each.
            assert statements <= 3 + 2 * 4
            assert default >= 3 * 30
        finally:
            saver.delete_thread(thread_id)


def test_unknown_thread_and_no_channels(libpq_url: str) -> None:
    with CockroachDBSaver.from_conn_string(libpq_url) as saver:
        saver.setup()
        config = {"configurable": {"thread_id": f"delta-missing-{uuid4()}"}}
        assert saver.get_delta_channel_history(config=config, channels=["items"]) == {
            "items": {"writes": []}
        }
        assert saver.get_delta_channel_history(config=config, channels=[]) == {}


@pytest.mark.parametrize("page_size", [256, 2])
async def test_async_matches_the_default_on_every_checkpoint(
    libpq_url: str, page_size: int
) -> None:
    thread_id = f"delta-{uuid4()}"
    async with AsyncCockroachDBSaver.from_conn_string(libpq_url) as saver:
        await saver.setup()
        saver.DELTA_HISTORY_PAGE_SIZE = page_size
        try:
            graph = _graph(saver)
            config = {"configurable": {"thread_id": thread_id}}
            for _ in range(6):
                await graph.ainvoke({"items": ["in"]}, config)
            history = [c async for c in saver.alist(config)]
            # Page size 2 makes the walk span several pages.
            assert len(history) > 2
            for checkpoint in history:
                expected = await BaseCheckpointSaver.aget_delta_channel_history(
                    saver, config=checkpoint.config, channels=CHANNELS
                )
                actual = await saver.aget_delta_channel_history(
                    config=checkpoint.config, channels=CHANNELS
                )
                assert actual == expected
            assert len((await graph.aget_state(config)).values["items"]) == 6 * 3
        finally:
            await saver.adelete_thread(thread_id)
