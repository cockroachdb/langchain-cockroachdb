"""DeltaChannel history reads the whole chain whatever the page sizes.

The chain query always returns its start row. With DELTA_HISTORY_FIRST_PAGE
or DELTA_HISTORY_PAGE_SIZE at 0 or less, a page held that one row, fewer than
its limit, so the walk took it for the end of the chain and stopped after one
ancestor. Here an in-memory chain stands in for the database: get_tuple reads
one checkpoint, and _fetch_chain_page returns the rows the chain query would.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, empty_checkpoint
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver

# Older langgraph-checkpoint releases have no default walk to compare with
pytestmark = pytest.mark.skipif(
    not hasattr(BaseCheckpointSaver, "get_delta_channel_history"),
    reason="langgraph-checkpoint has no get_delta_channel_history",
)

CHANNELS = ["items", "tags"]
LENGTH = 12


def _config(checkpoint_id: str) -> Any:
    return {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }


def _id(config: Any) -> str:
    return config["configurable"]["checkpoint_id"]


def _chain() -> dict[str, CheckpointTuple]:
    """LENGTH checkpoints, each the parent of the next, each with pending writes.

    The fourth stores a value for "items", so the walk for it ends there.
    "tags" has no stored value, so its walk reaches the root.
    """
    chain: dict[str, CheckpointTuple] = {}
    for step in range(LENGTH):
        checkpoint = empty_checkpoint()
        if step == 3:
            checkpoint["channel_values"] = {"items": ["seed"]}
        chain[f"{step:04}"] = CheckpointTuple(
            config=_config(f"{step:04}"),
            checkpoint=checkpoint,
            metadata={"step": step},
            parent_config=_config(f"{step - 1:04}") if step else None,
            pending_writes=[
                ("task-a", "items", [f"a{step}"]),
                ("task-b", "items", [f"b{step}"]),
                ("task-b", "tags", [f"t{step}"]),
            ],
        )
    return chain


def _chain_page(chain: dict[str, CheckpointTuple], start: Any, limit: int) -> list[Any]:
    """What the chain query returns: the start row, then parents up to ``limit`` rows."""
    page = [chain[_id(start)]]
    while len(page) < limit and page[-1].parent_config is not None:
        page.append(chain[_id(page[-1].parent_config)])
    return page


@pytest.mark.parametrize("size", [0, -1])
def test_page_sizes_below_one_still_read_the_whole_chain(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _chain()
    saver = CockroachDBSaver(ConnectionPool("", open=False))
    saver.DELTA_HISTORY_FIRST_PAGE = size
    saver.DELTA_HISTORY_PAGE_SIZE = size
    monkeypatch.setattr(saver, "get_tuple", lambda config: chain.get(_id(config)))
    monkeypatch.setattr(
        saver, "_fetch_chain_page", lambda start, limit: _chain_page(chain, start, limit)
    )
    target = _config(max(chain))
    expected = BaseCheckpointSaver.get_delta_channel_history(
        saver, config=target, channels=CHANNELS
    )
    assert len(expected["tags"]["writes"]) == LENGTH - 1
    assert expected["items"]["seed"] == ["seed"]
    assert saver.get_delta_channel_history(config=target, channels=CHANNELS) == expected


@pytest.mark.parametrize("size", [0, -1])
async def test_async_page_sizes_below_one_still_read_the_whole_chain(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _chain()
    saver = AsyncCockroachDBSaver(AsyncConnectionPool("", open=False))
    saver.DELTA_HISTORY_FIRST_PAGE = size
    saver.DELTA_HISTORY_PAGE_SIZE = size

    async def aget_tuple(config: Any) -> Any:
        return chain.get(_id(config))

    async def fetch_chain_page(start: Any, limit: int) -> list[Any]:
        return _chain_page(chain, start, limit)

    monkeypatch.setattr(saver, "aget_tuple", aget_tuple)
    monkeypatch.setattr(saver, "_fetch_chain_page", fetch_chain_page)
    target = _config(max(chain))
    expected = await BaseCheckpointSaver.aget_delta_channel_history(
        saver, config=target, channels=CHANNELS
    )
    assert len(expected["tags"]["writes"]) == LENGTH - 1
    assert expected["items"]["seed"] == ["seed"]
    assert await saver.aget_delta_channel_history(config=target, channels=CHANNELS) == expected
