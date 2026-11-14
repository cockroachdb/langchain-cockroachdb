"""The refresh recipe preserves old chunks on failure and reuses IDs on replay."""

from unittest.mock import AsyncMock, Mock

import pytest

from examples.document_refresh import refresh_source


async def test_failed_ingestion_does_not_delete_old_chunks():
    store = Mock(namespace="tenant", aadd_texts=AsyncMock(side_effect=RuntimeError("failed")))
    store.adelete_by_filter = AsyncMock()
    with pytest.raises(RuntimeError, match="failed"):
        await refresh_source(store, "manual.pdf", ["new"])
    store.adelete_by_filter.assert_not_called()


async def test_replay_uses_stable_ids_and_tenants_have_distinct_ids():
    store = Mock(namespace="a", aadd_texts=AsyncMock(), adelete_by_filter=AsyncMock())
    first = await refresh_source(store, "manual.pdf", ["one", "two"])
    assert await refresh_source(store, "manual.pdf", ["one", "two"]) == first
    metadata = store.aadd_texts.call_args.kwargs["metadatas"]
    store.adelete_by_filter.assert_awaited_with(
        {"source": "manual.pdf", "revision": {"$ne": metadata[0]["revision"]}}
    )
    assert await refresh_source(store, "manual.pdf", ["new"]) != first
    store.namespace = "b"
    assert set(await refresh_source(store, "manual.pdf", ["one", "two"])).isdisjoint(first)


async def test_removed_source_does_not_call_embeddings():
    store = Mock(namespace="a", aadd_texts=AsyncMock(), adelete_by_filter=AsyncMock())
    assert await refresh_source(store, "manual.pdf", []) == []
    store.aadd_texts.assert_not_called()
    store.adelete_by_filter.assert_awaited_once_with({"source": "manual.pdf"})
