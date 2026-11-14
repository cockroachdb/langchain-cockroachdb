"""Integration: delete_by_filter removes matching rows and respects namespaces."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CockroachDBVectorStore,
)


@pytest.mark.parametrize("sync", [False, True])
async def test_adelete_by_filter_removes_only_matching_rows(
    cockroachdb_engine: CockroachDBEngine,
    sync: bool,
) -> None:
    embeddings = DeterministicFakeEmbedding(size=3)
    table = "test_delete_by_filter"
    await cockroachdb_engine.ainit_vectorstore_table(
        table_name=table, vector_dimension=3, drop_if_exists=True
    )
    store = CockroachDBVectorStore(
        engine=cockroachdb_engine, embeddings=embeddings, collection_name=table
    )
    keep = str(uuid.uuid4())
    drop_one = str(uuid.uuid4())
    drop_two = str(uuid.uuid4())
    await store.aadd_texts(
        ["keep", "old one", "old two"],
        metadatas=[{"source": "keep.pdf"}, {"source": "old.pdf"}, {"source": "old.pdf"}],
        ids=[keep, drop_one, drop_two],
    )

    for _ in range(2):
        if sync:
            assert store.delete_by_filter({"source": "old.pdf"}) is True
        else:
            assert await store.adelete_by_filter({"source": "old.pdf"}) is True

    remaining = {doc.id for doc in await store.aget_by_ids([keep, drop_one, drop_two])}
    assert remaining == {keep}


@pytest.mark.parametrize("sync", [False, True])
async def test_adelete_by_filter_is_scoped_to_namespace(
    cockroachdb_engine: CockroachDBEngine,
    sync: bool,
) -> None:
    embeddings = DeterministicFakeEmbedding(size=3)
    table = "test_delete_by_filter_ns"
    await cockroachdb_engine.ainit_vectorstore_table(
        table_name=table,
        vector_dimension=3,
        namespace_column="namespace",
        drop_if_exists=True,
    )
    store_a = CockroachDBVectorStore(
        engine=cockroachdb_engine,
        embeddings=embeddings,
        collection_name=table,
        namespace="tenant-a",
    )
    store_b = AsyncCockroachDBVectorStore(
        engine=cockroachdb_engine,
        embeddings=embeddings,
        collection_name=table,
        namespace="tenant-b",
    )
    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())
    await store_a.aadd_texts(["a"], metadatas=[{"source": "shared.pdf"}], ids=[id_a])
    await store_b.aadd_texts(["b"], metadatas=[{"source": "shared.pdf"}], ids=[id_b])

    # Deleting in tenant-a with a filter that also matches tenant-b's row
    # must leave tenant-b untouched.
    if sync:
        assert store_a.delete_by_filter({"source": "shared.pdf"}) is True
    else:
        assert await store_a.adelete_by_filter({"source": "shared.pdf"}) is True

    assert await store_a.aget_by_ids([id_a]) == []
    assert len(await store_b.aget_by_ids([id_b])) == 1
