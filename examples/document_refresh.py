"""Refresh source chunks with stable IDs; serialize writers for each tenant/source."""

import asyncio
import hashlib
import json
import os
import uuid

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine


async def refresh_source(
    store: AsyncCockroachDBVectorStore, source: str, chunks: list[str]
) -> list[str]:
    """Write the replacement before deleting old chunks, with replayable IDs.

    This is a recipe, not an atomic replacement. Readers may see both versions
    during refresh. Serialize refreshes for the same tenant/source, including
    retries; a stale writer can otherwise remove a newer writer's chunks.
    Include your chunker/model version in the source identity when changing it.
    """
    if not source:
        raise ValueError("source must be non-empty")
    if not chunks:
        await store.adelete_by_filter({"source": source})
        return []
    revision = hashlib.sha256(json.dumps(chunks, ensure_ascii=False).encode()).hexdigest()
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([store.namespace, source, revision, i])))
        for i in range(len(chunks))
    ]
    await store.aadd_texts(
        chunks, ids=ids, metadatas=[{"source": source, "revision": revision} for _ in chunks]
    )
    # Do not remove old chunks if embedding or any insert batch failed.
    await store.adelete_by_filter({"source": source, "revision": {"$ne": revision}})
    return ids


async def main() -> None:
    """Run without an embedding API key and clean up the example's own table."""
    engine = CockroachDBEngine.from_connection_string(
        os.environ.get(
            "COCKROACHDB_URL", "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
        )
    )
    table = "refresh_example_" + uuid.uuid4().hex
    try:
        await engine.ainit_vectorstore_table(table, 8, namespace_column="namespace")
        store = AsyncCockroachDBVectorStore(
            engine, DeterministicFakeEmbedding(size=8), table, namespace="tenant-a"
        )
        old = await refresh_source(store, "manual.pdf", ["Old introduction", "Removed section"])
        new = await refresh_source(store, "manual.pdf", ["Updated introduction"])
        assert await store.aget_by_ids(old) == []
        assert len(await store.aget_by_ids(new)) == 1
        assert await refresh_source(store, "manual.pdf", ["Updated introduction"]) == new
        await refresh_source(store, "manual.pdf", [])
        assert await store.aget_by_ids(new) == []
        print("Updated, replayed, and removed a source without orphaned chunks.")
    finally:
        try:
            await engine.adrop_table(table)
        finally:
            await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
