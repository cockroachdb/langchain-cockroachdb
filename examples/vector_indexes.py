"""Example demonstrating vector indexes with C-SPANN."""

import asyncio
import os

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CSPANNIndex,
    CSPANNQueryOptions,
    DistanceStrategy,
)

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


async def main() -> None:
    """Demonstrate vector index creation and usage."""
    print("🪳 CockroachDB C-SPANN Vector Indexes\n")

    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)
    embeddings = DeterministicFakeEmbedding(size=768)

    table_name = "indexed_vectors"

    print("1. Creating vector store...")
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_dimension=768,
        drop_if_exists=True,
    )

    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name=table_name,
        distance_strategy=DistanceStrategy.COSINE,
    )

    print("\n2. Inserting sample documents...")
    texts = [f"Document about topic {i}" for i in range(100)]
    await vectorstore.aadd_texts(texts)
    print(f"   Inserted {len(texts)} documents")

    print("\n3. Creating C-SPANN vector index...")
    index = CSPANNIndex(
        distance_strategy=DistanceStrategy.COSINE,
        min_partition_size=10,
        max_partition_size=50,
    )

    await vectorstore.aapply_vector_index(index)
    print("   ✓ Index created")

    print("\n4. Searching without beam size tuning (default)...")
    results = await vectorstore.asimilarity_search("topic 50", k=5)
    print(f"   Found {len(results)} results")

    print("\n5. Searching with higher beam size (more accurate)...")
    query_options = CSPANNQueryOptions(beam_size=200)
    results = await vectorstore.asimilarity_search_with_score(
        "topic 50",
        k=5,
        query_options=query_options,
    )

    print(f"   Found {len(results)} results with scores:")
    for doc, score in results[:3]:
        print(f"   - Score: {score:.4f}, Content: {doc.page_content}")

    print("\n6. Testing different distance strategies...")

    for strategy in [DistanceStrategy.EUCLIDEAN, DistanceStrategy.INNER_PRODUCT]:
        strategy_table = f"vectors_{strategy.value}"

        await engine.ainit_vectorstore_table(
            table_name=strategy_table,
            vector_dimension=768,
            drop_if_exists=True,
        )

        vs = AsyncCockroachDBVectorStore(
            engine=engine,
            embeddings=embeddings,
            collection_name=strategy_table,
            distance_strategy=strategy,
        )

        await vs.aadd_texts(texts[:20])

        index = CSPANNIndex(distance_strategy=strategy)
        await vs.aapply_vector_index(index)

        print(f"   ✓ Created {strategy.value} index")

    print("\n7. Multi-tenant index with prefix columns...")
    tenant_table = "tenant_vectors"

    await engine.ainit_vectorstore_table(
        table_name=tenant_table,
        vector_dimension=768,
        drop_if_exists=True,
    )

    print("   (Prefix column support - schema would need tenant_id column)")
    print("   CREATE VECTOR INDEX ON tenant_vectors (tenant_id, embedding)")

    print("\n✅ Vector index demo complete!")

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
