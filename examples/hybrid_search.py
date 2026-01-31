"""Example demonstrating hybrid search (FTS + vector similarity)."""

import asyncio
import os

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    HybridSearchConfig,
)

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


async def main() -> None:
    """Demonstrate hybrid search combining FTS and vector similarity."""
    print("🪳 Hybrid Search (FTS + Vector)\n")

    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)
    embeddings = DeterministicFakeEmbedding(size=768)

    table_name = "hybrid_docs"

    print("1. Creating table with TSVECTOR support...")
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_dimension=768,
        create_tsvector=True,
        drop_if_exists=True,
    )

    print("\n2. Creating vector store with hybrid search config...")

    hybrid_config = HybridSearchConfig(
        fts_weight=0.3,
        vector_weight=0.7,
        fusion_type="weighted_sum",
    )

    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name=table_name,
        hybrid_search_config=hybrid_config,
    )

    print("\n3. Adding sample documents...")
    documents = [
        "CockroachDB is a distributed SQL database built for cloud-native applications",
        "PostgreSQL is a powerful open-source relational database",
        "Vector databases specialize in storing and querying high-dimensional embeddings",
        "Full-text search enables finding documents based on keyword matching",
        "Hybrid search combines the best of keyword search and semantic similarity",
        "LangChain framework helps developers build LLM-powered applications",
        "Semantic search uses vector embeddings to find conceptually similar content",
    ]

    await vectorstore.aadd_texts(documents)
    print(f"   Added {len(documents)} documents")

    print("\n4. Pure vector search...")
    vector_results = await vectorstore.asimilarity_search("databases for distributed systems", k=3)

    print("   Vector search results:")
    for i, doc in enumerate(vector_results, 1):
        print(f"   {i}. {doc.page_content[:60]}...")

    print("\n5. Hybrid search (if fully implemented)...")
    print("   Note: Full hybrid search requires additional implementation")
    print("   - FTS query generation")
    print("   - Score normalization")
    print("   - Fusion algorithm")

    print("\n6. Different fusion strategies...")

    configs = [
        (
            "Weighted Sum (70% vector, 30% FTS)",
            HybridSearchConfig(
                vector_weight=0.7,
                fts_weight=0.3,
                fusion_type="weighted_sum",
            ),
        ),
        (
            "Reciprocal Rank Fusion",
            HybridSearchConfig(
                vector_weight=0.5,
                fts_weight=0.5,
                fusion_type="reciprocal_rank_fusion",
            ),
        ),
    ]

    for name, config in configs:
        print(f"\n   {name}:")
        print(f"   - Vector weight: {config.vector_weight}")
        print(f"   - FTS weight: {config.fts_weight}")
        print(f"   - Fusion: {config.fusion_type}")

    print("\n✅ Hybrid search demo complete!")

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
