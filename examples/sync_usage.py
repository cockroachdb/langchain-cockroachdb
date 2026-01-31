# ruff: noqa: PLR0915
"""Synchronous usage example for langchain-cockroachdb.

This example demonstrates using the sync wrapper (CockroachDBVectorStore)
for simple scripts, batch jobs, or legacy code that doesn't use async/await.

For high-performance applications with concurrent operations, use the
async version (AsyncCockroachDBVectorStore) instead.
"""

import os

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import CockroachDBEngine, CockroachDBVectorStore
from langchain_cockroachdb.indexes import CSPANNIndex, DistanceStrategy

# Replace with your connection string
CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


def main() -> None:
    """Run sync usage example."""
    print("🪳 LangChain CockroachDB - Synchronous Usage\n")

    # 1. Create engine (sync wrapper)
    print("1. Creating engine...")
    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)

    embeddings = DeterministicFakeEmbedding(size=768)
    table_name = "sync_example_docs"
    vector_dim = 768

    # 2. Initialize table (sync)
    print(f"2. Initializing table '{table_name}'...")
    engine.init_vectorstore_table(
        table_name=table_name,
        vector_dimension=vector_dim,
        drop_if_exists=True,
    )

    # 3. Create sync vectorstore
    print("3. Creating vectorstore...")
    vectorstore = CockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name=table_name,
    )

    # 4. Add documents (sync)
    print("\n4. Adding documents...")
    documents = [
        Document(
            page_content="CockroachDB is a distributed SQL database",
            metadata={"source": "docs", "category": "database"},
        ),
        Document(
            page_content="LangChain helps build LLM applications",
            metadata={"source": "docs", "category": "framework"},
        ),
        Document(
            page_content="Vector search enables semantic similarity",
            metadata={"source": "blog", "category": "search"},
        ),
        Document(
            page_content="Python is great for data science",
            metadata={"source": "blog", "category": "programming"},
        ),
    ]

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    ids = vectorstore.add_texts(texts, metadatas=metadatas)
    print(f"   Added {len(ids)} documents")

    # 5. Similarity search (sync)
    print("\n5. Similarity search for 'database':")
    results = vectorstore.similarity_search("database", k=2)
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      Metadata: {doc.metadata}")

    # 6. Search with scores (sync)
    print("\n6. Similarity search with scores:")
    results_with_scores = vectorstore.similarity_search_with_score("programming", k=2)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      Score: {score:.4f}")
        print(f"      Metadata: {doc.metadata}")

    # 7. Search with metadata filter (sync)
    print("\n7. Search with metadata filter (category='blog'):")
    results = vectorstore.similarity_search(
        "technology",
        k=5,
        filter={"category": {"$eq": "blog"}},
    )
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      Category: {doc.metadata['category']}")

    # 8. Apply vector index (sync)
    print("\n8. Creating vector index...")
    index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
    vectorstore.apply_vector_index(index)
    print("   Index created successfully!")

    # 9. Search still works with index
    print("\n9. Search with index:")
    results = vectorstore.similarity_search("LangChain", k=2)
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")

    # 10. MMR search (sync)
    print("\n10. Max Marginal Relevance search:")
    mmr_results = vectorstore.max_marginal_relevance_search(
        "database technology",
        k=2,
        fetch_k=4,
    )
    for i, doc in enumerate(mmr_results, 1):
        print(f"   {i}. {doc.page_content}")

    # 11. Delete documents (sync)
    print("\n11. Deleting one document...")
    deleted = vectorstore.delete([ids[0]])
    print(f"   Deleted: {deleted}")

    remaining = vectorstore.similarity_search("", k=10)
    print(f"   Remaining documents: {len(remaining)}")

    # 12. Cleanup
    print("\n12. Cleaning up...")
    engine.close()
    print("   Done!")

    print("\n" + "=" * 60)
    print("SYNC vs ASYNC:")
    print("=" * 60)
    print("✅ Use SYNC when:")
    print("   - Simple scripts or batch jobs")
    print("   - Sequential processing (one doc at a time)")
    print("   - Legacy code without async/await")
    print("   - Simpler mental model")
    print()
    print("✅ Use ASYNC when:")
    print("   - High-performance web applications")
    print("   - Concurrent operations (many searches/inserts)")
    print("   - Modern LLM apps (OpenAI/Anthropic APIs are async)")
    print("   - Distributed CockroachDB (10-100x throughput gain)")
    print("=" * 60)


if __name__ == "__main__":
    main()
