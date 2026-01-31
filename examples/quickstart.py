"""Quickstart example for langchain-cockroachdb."""

import asyncio
import os

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

# Replace with your connection string
CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


async def main() -> None:
    """Run quickstart example."""
    print("🪳 LangChain CockroachDB Quickstart\n")

    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)
    embeddings = DeterministicFakeEmbedding(size=768)

    table_name = "quickstart_docs"
    vector_dim = 768

    print(f"1. Initializing table '{table_name}'...")
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_dimension=vector_dim,
        drop_if_exists=True,
    )

    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name=table_name,
    )

    print("\n2. Adding documents...")
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
    ]

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    ids = await vectorstore.aadd_texts(texts, metadatas=metadatas)
    print(f"   Added {len(ids)} documents")

    print("\n3. Similarity search...")
    results = await vectorstore.asimilarity_search("What is CockroachDB?", k=2)

    for i, doc in enumerate(results, 1):
        print(f"   Result {i}: {doc.page_content[:50]}...")
        print(f"   Metadata: {doc.metadata}")

    print("\n4. Search with scores...")
    results_with_scores = await vectorstore.asimilarity_search_with_score("databases", k=3)

    for doc, score in results_with_scores:
        print(f"   Score: {score:.4f} - {doc.page_content[:50]}...")

    print("\n5. Filtered search...")
    filtered_results = await vectorstore.asimilarity_search(
        "technology",
        k=5,
        filter={"category": "database"},
    )

    print(f"   Found {len(filtered_results)} results with category='database'")
    for doc in filtered_results:
        print(f"   - {doc.page_content}")

    print("\n✅ Quickstart complete!")

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
