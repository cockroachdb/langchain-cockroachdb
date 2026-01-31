"""Example demonstrating advanced metadata filtering."""

import asyncio
import os

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


async def main() -> None:
    """Demonstrate metadata filtering capabilities."""
    print("🪳 Advanced Metadata Filtering\n")

    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)
    embeddings = DeterministicFakeEmbedding(size=768)

    table_name = "filtered_docs"

    print("1. Setting up vector store...")
    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_dimension=768,
        drop_if_exists=True,
    )

    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name=table_name,
    )

    print("\n2. Adding documents with rich metadata...")

    documents = [
        ("Python tutorial for beginners", {"lang": "python", "level": "beginner", "year": 2024}),
        ("Advanced Python patterns", {"lang": "python", "level": "advanced", "year": 2024}),
        ("JavaScript basics", {"lang": "javascript", "level": "beginner", "year": 2023}),
        ("Go programming guide", {"lang": "go", "level": "intermediate", "year": 2024}),
        ("Rust advanced techniques", {"lang": "rust", "level": "advanced", "year": 2023}),
        ("Python data science", {"lang": "python", "level": "intermediate", "year": 2024}),
    ]

    texts = [doc[0] for doc in documents]
    metadatas = [doc[1] for doc in documents]

    await vectorstore.aadd_texts(texts, metadatas=metadatas)
    print(f"   Added {len(documents)} documents")

    print("\n3. Simple equality filter...")
    results = await vectorstore.asimilarity_search(
        "programming",
        k=10,
        filter={"lang": "python"},
    )
    print(f"   Python docs: {len(results)} found")
    for doc in results:
        print(f"   - {doc.page_content} | {doc.metadata}")

    print("\n4. Comparison operators...")
    results = await vectorstore.asimilarity_search(
        "programming",
        k=10,
        filter={"year": {"$gte": 2024}},
    )
    print(f"   Docs from 2024+: {len(results)} found")

    print("\n5. IN operator...")
    results = await vectorstore.asimilarity_search(
        "programming",
        k=10,
        filter={"lang": {"$in": ["python", "go"]}},
    )
    print(f"   Python or Go docs: {len(results)} found")

    print("\n6. Complex AND filter...")
    results = await vectorstore.asimilarity_search(
        "programming",
        k=10,
        filter={
            "$and": [
                {"lang": "python"},
                {"year": {"$gte": 2024}},
                {"level": {"$in": ["beginner", "intermediate"]}},
            ]
        },
    )
    print(f"   Python 2024+ beginner/intermediate: {len(results)} found")
    for doc in results:
        print(f"   - {doc.page_content} | {doc.metadata}")

    print("\n7. OR filter...")
    results = await vectorstore.asimilarity_search(
        "programming",
        k=10,
        filter={
            "$or": [
                {"level": "advanced"},
                {"year": 2023},
            ]
        },
    )
    print(f"   Advanced OR 2023 docs: {len(results)} found")

    print("\n8. Nested logical operators...")
    results = await vectorstore.asimilarity_search(
        "programming",
        k=10,
        filter={
            "$and": [
                {"year": 2024},
                {
                    "$or": [
                        {"lang": "python"},
                        {"lang": "go"},
                    ]
                },
            ]
        },
    )
    print(f"   2024 AND (Python OR Go): {len(results)} found")

    print("\n9. Supported filter operators:")
    operators = [
        ("$eq", "Equal to"),
        ("$ne", "Not equal to"),
        ("$gt", "Greater than"),
        ("$gte", "Greater than or equal"),
        ("$lt", "Less than"),
        ("$lte", "Less than or equal"),
        ("$in", "In list"),
        ("$nin", "Not in list"),
        ("$and", "Logical AND"),
        ("$or", "Logical OR"),
    ]

    for op, desc in operators:
        print(f"   {op:8s} - {desc}")

    print("\n✅ Metadata filtering demo complete!")

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
