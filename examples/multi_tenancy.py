"""Example demonstrating multi-tenancy with namespace-based isolation.

Shows how to use the namespace parameter to isolate documents by tenant
within a single CockroachDB table.
"""

import asyncio
import os
import uuid

from langchain_core.embeddings import Embeddings

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


class DemoEmbeddings(Embeddings):
    """Simple embeddings for demonstration (replace with real model)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), float(i + 1), float(i + 2)] for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


async def main() -> None:
    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)
    embeddings = DemoEmbeddings()

    # 1. Create table with namespace column (opt-in)
    print("1. Creating table with namespace column\n")
    await engine.ainit_vectorstore_table(
        table_name="multi_tenant_docs",
        vector_dimension=3,
        namespace_column="namespace",
        drop_if_exists=True,
    )
    print("   Table created with namespace column\n")

    # 2. Create per-tenant stores
    print("2. Creating per-tenant vector stores\n")
    store_acme = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="multi_tenant_docs",
        namespace="acme-corp",
    )
    store_globex = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="multi_tenant_docs",
        namespace="globex-inc",
    )
    print("   Created stores for acme-corp and globex-inc\n")

    # 3. Add documents per tenant
    print("3. Adding documents per tenant\n")
    acme_ids = [str(uuid.uuid4()) for _ in range(2)]
    await store_acme.aadd_texts(
        ["Acme quarterly report", "Acme product roadmap"],
        ids=acme_ids,
    )
    globex_ids = [str(uuid.uuid4()) for _ in range(2)]
    await store_globex.aadd_texts(
        ["Globex financial summary", "Globex hiring plan"],
        ids=globex_ids,
    )
    print("   Added 2 docs each for acme-corp and globex-inc\n")

    # 4. Search is scoped to each tenant
    print("4. Search isolation\n")
    acme_results = await store_acme.asimilarity_search("report", k=10)
    print(f"   Acme search results ({len(acme_results)} docs):")
    for doc in acme_results:
        print(f"     - {doc.page_content}")

    globex_results = await store_globex.asimilarity_search("report", k=10)
    print(f"   Globex search results ({len(globex_results)} docs):")
    for doc in globex_results:
        print(f"     - {doc.page_content}")

    # 5. Admin view (no namespace) sees all
    print("\n5. Admin view (no namespace)\n")
    admin_store = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="multi_tenant_docs",
    )
    all_results = await admin_store.asimilarity_search("report", k=10)
    print(f"   Admin sees all {len(all_results)} docs:")
    for doc in all_results:
        print(f"     - {doc.page_content}")

    # 6. Delete isolation
    print("\n6. Delete isolation\n")
    await store_acme.adelete(acme_ids)
    print("   Deleted all Acme docs")

    globex_after = await store_globex.asimilarity_search("plan", k=10)
    print(f"   Globex still has {len(globex_after)} docs (unaffected)")

    # Cleanup
    await engine.adrop_table("multi_tenant_docs")
    await engine.aclose()
    print("\n   Done!")


if __name__ == "__main__":
    asyncio.run(main())
