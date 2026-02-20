# Multi-Tenancy Guide

Isolate documents by namespace within a single table for multi-tenant applications.

## Overview

Multi-tenancy lets multiple tenants share the same vector store table while keeping their data completely isolated. Each tenant's documents are invisible to other tenants during search, retrieval, and deletion.

CockroachDB is uniquely well-suited for this pattern because its C-SPANN vector indexes support **prefix columns** -- the namespace filter is applied at the index level, not as a post-filter.

## Quick Start

### 1. Create a table with a namespace column

```python
from langchain_cockroachdb import CockroachDBEngine

engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
)

await engine.ainit_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
    namespace_column="namespace",  # opt-in: adds the namespace column
)
```

### 2. Create per-tenant vector stores

```python
from langchain_cockroachdb import AsyncCockroachDBVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Tenant A's view
store_a = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    namespace="tenant-a",
)

# Tenant B's view
store_b = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    namespace="tenant-b",
)
```

### 3. Operations are automatically scoped

```python
# Each tenant only sees their own documents
await store_a.aadd_texts(["Tenant A's document"])
await store_b.aadd_texts(["Tenant B's document"])

results_a = await store_a.asimilarity_search("document", k=10)
# Returns only: ["Tenant A's document"]

results_b = await store_b.asimilarity_search("document", k=10)
# Returns only: ["Tenant B's document"]
```

## How It Works

When `namespace` is set on a vector store instance:

- **Insert**: Documents are tagged with the namespace value
- **Search**: A `WHERE namespace = '...'` filter is automatically injected
- **Delete**: Only documents in the namespace can be deleted
- **Get by IDs**: Only returns documents belonging to the namespace

When `namespace` is `None` (the default), no filtering is applied and all documents are visible. This preserves full backward compatibility.

## Scoped Operations

### Search isolation

```python
await store_a.aadd_texts(["AI research paper"], ids=[id_1])
await store_b.aadd_texts(["AI product docs"], ids=[id_2])

# Tenant A only finds their documents
results = await store_a.asimilarity_search("AI")
assert len(results) == 1
assert results[0].page_content == "AI research paper"
```

### Delete isolation

```python
# Deleting in tenant A doesn't affect tenant B
await store_a.adelete([id_1])

results_b = await store_b.asimilarity_search("AI")
assert len(results_b) == 1  # Tenant B's doc is untouched
```

### Get by IDs isolation

```python
# Tenant A can't see tenant B's documents even by ID
docs = await store_a.aget_by_ids([id_2])
assert len(docs) == 0  # Not visible across namespaces
```

## Admin Access (No Namespace)

Create a store without a namespace to see all documents across all tenants:

```python
admin_store = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    # no namespace parameter -- sees everything
)

all_docs = await admin_store.asimilarity_search("AI", k=100)
# Returns documents from all tenants
```

## Performance: C-SPANN Prefix Columns

For best performance with multi-tenancy, create a vector index with the namespace as a prefix column:

```python
from langchain_cockroachdb.indexes import CSPANNIndex

index = CSPANNIndex(
    name="documents_ns_vec_idx",
    prefix_columns=["namespace"],
)
await store_a.aapply_vector_index(index)
```

This creates a `CREATE VECTOR INDEX ... ON documents (namespace, embedding)` which partitions the index by namespace. Searches within a namespace only scan that namespace's partition -- no wasted work scanning other tenants' data.

## Metadata Filtering with Namespaces

Namespace filtering composes with metadata filters:

```python
results = await store_a.asimilarity_search(
    "machine learning",
    k=5,
    filter={"category": "research"},
)
# Filters by namespace="tenant-a" AND metadata.category="research"
```

## When to Use Namespaces vs Separate Tables

| Approach | Use when |
|----------|----------|
| **Namespace column** | Many tenants, shared schema, need cross-tenant admin queries |
| **Separate tables** (`collection_name`) | Few tenants, different schemas, strict physical isolation |

For most multi-tenant SaaS applications, the namespace approach is recommended because it avoids table proliferation and works naturally with CockroachDB's prefix column indexing.

## Next Steps

- [Vector Indexes](vector-indexes.md) - Create prefix-column indexes for multi-tenant performance
- [Vector Store](vector-store.md) - Core vector store operations
