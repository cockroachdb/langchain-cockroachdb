# Vector Store Guide

Complete guide to using the CockroachDB vector store.

## Overview

The `AsyncCockroachDBVectorStore` provides a high-performance interface for storing and searching vector embeddings in CockroachDB.

## Basic Usage

### Initialization

```python
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine
from langchain_openai import OpenAIEmbeddings

# Create engine
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://user:pass@host:26257/db"
)

# Initialize table
await engine.ainit_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
)

# Create vector store
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=OpenAIEmbeddings(),
    collection_name="documents",
)
```

## Adding Documents

### From Text

```python
# Simple text
texts = [
    "CockroachDB is a distributed database",
    "LangChain simplifies LLM applications",
]
ids = await vectorstore.aadd_texts(texts)
print(f"Added {len(ids)} documents")
```

### With Metadata

```python
texts = ["Doc 1", "Doc 2", "Doc 3"]
metadatas = [
    {"source": "web", "category": "tech", "year": 2024},
    {"source": "pdf", "category": "science", "year": 2023},
    {"source": "api", "category": "tech", "year": 2024},
]

ids = await vectorstore.aadd_texts(texts, metadatas=metadatas)
```

### From Documents

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Content here",
        metadata={"source": "file.txt", "page": 1}
    ),
    Document(
        page_content="More content",
        metadata={"source": "file.txt", "page": 2}
    ),
]

ids = await vectorstore.aadd_documents(documents)
```

### With Custom IDs

```python
import uuid

custom_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
ids = await vectorstore.aadd_texts(texts, ids=custom_ids)
```

### Batch Size Control

```python
# Override default batch size at runtime
ids = await vectorstore.aadd_texts(
    large_text_list,
    batch_size=500  # Larger batches for throughput
)
```

## Searching

### Similarity Search

```python
# Basic search
results = await vectorstore.asimilarity_search(
    "distributed databases",
    k=5
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### With Scores

```python
results = await vectorstore.asimilarity_search_with_score(
    "distributed databases",
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
```

### By Vector

```python
# Search with your own vector
embedding = embeddings.embed_query("my query")
results = await vectorstore.asimilarity_search_by_vector(
    embedding,
    k=5
)
```

### With Relevance Scores

```python
results = await vectorstore.asimilarity_search_with_relevance_scores(
    "query",
    k=5,
    score_threshold=0.7  # Only return scores above threshold
)
```

## Filtering

### Basic Filters

```python
# Equality
results = await vectorstore.asimilarity_search(
    "tech content",
    k=5,
    filter={"category": "tech"}
)

# Multiple conditions (implicit AND)
results = await vectorstore.asimilarity_search(
    "recent tech",
    k=5,
    filter={"category": "tech", "year": 2024}
)
```

### Advanced Filters

#### Comparison Operators

```python
# Greater than
filter={"year": {"$gt": 2020}}

# Greater than or equal
filter={"year": {"$gte": 2020}}

# Less than
filter={"score": {"$lt": 0.9}}

# Less than or equal
filter={"score": {"$lte": 0.9}}

# Not equal
filter={"status": {"$ne": "archived"}}
```

#### IN and NOT IN

```python
# IN operator
filter={"category": {"$in": ["tech", "science", "engineering"]}}

# NOT IN operator
filter={"status": {"$nin": ["draft", "archived"]}}
```

#### Logical Operators

```python
# AND (explicit)
filter={
    "$and": [
        {"category": "tech"},
        {"year": {"$gte": 2023}}
    ]
}

# OR
filter={
    "$or": [
        {"category": "tech"},
        {"category": "science"}
    ]
}

# Complex nested
filter={
    "$and": [
        {"year": {"$gte": 2023}},
        {
            "$or": [
                {"category": "tech"},
                {"category": "science"}
            ]
        }
    ]
}
```

### Filter Examples

```python
# Recent tech or science documents
results = await vectorstore.asimilarity_search(
    "innovation",
    k=10,
    filter={
        "$and": [
            {"year": {"$gte": 2023}},
            {"category": {"$in": ["tech", "science"]}}
        ]
    }
)

# High-quality documents from specific sources
results = await vectorstore.asimilarity_search(
    "research",
    k=10,
    filter={
        "$and": [
            {"source": {"$in": ["arxiv", "nature", "ieee"]}},
            {"quality_score": {"$gte": 0.8}}
        ]
    }
)
```

## Updating and Deleting

### Update Documents

```python
# Re-add with same ID to update
await vectorstore.aadd_texts(
    ["Updated content"],
    ids=["existing-id"]
)
```

### Delete by IDs

```python
ids_to_delete = ["id1", "id2", "id3"]
await vectorstore.adelete(ids=ids_to_delete)
```

### Delete All

```python
# Delete all documents in collection
await vectorstore.adelete()
```

## Retriever Interface

### As Retriever

```python
# Convert to LangChain retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Use in chains
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=retriever,
)

answer = await qa.ainvoke("What is CockroachDB?")
```

### MMR (Maximal Marginal Relevance)

```python
# Diverse results (less similar to each other)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 50,  # Fetch more candidates
        "lambda_mult": 0.5  # Balance relevance vs diversity
    }
)

# Or directly
results = await vectorstore.amax_marginal_relevance_search(
    "query",
    k=10,
    fetch_k=50,
    lambda_mult=0.5
)
```

### Similarity Score Threshold

```python
# Only return highly relevant results
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.8
    }
)
```

## Distance Strategies

Choose distance metric based on your embeddings:

```python
from langchain_cockroachdb import DistanceStrategy

# Cosine similarity (default, normalized vectors)
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    distance_strategy=DistanceStrategy.COSINE
)

# Euclidean distance (L2)
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    distance_strategy=DistanceStrategy.EUCLIDEAN
)

# Inner product
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    distance_strategy=DistanceStrategy.INNER_PRODUCT
)
```

### Which to Use?

| Distance Strategy | Best For | Normalized? |
|-------------------|----------|-------------|
| **Cosine** | Most embeddings (OpenAI, Anthropic) | Yes |
| **Euclidean (L2)** | Spatial data, distances matter | No |
| **Inner Product** | Pre-normalized vectors, speed priority | Yes |

## Table Management

### Drop Table

```python
await vectorstore.adrop_table()
```

### Check if Table Exists

```python
async with engine.engine.connect() as conn:
    from sqlalchemy import text
    result = await conn.execute(
        text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'documents'
            )
        """)
    )
    exists = result.scalar()
```

## Performance Tips

### 1. Use Appropriate Batch Sizes

```python
# Smaller embeddings
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=500  # < 512 dims
)

# Larger embeddings
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=100  # > 1024 dims
)
```

### 2. Create Indexes

```python
from langchain_cockroachdb import CSPANNIndex

# Create vector index for faster queries
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)
```

See [Vector Indexes Guide](vector-indexes.md) for details.

### 3. Use Connection Pooling

```python
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=20,      # More connections for concurrency
    max_overflow=40,
)
```

### 4. Filter Early

```python
# Good: Filter reduces candidates before vector search
results = await vectorstore.asimilarity_search(
    "query",
    k=10,
    filter={"category": "tech"}  # Narrows search space
)

# Less efficient: No filter means searching all vectors
results = await vectorstore.asimilarity_search(
    "query",
    k=10
)
# Then filter results in Python
```

## Common Patterns

### Multi-Tenant Isolation

```python
# Store tenant_id in metadata
await vectorstore.aadd_texts(
    texts,
    metadatas=[{"tenant_id": "tenant-123"}] * len(texts)
)

# Query with tenant filter
results = await vectorstore.asimilarity_search(
    "query",
    k=5,
    filter={"tenant_id": "tenant-123"}
)
```

### Versioning Documents

```python
# Include version in metadata
await vectorstore.aadd_texts(
    ["Document content"],
    metadatas=[{"doc_id": "abc123", "version": 2}]
)

# Query latest version
results = await vectorstore.asimilarity_search(
    "query",
    k=5,
    filter={"version": {"$gte": 2}}
)
```

### Source Attribution

```python
# Track sources
metadatas = [
    {
        "source": "https://example.com/page1",
        "title": "Page Title",
        "author": "John Doe",
        "date": "2024-01-15"
    }
]

await vectorstore.aadd_texts(texts, metadatas=metadatas)

# Search and cite sources
results = await vectorstore.asimilarity_search_with_score(
    "query", k=3
)

for doc, score in results:
    print(f"Source: {doc.metadata['source']}")
    print(f"Relevance: {score:.2f}")
```

## Error Handling

```python
import asyncio

try:
    results = await vectorstore.asimilarity_search("query", k=5)
except Exception as e:
    print(f"Search failed: {e}")
    # Retry logic is automatic, but you can catch final failure
```

## Next Steps

- [Vector Indexes](vector-indexes.md) - Optimize query performance
- [Hybrid Search](hybrid-search.md) - Combine FTS with vectors
- [Configuration](../getting-started/configuration.md) - Tune for your workload
