# Async vs Sync Guide

Understanding when to use async and sync APIs.

## Overview

langchain-cockroachdb provides both async and sync interfaces:
- **Async (recommended)**: High-performance, I/O concurrency
- **Sync**: Simple wrapper for scripts and batch jobs

## Quick Comparison

| Feature | Async | Sync |
|---------|-------|------|
| **Throughput** | High (100-1000+ req/s) | Low (10-50 req/s) |
| **I/O Handling** | Non-blocking | Blocking |
| **Concurrency** | Native | Limited |
| **Code Style** | `async`/`await` | Regular Python |
| **Best For** | Web apps, APIs, high load | Scripts, batch jobs, simple tools |

## Async API

### When to Use

✅ **Web Applications**
```python
# FastAPI, Django Async, etc.
from fastapi import FastAPI
from langchain_cockroachdb import AsyncCockroachDBVectorStore

app = FastAPI()

@app.get("/search")
async def search(query: str):
    results = await vectorstore.asimilarity_search(query, k=5)
    return results
```

✅ **High Concurrent Load**
```python
# Handle 100s of requests simultaneously
async def handle_many_requests(queries: list[str]):
    tasks = [
        vectorstore.asimilarity_search(q, k=5) 
        for q in queries
    ]
    results = await asyncio.gather(*tasks)
    return results
```

✅ **Modern LLM APIs**
```python
# OpenAI, Anthropic APIs are async
async def rag_pipeline(query: str):
    # Vector search (async)
    docs = await vectorstore.asimilarity_search(query, k=5)
    
    # LLM call (async)
    response = await llm.ainvoke(docs)
    
    return response
```

✅ **Streaming Applications**
```python
async def stream_results(query: str):
    async for doc in vectorstore.astream_search(query):
        yield doc
```

### Basic Usage

```python
import asyncio
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

async def main():
    # Initialize
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://..."
    )
    
    await engine.ainit_vectorstore_table(
        table_name="docs",
        vector_dimension=1536,
    )
    
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="docs",
    )
    
    # Add documents
    ids = await vectorstore.aadd_texts([
        "Document 1",
        "Document 2",
    ])
    
    # Search
    results = await vectorstore.asimilarity_search(
        "query",
        k=5
    )
    
    # Cleanup
    await engine.aclose()

# Run
asyncio.run(main())
```

### Async Methods

All methods prefixed with `a`:
- `aadd_texts()` - Add documents
- `aadd_documents()` - Add document objects
- `asimilarity_search()` - Search
- `asimilarity_search_with_score()` - Search with scores
- `adelete()` - Delete documents
- `aapply_vector_index()` - Create index
- `adrop_vector_index()` - Drop index

## Sync API

### When to Use

✅ **Simple Scripts**
```python
# One-off data processing
from langchain_cockroachdb import CockroachDBVectorStore

vectorstore = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
)

# No async/await needed
ids = vectorstore.add_texts(["Doc 1", "Doc 2"])
results = vectorstore.similarity_search("query", k=5)
```

✅ **Batch Jobs**
```python
# Sequential processing
for batch in large_dataset:
    ids = vectorstore.add_texts(batch)
    print(f"Processed {len(ids)} documents")
```

✅ **Interactive Development**
```python
# Jupyter notebooks, REPL
vectorstore.add_texts(["Test doc"])
results = vectorstore.similarity_search("test", k=1)
print(results)
```

✅ **Legacy Code**
```python
# Integrating with existing sync codebase
def process_documents(docs: list[str]):
    # No need to refactor to async
    return vectorstore.add_texts(docs)
```

### Basic Usage

```python
from langchain_cockroachdb import CockroachDBVectorStore, CockroachDBEngine

# Initialize (sync wrapper handles async internally)
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://..."
)

# Sync API - no async/await
engine.init_vectorstore_table(
    table_name="docs",
    vector_dimension=1536,
)

vectorstore = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
)

# Add documents (sync)
ids = vectorstore.add_texts([
    "Document 1",
    "Document 2",
])

# Search (sync)
results = vectorstore.similarity_search("query", k=5)

# Cleanup
engine.close()
```

### Sync Methods

Same names without `a` prefix:
- `add_texts()` - Add documents
- `add_documents()` - Add document objects
- `similarity_search()` - Search
- `similarity_search_with_score()` - Search with scores
- `delete()` - Delete documents
- `apply_vector_index()` - Create index
- `drop_vector_index()` - Drop index

## Performance Comparison

### Throughput Test

**Async:**
```python
async def async_benchmark():
    # 100 concurrent searches
    queries = ["query"] * 100
    start = time.time()
    
    tasks = [
        vectorstore.asimilarity_search(q, k=5) 
        for q in queries
    ]
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"Async: {100/elapsed:.0f} queries/sec")

# Result: ~800 queries/sec
```

**Sync:**
```python
def sync_benchmark():
    # 100 sequential searches
    queries = ["query"] * 100
    start = time.time()
    
    for q in queries:
        vectorstore.similarity_search(q, k=5)
    
    elapsed = time.time() - start
    print(f"Sync: {100/elapsed:.0f} queries/sec")

# Result: ~40 queries/sec
```

**Result:** Async is **20x faster** for concurrent operations.

### Why Async is Faster

**Sync (blocking):**
```
Query 1: [DB wait] [process] ────┐
Query 2:                          [DB wait] [process] ────┐
Query 3:                                                  [DB wait] [process]
Total: 300ms
```

**Async (non-blocking):**
```
Query 1: [DB wait] [process]
Query 2: [DB wait] [process]
Query 3: [DB wait] [process]
         ▲ All waiting simultaneously
Total: 100ms
```

## Hybrid Approach

### Async Function with Sync Wrapper

```python
# Define async logic
async def complex_rag_pipeline(query: str) -> str:
    docs = await vectorstore.asimilarity_search(query, k=5)
    response = await llm.ainvoke(docs)
    return response.content

# Provide sync wrapper for simple use
def rag(query: str) -> str:
    """Sync wrapper for RAG pipeline."""
    return asyncio.run(complex_rag_pipeline(query))

# Use sync wrapper in scripts
result = rag("What is CockroachDB?")
print(result)
```

### Run Async from Sync

```python
import asyncio

def sync_function():
    # Run async code from sync context
    result = asyncio.run(vectorstore.asimilarity_search("query", k=5))
    return result
```

### Use Both APIs

```python
# Async for web endpoints
@app.get("/search")
async def search(query: str):
    return await async_vectorstore.asimilarity_search(query, k=5)

# Sync for admin scripts
def backfill_data(documents: list):
    sync_vectorstore.add_documents(documents)
```

## Common Patterns

### Web Application (FastAPI)

```python
from fastapi import FastAPI
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

app = FastAPI()

# Initialize at startup
@app.on_event("startup")
async def startup():
    global vectorstore
    engine = CockroachDBEngine.from_connection_string(
        os.getenv("COCKROACHDB_URL")
    )
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="docs",
    )

# Async endpoints
@app.get("/search")
async def search(query: str, k: int = 5):
    results = await vectorstore.asimilarity_search(query, k=k)
    return {"results": [doc.page_content for doc in results]}

@app.post("/add")
async def add_document(content: str):
    ids = await vectorstore.aadd_texts([content])
    return {"id": ids[0]}
```

### Data Migration Script

```python
from langchain_cockroachdb import CockroachDBVectorStore

def migrate_documents(source_file: str):
    """Simple sync script for one-time migration."""
    vectorstore = CockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="docs",
    )
    
    # Read documents
    with open(source_file) as f:
        docs = f.readlines()
    
    # Process in batches
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        ids = vectorstore.add_texts(batch)
        print(f"Migrated batch {i//batch_size + 1}: {len(ids)} docs")

# Run
migrate_documents("documents.txt")
```

### Jupyter Notebook

```python
# Sync API works well in notebooks
from langchain_cockroachdb import CockroachDBVectorStore

vectorstore = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
)

# Interactive exploration
vectorstore.add_texts(["Test document"])
results = vectorstore.similarity_search("test", k=1)
print(results[0].page_content)
```

## Best Practices

### 1. Choose Based on Context

| Context | Use | Why |
|---------|-----|-----|
| Web API | Async | High concurrency |
| CLI Tool | Sync | Simplicity |
| Data Pipeline | Async | Better throughput |
| Jupyter | Sync | Easier debugging |

### 2. Don't Mix in Same Function

```python
# Bad: Mixing async and sync
async def bad_example():
    await vectorstore.aadd_texts(["doc1"])
    results = vectorstore.similarity_search("query", k=5)  # Wrong!

# Good: Use one or the other
async def good_async_example():
    await vectorstore.aadd_texts(["doc1"])
    results = await vectorstore.asimilarity_search("query", k=5)

def good_sync_example():
    vectorstore.add_texts(["doc1"])
    results = vectorstore.similarity_search("query", k=5)
```

### 3. Async All the Way

```python
# If using async, use it throughout
async def full_async_pipeline(query: str):
    # All async
    docs = await vectorstore.asimilarity_search(query, k=5)
    response = await llm.ainvoke(docs)
    await log_to_db(query, response)
    return response
```

### 4. Close Connections Properly

```python
# Async
try:
    await vectorstore.aadd_texts(texts)
finally:
    await engine.aclose()

# Sync
try:
    vectorstore.add_texts(texts)
finally:
    engine.close()
```

## Troubleshooting

### "RuntimeError: asyncio.run() cannot be called from a running event loop"

You're trying to use `asyncio.run()` inside an async function:

```python
# Wrong
async def my_function():
    result = asyncio.run(vectorstore.asimilarity_search("query"))

# Right
async def my_function():
    result = await vectorstore.asimilarity_search("query")
```

### Slow Performance with Sync API

Consider switching to async if you have:
- Multiple concurrent operations
- I/O-bound workload
- Web application with many users

```python
# Before (slow)
for query in queries:
    vectorstore.similarity_search(query, k=5)

# After (fast)
async def process_queries(queries):
    tasks = [
        vectorstore.asimilarity_search(q, k=5) 
        for q in queries
    ]
    return await asyncio.gather(*tasks)
```

## Next Steps

- [Configuration](../getting-started/configuration.md) - Optimize connection pools
- [Examples](../examples/index.md) - See both APIs in action
- [API Reference](../api/vectorstore.md) - Full method documentation
