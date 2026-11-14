# Vector Indexes Guide

Optimize query performance with CockroachDB's C-SPANN vector indexes.

## Overview

CockroachDB uses **C-SPANN** (CockroachDB SPANN - Scalable Partitioned Approximate Nearest Neighbors) for vector indexing. This algorithm is optimized for distributed databases and provides:

- Real-time indexing (no rebuild needed)
- Automatic sharding and rebalancing
- Fresh results (queries see latest data)
- Distributed query execution

## When to Use Indexes

**Use indexes when:**
- ✅ Dataset > 10,000 vectors
- ✅ Query latency is critical
- ✅ Frequent searches on same dataset
- ✅ Multi-tenant with prefix columns

**Skip indexes when:**
- ❌ Dataset < 1,000 vectors
- ❌ Mostly writes, few reads
- ❌ Data changes constantly
- ❌ Acceptable to scan all vectors

## Basic Index Creation

### Create Index

```python
from langchain_cockroachdb import CSPANNIndex

# Create with defaults
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)
```

This creates:
```sql
CREATE VECTOR INDEX IF NOT EXISTS documents_embedding_vector_idx
ON public.documents (embedding vector_cosine_ops);
```

### Specify Distance Strategy

```python
from langchain_cockroachdb import CSPANNIndex, DistanceStrategy

# Cosine similarity (default)
index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
await vectorstore.aapply_vector_index(index)

# Euclidean distance (L2)
index = CSPANNIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
await vectorstore.aapply_vector_index(index)

# Inner product
index = CSPANNIndex(distance_strategy=DistanceStrategy.INNER_PRODUCT)
await vectorstore.aapply_vector_index(index)
```

## Advanced Index Configuration

### Partition Sizes

Control how vectors are partitioned:

```python
index = CSPANNIndex(
    distance_strategy=DistanceStrategy.COSINE,
    min_partition_size=10,    # Minimum vectors per partition
    max_partition_size=100,   # Maximum vectors per partition
)
await vectorstore.aapply_vector_index(index)
```

**Guidelines:**
- Smaller partitions = higher accuracy, slower queries
- Larger partitions = faster queries, lower accuracy
- Default (None) = CockroachDB chooses optimal size

### Custom Index Name

```python
index = CSPANNIndex(name="my_custom_index_name")
await vectorstore.aapply_vector_index(index)
```

## Multi-Tenant Indexes

### Prefix Columns

An index can have prefix columns before its vector column. They come only from `prefix_columns`, never from the order of the table's columns:

```python
# The table was created with namespace_column="namespace"
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index, prefix_columns=["namespace"])
```

This creates:
```sql
CREATE VECTOR INDEX IF NOT EXISTS documents_embedding_vector_idx
ON public.documents (namespace, embedding vector_cosine_ops);
```

CockroachDB uses such an index only for a search that constrains every prefix column to a value, with `=` or `IN`. A store created with `namespace="tenant-a"` adds `namespace = 'tenant-a'` to each of its searches, so an index whose only prefix column is the namespace serves them, with or without a metadata filter. A store without a namespace constrains no column, so it cannot use an index with prefix columns. The store never constrains a column other than the namespace.

A key in `filter` is a filter on the JSON metadata column, not a prefix column, even when it has the name of a column.

**Benefits:**
- Each tenant's vectors are stored together in the index
- A search reads only its own tenant's part of the index

## Filtered Search

CockroachDB uses a vector index only when every filter of the search matches the index's prefix columns. A metadata filter never does. So when the filter sits next to `ORDER BY distance LIMIT k`, CockroachDB scans the table and ranks every row that passes.

When the table has a usable index (see below), a search with a `filter` works in steps instead:

1. It takes the nearest rows from the index, with no filter: twice `k` plus one, and at least 41, which costs the index about the same as fewer (on a 20,000-row table, a window of 41 read about 10% more rows than a window of 9, in the same time). For a namespaced store, the namespace is the index's prefix column, so the index reads only that namespace.
2. It applies the filter to those rows and keeps the `k` nearest that pass. Steps 1 and 2 are one statement.
3. When fewer than `k` rows pass, matches may lie beyond the window, so the store runs the exact search: the statement every filtered search ran before 0.6.0, which filters before it ranks. When no row passes, it always does. When some do, it skips the exact search only if the window held every row of the table or namespace, since the short result is then the whole answer.

A filter that most rows near the query pass is served by the index alone, in one statement. A selective filter often needs the exact search as well, which costs the scan it always did, plus the short indexed read before it. The exact search returns only the store's namespace, but its plan can scan the whole table to find those rows.

A result that comes from the window is approximate, as every index search is: the index can miss a near neighbor, and the beam size sets how often (see [Beam Size](#beam-size)). Before 0.6.0, filtered searches scanned and were always exact.

A usable index:

- is on the store's embedding column;
- has the operator class of the store's distance strategy (see [Index Opclasses](#index-opclasses));
- has no prefix column for a store without a namespace, and the namespace column as its only prefix column for a namespaced store.

A namespaced store also needs the primary key or another index to lead with the namespace column, so that counting the namespace's rows does not scan the table. `ainit_vectorstore_table` creates that index with the namespace column. A table without one keeps the exact search.

The store reads the table's indexes with `SHOW CREATE TABLE` on its first filtered search. The engine keeps the answer for every store with the same table, embedding column, distance strategy and namespace column, so an application that builds a store per request, one per tenant for example, reads it once per table, not once per request. It is read again once `VECTOR_INDEX_REFRESH_SECONDS` (60) have passed, so an index that another process creates is used within a minute, and right after an index is applied or dropped through any store on the engine. Without a filter, or without a usable index, a search runs the same single statement as before.

This covers `similarity_search` and its variants, max marginal relevance search and the vector leg of hybrid search, in the sync and async stores. In MMR search the candidate pool, `fetch_k`, takes the place of `k`. In hybrid search, so does its pool: `fetch_k`, or `max(4 * k, 20)` without it.

## Query-Time Tuning

### Beam Size

The beam size sets how much of the index a search explores. CockroachDB's default, the `vector_search_beam_size` session setting, is 32. Pass `query_options` to change it for one search:

```python
from langchain_cockroachdb import CSPANNQueryOptions

# A larger beam: more accurate, slower
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=64)
)

# A smaller beam: faster, less accurate
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=16)
)
```

The store sets it with `SET LOCAL` in the search's own transaction, so it never stays on a pooled connection. The MMR methods take `query_options=` too.

**Guidelines:**
- Start with the default, 32
- If results miss near neighbors, try 64, then 128
- If latency matters more than recall, try 16
- Check recall against an exact search on a sample of real queries

## Index Management

### Drop Index

```python
await vectorstore.adrop_vector_index()
```

Or by name:
```sql
DROP INDEX index_name;
```

### Check Index Status

```python
async with engine.engine.connect() as conn:
    from sqlalchemy import text
    result = await conn.execute(
        text("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'documents'
        """)
    )
    for row in result:
        print(f"Index: {row[0]}")
        print(f"Definition: {row[1]}")
```

### Rebuild Index

Indexes are maintained automatically, but if needed:

```python
# Drop and recreate
await vectorstore.adrop_vector_index()
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)
```

## Performance Comparison

### Small Dataset (1,000 vectors)

| Configuration | Query Latency | Notes |
|---------------|---------------|-------|
| No index | 4-5ms | Fast enough, index not needed |
| With index | 5-10ms | Index overhead > benefit |

**Recommendation:** Skip index for < 10K vectors

### Medium Dataset (50,000 vectors)

| Configuration | Query Latency | Notes |
|---------------|---------------|-------|
| No index | 50-200ms | Sequential scan slow |
| With index | 10-30ms | 5-10x faster |
| Index, beam below the default 32 | 5-15ms | Fast but less accurate |
| Index, beam above the default 32 | 15-40ms | More accurate |

**Recommendation:** Create index, tune beam size for workload

### Large Dataset (1M+ vectors)

| Configuration | Query Latency | Notes |
|---------------|---------------|-------|
| No index | 5-30s | Unusable |
| With index | 20-100ms | Essential |
| Index + partitions | 10-50ms | Tuned partitions help |

**Recommendation:** Always use index, tune partitions and beam

## Index Opclasses

The index uses the operator class of its distance strategy. The vector store sets this automatically when you call `aapply_vector_index`:

| Distance Strategy | Opclass | Operator |
|-------------------|---------|----------|
| COSINE | `vector_cosine_ops` | `<=>` |
| EUCLIDEAN (L2) | `vector_l2_ops` | `<->` |
| INNER_PRODUCT | `vector_ip_ops` | `<#>` |

!!! warning "Upgrading from 0.3.x"
    Indexes created by earlier versions for COSINE or INNER_PRODUCT are `vector_l2_ops` and are not used by those searches. Drop them with `await vectorstore.adrop_vector_index()` and apply again. `aapply_vector_index` logs a warning when it finds a mismatched opclass, different prefix columns, or a partition size other than one you set. Applying the same index again logs nothing.

## Best Practices

### 1. Create the Index Before Writes Start

```python
# Create the table, then the index while the table is empty
await engine.ainit_vectorstore_table("documents", vector_dimension=1536)
await vectorstore.aapply_vector_index(CSPANNIndex())

# Then write
await vectorstore.aadd_texts(texts)
```

Building an index on a table that already holds data starts a backfill, and CockroachDB blocks writes to the table until the backfill ends. This is a known limitation of vector indexes. On a table that takes writes all the time, create the index before the writes start. To index a table that already holds data, choose a time when its writes can pause.

### 2. Use the Namespace as the Prefix Column for Multi-Tenancy

```python
# One store per tenant, on a table created with namespace_column="namespace"
store = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    namespace="tenant-123",
)
await store.aapply_vector_index(CSPANNIndex(), prefix_columns=["namespace"])

# The index reads only tenant-123's vectors; the filter applies to them
results = await store.asimilarity_search("query", k=10, filter={"category": "news"})
```

A key in `filter`, such as `"tenant_id"`, filters the JSON metadata. It never uses an index prefix.

### 3. Monitor and Tune Beam Size

```python
# Start with the default, 32
# Monitor query latency and recall
# If too slow: try 16
# If results miss near neighbors: try 64, then 128
```

### 4. Consider Workload Balance

| Workload | Strategy |
|----------|----------|
| Read-heavy | Create the index right away |
| Write-heavy | Create the index on the empty table, before writes start: building it on a table that holds data blocks writes until the backfill ends |
| Balanced | Create the index early, and monitor performance |

### 5. Index Before Production

```python
# Development: test without index first
# Staging: create index, benchmark
# Production: deploy with index
```

## Troubleshooting

### Index Not Being Used

Check query plan:
```sql
EXPLAIN SELECT * FROM documents 
ORDER BY embedding <=> '[...]' LIMIT 10;
```

If index not used:
- Check distance operator matches index opclass
- Check that the query constrains every prefix column to a value; a metadata filter alone never uses the index (see [Filtered Search](#filtered-search))
- Verify table statistics are up to date
- Use CockroachDB v25.4 or later, the oldest version this library supports

### Slow Index Creation

Large datasets take time:
```python
import time

start = time.time()
await vectorstore.aapply_vector_index(index)
elapsed = time.time() - start
print(f"Index created in {elapsed:.1f}s")
```

**Typical times:**
- 10K vectors: 1-5s
- 100K vectors: 10-30s
- 1M vectors: 1-5 minutes

### Accuracy Issues

Increase the beam size above the default of 32:
```python
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=64)
)
```

Or adjust partition sizes:
```python
index = CSPANNIndex(
    min_partition_size=5,   # Smaller = more accurate
    max_partition_size=50,  # Smaller = more accurate
)
```

## Next Steps

- [Hybrid Search](hybrid-search.md) - Combine indexes with FTS
- [Configuration](../getting-started/configuration.md) - Optimize for your workload
- [API Reference](../api/indexes.md) - Full index API

## Observe search execution

Pass a synchronous callback through `CSPANNQueryOptions` to receive one immutable
`SearchDiagnostics` record after each successful vector search:

```python
from langchain_cockroachdb import CSPANNQueryOptions

observations = []
results = await vectorstore.asimilarity_search(
    "query", k=10, filter={"category": "research"},
    query_options=CSPANNQueryOptions(beam_size=64, on_search=observations.append),
)
print(observations[-1])
```

The callback is optional and adds no SQL. It runs on the engine's event loop;
keep it fast and thread-safe, for example by forwarding events to a queue.
Callback exceptions are logged by type without their message and do not fail
retrieval. Failed searches do not emit a success record.

| Path | Meaning |
|------|---------|
| `direct` | No metadata filter; the database optimizer selects the plan |
| `filtered_exact` | Metadata filter without a usable indexed-window path |
| `filtered_index` | The filtered indexed window supplied the result |
| `exact_fallback` | The window could have missed matches, so an exact query followed |
| `empty` | A non-positive k returned no rows without SQL |

Records include requested k, returned count, elapsed seconds, and the window's
limit, candidate count, and returned match count when available. If a window
returns no matching rows, its candidate count is unknown (`None`). The match
count is capped by k. No query text, vectors, metadata, or tenant IDs are included.

Elapsed time includes connection checkout, retries, and index discovery, but
excludes embedding calls, MMR selection, and hybrid fusion. For MMR and hybrid
search, diagnostics describe the vector candidate search, so k is its candidate
pool size. The path is a library decision, not an `EXPLAIN` plan.

For a reproducible exact-versus-indexed comparison, run
`tests/performance/bench_vector_search.py`. Its README explains the JSON output,
filter selectivity, recall@k, latency percentiles, and measurement limits.
