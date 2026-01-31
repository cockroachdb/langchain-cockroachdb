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
CREATE VECTOR INDEX ON documents (embedding);
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

Create indexes with prefix columns for tenant isolation:

```python
# Assume table has tenant_id column as first column
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)
```

**CockroachDB will automatically use prefix columns if:**
1. Table has non-vector columns before embedding column
2. Query filters on those columns

**Example:**
```sql
-- Table structure
CREATE TABLE documents (
    tenant_id UUID,
    collection_id TEXT,
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536),
    metadata JSONB
);

-- Index with prefix columns
CREATE VECTOR INDEX ON documents (tenant_id, collection_id, embedding);
```

**Benefits:**
- Each tenant's vectors are colocated
- Filters on prefix columns use index efficiently
- Better performance for multi-tenant apps

## Query-Time Tuning

### Beam Size

Adjust accuracy/speed tradeoff at query time:

```python
from langchain_cockroachdb import CSPANNQueryOptions

# Higher beam = more accurate, slower
query_options = CSPANNQueryOptions(beam_size=200)

# Lower beam = faster, less accurate
query_options = CSPANNQueryOptions(beam_size=50)

# Apply to search
results = await vectorstore.asimilarity_search(
    "query",
    k=10,
    # TODO: Pass query_options when implemented
)
```

Currently set via session:
```python
async with engine.engine.connect() as conn:
    await conn.execute(text("SET vector_search_beam_size = 200"))
    # Run queries
```

**Guidelines:**
- Default: 100 (good balance)
- High accuracy: 200-500
- Low latency: 25-50
- Production: 100-200

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
| Index + beam=50 | 5-15ms | Fast but less accurate |
| Index + beam=200 | 15-40ms | More accurate |

**Recommendation:** Create index, tune beam size for workload

### Large Dataset (1M+ vectors)

| Configuration | Query Latency | Notes |
|---------------|---------------|-------|
| No index | 5-30s | Unusable |
| With index | 20-100ms | Essential |
| Index + partitions | 10-50ms | Tuned partitions help |

**Recommendation:** Always use index, tune partitions and beam

## Index Opclasses

CockroachDB automatically chooses the correct opclass based on distance strategy:

| Distance Strategy | Opclass | Operator |
|-------------------|---------|----------|
| COSINE | `vector_cosine_ops` | `<=>` |
| EUCLIDEAN (L2) | `vector_l2_ops` | `<->` |
| INNER_PRODUCT | `vector_ip_ops` | `<#>` |

## Best Practices

### 1. Create Index After Bulk Insert

```python
# Insert data first
await vectorstore.aadd_texts(large_list_of_texts)

# Then create index
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)
```

Building index on existing data is faster than incremental updates.

### 2. Use Prefix Columns for Multi-Tenancy

```python
# Structure table with tenant_id first
# Then create index - it will use prefix automatically
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)

# Queries with tenant filter will be fast
results = await vectorstore.asimilarity_search(
    "query",
    k=10,
    filter={"tenant_id": "tenant-123"}  # Uses index prefix
)
```

### 3. Monitor and Tune Beam Size

```python
# Start with default (100)
# Monitor query latency
# If too slow: reduce beam (50-75)
# If accuracy issues: increase beam (150-200)
```

### 4. Consider Workload Balance

| Workload | Strategy |
|----------|----------|
| Read-heavy | Create index immediately |
| Write-heavy | Delay index creation |
| Balanced | Create index, monitor performance |

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
- Verify table statistics are up to date
- Ensure CockroachDB version supports vector indexes (23.1+)

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

Increase beam size:
```sql
SET vector_search_beam_size = 200;
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
