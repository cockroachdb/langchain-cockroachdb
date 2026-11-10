# Index Optimization

Performance optimization with C-SPANN indexes.

**File:** [`examples/vector_indexes.py`](https://github.com/cockroachdb/langchain-cockroachdb/blob/main/examples/vector_indexes.py)

## Creating Indexes

```python
from langchain_cockroachdb import CSPANNIndex, DistanceStrategy

# Basic index
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)

# With distance strategy
index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
await vectorstore.aapply_vector_index(index)

# With partition tuning
index = CSPANNIndex(
    min_partition_size=10,
    max_partition_size=100,
)
await vectorstore.aapply_vector_index(index)
```

## Performance Comparison

```python
import time

# Without index
start = time.time()
results = await vectorstore.asimilarity_search("query", k=10)
no_index_time = time.time() - start

# Create index
index = CSPANNIndex()
await vectorstore.aapply_vector_index(index)

# With index
start = time.time()
results = await vectorstore.asimilarity_search("query", k=10)
with_index_time = time.time() - start

print(f"Without index: {no_index_time*1000:.1f}ms")
print(f"With index: {with_index_time*1000:.1f}ms")
print(f"Speedup: {no_index_time/with_index_time:.1f}x")
```

## Query-Time Tuning

```python
from langchain_cockroachdb import CSPANNQueryOptions

# The default beam size is 32. A larger beam is more accurate and slower
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=64)
)

# A smaller beam is faster and less accurate
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=16)
)
```

The store sets the beam size with `SET LOCAL` in the search's own transaction. A `SET` on a connection you open yourself does not reach the store's searches, which run on connections from the pool.

## Next Steps

- [Vector Indexes Guide](../guides/vector-indexes.md) - Complete guide
- [Configuration](../getting-started/configuration.md) - Tuning options
