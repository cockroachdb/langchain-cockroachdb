# Index Optimization

Performance optimization with C-SPANN indexes.

**File:** [`examples/vector_indexes.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/vector_indexes.py)

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
# Adjust beam size for accuracy/speed tradeoff
async with engine.engine.connect() as conn:
    # Higher beam = more accurate, slower
    await conn.execute(text("SET vector_search_beam_size = 200"))
    results = await vectorstore.asimilarity_search("query", k=10)
    
    # Lower beam = faster, less accurate
    await conn.execute(text("SET vector_search_beam_size = 50"))
    results = await vectorstore.asimilarity_search("query", k=10)
```

## Next Steps

- [Vector Indexes Guide](../guides/vector-indexes.md) - Complete guide
- [Configuration](../getting-started/configuration.md) - Tuning options
