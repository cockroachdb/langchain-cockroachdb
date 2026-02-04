# Advanced Filtering & Hybrid Search

Complex queries and hybrid search examples.

## Advanced Metadata Filtering

**File:** [`examples/metadata_filtering.py`](https://github.com/cockroachdb/langchain-cockroachdb/blob/main/examples/metadata_filtering.py)

### Comparison Operators

```python
# Greater than
results = await vectorstore.asimilarity_search(
    "recent articles",
    k=10,
    filter={"year": {"$gte": 2023}}
)

# Multiple conditions
results = await vectorstore.asimilarity_search(
    "high quality tech content",
    k=10,
    filter={
        "category": "tech",
        "score": {"$gte": 0.8},
        "year": {"$gte": 2023}
    }
)
```

### IN and NOT IN

```python
# IN operator
results = await vectorstore.asimilarity_search(
    "science content",
    k=10,
    filter={"source": {"$in": ["arxiv", "nature", "ieee"]}}
)

# NOT IN operator
results = await vectorstore.asimilarity_search(
    "published content",
    k=10,
    filter={"status": {"$nin": ["draft", "archived"]}}
)
```

### Logical Operators

```python
# OR operator
results = await vectorstore.asimilarity_search(
    "research",
    k=10,
    filter={
        "$or": [
            {"category": "science"},
            {"category": "research"}
        ]
    }
)

# Complex nested filters
results = await vectorstore.asimilarity_search(
    "innovation",
    k=10,
    filter={
        "$and": [
            {"year": {"$gte": 2023}},
            {
                "$or": [
                    {"category": "tech"},
                    {"category": "science"}
                ]
            },
            {"quality_score": {"$gte": 0.7}}
        ]
    }
)
```

## Hybrid Search

Combining full-text search with vector similarity.

**File:** [`examples/hybrid_search.py`](https://github.com/cockroachdb/langchain-cockroachdb/blob/main/examples/hybrid_search.py)

### Basic Hybrid Search

```python
from langchain_cockroachdb import HybridSearchConfig

# Configure hybrid search
hybrid_config = HybridSearchConfig(
    vector_weight=0.5,
    fts_weight=0.5,
    fusion_type="weighted_sum",
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    hybrid_search_config=hybrid_config,
)

# Enable FTS
await vectorstore.aapply_hybrid_search()

# Search with both methods
results = await vectorstore.ahybrid_search(
    "CockroachDB SERIALIZABLE isolation",
    k=10
)
```

### Fusion Strategies

```python
# Weighted sum (emphasize semantic)
hybrid_config = HybridSearchConfig(
    vector_weight=0.7,
    fts_weight=0.3,
    fusion_type="weighted_sum",
)

# Reciprocal Rank Fusion
hybrid_config = HybridSearchConfig(
    fusion_type="reciprocal_rank_fusion",
    rrf_k=60,
)
```

## Multi-Tenant Patterns

```python
# Add tenant isolation
texts = ["Doc 1", "Doc 2", "Doc 3"]
metadatas = [
    {"tenant_id": "tenant-123", "category": "tech"},
    {"tenant_id": "tenant-456", "category": "science"},
    {"tenant_id": "tenant-123", "category": "tech"},
]

await vectorstore.aadd_texts(texts, metadatas=metadatas)

# Query with tenant filter
results = await vectorstore.asimilarity_search(
    "tech content",
    k=10,
    filter={"tenant_id": "tenant-123"}
)
```

## Next Steps

- [Index Optimization](index-optimization.md) - Performance tuning
- [Hybrid Search Guide](../guides/hybrid-search.md) - Detailed guide
- [Vector Store Guide](../guides/vector-store.md) - Full API
