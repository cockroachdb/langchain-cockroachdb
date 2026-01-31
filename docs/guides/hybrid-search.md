# Hybrid Search Guide

Combine full-text search with vector similarity for best search results.

## Overview

Hybrid search combines:
- **Full-Text Search (FTS)**: Keyword matching using PostgreSQL's TSVECTOR
- **Vector Search**: Semantic similarity using embeddings
- **Score Fusion**: Merges results using weighted sum or RRF

**Benefits:**
- Better recall (finds more relevant results)
- Handles both exact keywords and semantic meaning
- Robust to query formulation

## Configuration

### Enable Hybrid Search

```python
from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    HybridSearchConfig,
)

engine = CockroachDBEngine.from_connection_string(connection_string)

# Initialize table with FTS support
await engine.ainit_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
)

# Create vector store with hybrid search
hybrid_config = HybridSearchConfig(
    vector_weight=0.5,      # Weight for vector search (0-1)
    fts_weight=0.5,         # Weight for FTS (0-1)
    fusion_type="weighted_sum",  # or "reciprocal_rank_fusion"
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    hybrid_search_config=hybrid_config,
)
```

### Create FTS Index

```python
# Create TSVECTOR column and GIN index
await vectorstore.aapply_hybrid_search()
```

This creates:
```sql
-- Add tsvector column
ALTER TABLE documents 
ADD COLUMN content_tsvector TSVECTOR 
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Create GIN index
CREATE INDEX ON documents USING GIN (content_tsvector);
```

## Fusion Strategies

### Weighted Sum

Combines scores with weights:

```python
hybrid_config = HybridSearchConfig(
    vector_weight=0.7,      # Emphasize semantic similarity
    fts_weight=0.3,         # Some weight for keywords
    fusion_type="weighted_sum",
)
```

**Formula:**
```
final_score = (vector_weight * vector_score) + (fts_weight * fts_score)
```

**Use when:**
- You trust one method more than the other
- Clear preference for semantic vs keyword matching

### Reciprocal Rank Fusion (RRF)

Combines based on rank position:

```python
hybrid_config = HybridSearchConfig(
    fusion_type="reciprocal_rank_fusion",
    rrf_k=60,  # RRF constant (higher = less emphasis on top ranks)
)
```

**Formula:**
```
score(doc) = sum(1 / (k + rank(doc, method))) for each method
```

**Use when:**
- No clear preference between methods
- Want balanced results
- Popular in academic research

## Searching

### Basic Hybrid Search

```python
# Performs both FTS and vector search, then fuses results
results = await vectorstore.ahybrid_search(
    "distributed databases performance",
    k=10
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### With Scores

```python
results = await vectorstore.ahybrid_search_with_score(
    "CockroachDB scalability",
    k=10
)

for doc, score in results:
    print(f"Score: {score:.4f} - {doc.page_content[:100]}")
```

### With Filters

```python
# Apply metadata filters
results = await vectorstore.ahybrid_search(
    "database query",
    k=10,
    filter={"category": "tech", "year": {"$gte": 2023}}
)
```

## Tuning Weights

### Emphasize Semantic Search

```python
# Good for: Conceptual queries, synonym matching
hybrid_config = HybridSearchConfig(
    vector_weight=0.8,
    fts_weight=0.2,
)
```

**Example queries:**
- "What's a good scalable database?" (no keyword "CockroachDB")
- "distributed SQL solutions" (semantic understanding)

### Emphasize Keyword Search

```python
# Good for: Exact term matching, technical queries
hybrid_config = HybridSearchConfig(
    vector_weight=0.3,
    fts_weight=0.7,
)
```

**Example queries:**
- "CockroachDB SERIALIZABLE isolation" (exact terms)
- "error code 40001" (precise matching)

### Balanced

```python
# Good for: General purpose, unknown query types
hybrid_config = HybridSearchConfig(
    vector_weight=0.5,
    fts_weight=0.5,
)
```

## Advanced Configuration

### Language Configuration

```python
# FTS supports multiple languages
await vectorstore.aapply_hybrid_search(language="english")

# Other languages: spanish, french, german, etc.
await vectorstore.aapply_hybrid_search(language="spanish")
```

### Custom TSVECTOR Column

```python
# Use different text column for FTS
hybrid_config = HybridSearchConfig(
    tsvector_column="title_tsvector",  # Index title instead of content
)
```

### Fetch More Candidates

```python
# Fetch more results from each method before fusion
results = await vectorstore.ahybrid_search(
    "query",
    k=10,            # Return top 10
    fetch_k=50,      # But fetch top 50 from each method first
)
```

## Performance Considerations

### Index Both Types

```python
# Create vector index
from langchain_cockroachdb import CSPANNIndex
vector_index = CSPANNIndex()
await vectorstore.aapply_vector_index(vector_index)

# Create FTS index
await vectorstore.aapply_hybrid_search()
```

**Result:** Fast hybrid search with both indexes

### Batch Queries

```python
# For multiple queries, reuse connection
queries = ["query1", "query2", "query3"]
all_results = []

for query in queries:
    results = await vectorstore.ahybrid_search(query, k=5)
    all_results.append(results)
```

## Comparison: Vector vs FTS vs Hybrid

### Vector Search Only

```python
results = await vectorstore.asimilarity_search(
    "scalable databases", k=10
)
```

**Strengths:**
- Understands semantic meaning
- Handles synonyms and paraphrasing
- Works across languages (with multilingual embeddings)

**Weaknesses:**
- May miss exact keyword matches
- Embedding quality matters
- Slower for very large datasets

### FTS Only

```python
# (Hypothetical FTS-only method)
results = await vectorstore.afts_search(
    "CockroachDB SERIALIZABLE", k=10
)
```

**Strengths:**
- Fast exact keyword matching
- No embedding computation needed
- Good for technical terms

**Weaknesses:**
- No semantic understanding
- Misses synonyms
- Query formulation matters

### Hybrid (Best of Both)

```python
results = await vectorstore.ahybrid_search(
    "scalable database systems", k=10
)
```

**Strengths:**
- Combines semantic + keyword matching
- More robust results
- Better recall

**Weaknesses:**
- Slightly slower (two searches)
- More complex configuration
- Requires both indexes

## Example Use Cases

### Documentation Search

```python
# Users might search exact error codes or concepts
hybrid_config = HybridSearchConfig(
    vector_weight=0.4,
    fts_weight=0.6,  # Prefer exact matches
)
```

### E-commerce Product Search

```python
# Balance between product names and descriptions
hybrid_config = HybridSearchConfig(
    vector_weight=0.5,
    fts_weight=0.5,
)
```

### Academic Paper Search

```python
# Semantic understanding important
hybrid_config = HybridSearchConfig(
    vector_weight=0.7,
    fts_weight=0.3,
)
```

### Code Search

```python
# Exact function/variable names matter
hybrid_config = HybridSearchConfig(
    vector_weight=0.3,
    fts_weight=0.7,
)
```

## Troubleshooting

### FTS Not Working

Check if tsvector column and index exist:
```python
async with engine.engine.connect() as conn:
    from sqlalchemy import text
    result = await conn.execute(
        text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' 
            AND column_name LIKE '%tsvector%'
        """)
    )
    print(list(result))
```

### Poor Hybrid Results

Try different fusion strategies:
```python
# Try RRF instead of weighted sum
hybrid_config = HybridSearchConfig(
    fusion_type="reciprocal_rank_fusion",
)
```

Or adjust weights:
```python
# If semantic search alone works better, increase vector weight
hybrid_config = HybridSearchConfig(
    vector_weight=0.8,
    fts_weight=0.2,
)
```

## Next Steps

- [Vector Indexes](vector-indexes.md) - Optimize vector search
- [Configuration](../getting-started/configuration.md) - Tune performance
- [API Reference](../api/vectorstore.md) - Full API details
