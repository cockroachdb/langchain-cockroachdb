# Hybrid Search Guide

Combine full-text search with vector similarity for better retrieval.

## Overview

Hybrid search runs two searches for every query and merges the results:

- **Full-text search (FTS)**: keyword matching over a TSVECTOR column using `ts_rank`
- **Vector search**: semantic similarity over embeddings
- **Score fusion**: the two ranked lists are merged with reciprocal rank fusion or a weighted sum

Both searches run in parallel, so the latency cost over a plain vector search is small.

**Benefits:**

- Better recall: exact keywords and semantic matches both surface
- Robust to query formulation
- Keyword-only matches that vector search would miss still come back as full documents

## Setup

Hybrid search needs a tsvector column and GIN index on the table. Create the table with `create_tsvector=True` and pass a `HybridSearchConfig` to the store:

```python
from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    HybridSearchConfig,
)

engine = CockroachDBEngine.from_connection_string(connection_string)

# create_tsvector=True adds the tsvector column and GIN index
await engine.ainit_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
    create_tsvector=True,
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    hybrid_search_config=HybridSearchConfig(),
)
```

Behind the scenes the table gets:

```sql
ALTER TABLE documents
ADD COLUMN content_tsvector TSVECTOR
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ON documents USING GIN (content_tsvector);
```

## Searching

Once the config is set, the normal search methods do hybrid search transparently. There is no separate hybrid method to call:

```python
# Runs FTS and vector search in parallel, fuses, returns top k
results = await vectorstore.asimilarity_search(
    "distributed databases performance", k=10
)

for doc in results:
    print(doc.page_content)
```

With scores (these are fused scores, not raw distances):

```python
results = await vectorstore.asimilarity_search_with_score(
    "CockroachDB scalability", k=10
)

for doc, score in results:
    print(f"{score:.4f}  {doc.page_content[:80]}")
```

Metadata filters apply to both the FTS and vector sides:

```python
results = await vectorstore.asimilarity_search(
    "database query",
    k=10,
    filter={"category": "tech", "year": {"$gte": 2023}},
)
```

The sync store works the same way:

```python
from langchain_cockroachdb import CockroachDBVectorStore

store = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    hybrid_search_config=HybridSearchConfig(),
)
results = store.similarity_search("distributed databases", k=10)
```

## Fusion Strategies

### Reciprocal Rank Fusion (default)

RRF combines results by rank position, ignoring raw scores entirely:

```
score(doc) = sum(weight / (k + rank(doc))) over both searches
```

```python
config = HybridSearchConfig(
    fusion_type="reciprocal_rank_fusion",
    k=60,  # RRF constant, higher spreads the top ranks less
)
```

RRF is the default because ts_rank scores and vector distances live on
completely different scales, and rank based fusion sidesteps that problem.
It is also what most search engines default to for hybrid retrieval.

**Use when:**

- You want a solid default without tuning
- No strong preference between keyword and semantic matching

### Weighted Sum

Both score sets are min-max normalized to [0, 1] first, then combined:

```
final_score = vector_weight * vector_score + fts_weight * fts_score
```

```python
config = HybridSearchConfig(
    fusion_type="weighted_sum",
    vector_weight=0.7,  # emphasize semantic similarity
    fts_weight=0.3,     # some weight for keywords
)
```

**Use when:**

- You clearly trust one method more than the other
- You want scores that reflect relative strength, not just rank

## Tuning

### Weights

Weights apply to both fusion strategies.

```python
# Conceptual queries, synonym matching
HybridSearchConfig(vector_weight=0.8, fts_weight=0.2)

# Exact terms, error codes, technical identifiers
HybridSearchConfig(vector_weight=0.3, fts_weight=0.7)

# General purpose
HybridSearchConfig(vector_weight=0.5, fts_weight=0.5)
```

### Candidate pool size

Each search fetches a candidate pool before fusion, `max(k * 4, 20)` by
default. Fetch more when you want fusion to consider a wider net:

```python
results = await vectorstore.asimilarity_search(
    "query",
    k=10,       # return top 10 after fusion
    fetch_k=50, # but pull top 50 from each search first
)
```

### Rank normalization for long documents

Raw `ts_rank` favors long documents. The `fts_rank_normalization` bitmask is
passed straight through to `ts_rank` to damp that:

```python
# 1 divides by 1 + log(document length), 32 maps ranks to rank / (rank + 1)
config = HybridSearchConfig(fts_rank_normalization=1 | 32)
```

See the PostgreSQL `ts_rank` documentation for all flag values.

### Language

The tsvector column and the query must use the same text search
configuration. Set it in both places:

```python
await engine.ainit_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
    create_tsvector=True,
    fts_language="spanish",
)

config = HybridSearchConfig(fts_query_language="spanish")
```

## CockroachDB specifics

- Queries are parsed with `plainto_tsquery`, which ANDs the query terms.
  CockroachDB does not support `websearch_to_tsquery`.
- A query whose keywords match nothing simply contributes an empty FTS list,
  and you get vector results as usual.
- The query text is passed to the database as a bound parameter.

## Performance Considerations

Index both sides for large tables:

```python
from langchain_cockroachdb import CSPANNIndex

# Vector index (C-SPANN)
await vectorstore.aapply_vector_index(CSPANNIndex())

# The GIN index for FTS was already created by create_tsvector=True
```

## Troubleshooting

### Column does not exist errors on search

The table was created without `create_tsvector=True`. Recreate it, or add
the column manually following the SQL shown in Setup.

### Poor hybrid results

Try the other fusion strategy first, then adjust weights:

```python
config = HybridSearchConfig(fusion_type="weighted_sum", vector_weight=0.8, fts_weight=0.2)
```

If long documents dominate, set `fts_rank_normalization=1` or `1 | 32`.

### No FTS influence on results

`plainto_tsquery` ANDs all terms, so multi-word queries only produce FTS hits
for documents containing every term. Check what the query parses to:

```sql
SELECT plainto_tsquery('english', 'your query here');
```

## Next Steps

- [Vector Indexes](vector-indexes.md) - Optimize vector search
- [Configuration](../getting-started/configuration.md) - Tune performance
- [API Reference](../api/vectorstore.md) - Full API details
