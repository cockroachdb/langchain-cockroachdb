# <img src="https://raw.githubusercontent.com/viragtripathi/langchain-cockroachdb/main/assets/cockroachdb_logo.svg" alt="🪳" width="25" height="25" style="vertical-align: middle;"/> langchain-cockroachdb

[![Tests](https://github.com/viragtripathi/langchain-cockroachdb/actions/workflows/test.yml/badge.svg)](https://github.com/viragtripathi/langchain-cockroachdb/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/langchain-cockroachdb.svg)](https://badge.fury.io/py/langchain-cockroachdb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/badge/langchain-cockroachdb/month)](https://pepy.tech/project/langchain-cockroachdb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**High-performance LangChain integration for CockroachDB with native vector support, optimized for distributed SQL workloads.**

---

This package provides LangChain abstractions backed by CockroachDB, leveraging CockroachDB's native `VECTOR` type and C-SPANN (CockroachDB SPANN) indexes for real-time vector search at massive scale in a distributed, horizontally scalable database.

## ✨ Features

### 🎯 Vector Store
- **Native Vector Support**: Uses CockroachDB's native `VECTOR` type (not pgvector)
- **C-SPANN Indexes**: CockroachDB's vector index algorithm for real-time indexing of billions of vectors
- **Multiple Distance Strategies**: Cosine similarity, Euclidean (L2), Inner Product
- **Async-First Design**: Full async/await support with sync wrappers
- **Optimized Batch Inserts**: Tuned for CockroachDB's distributed architecture
- **Rich Metadata Filtering**: Complex filters with `$and`, `$or`, `$gt`, `$lt`, `$in`, `$between`, etc.

### 🔍 Advanced Search
- **Similarity Search**: Fast vector similarity search with configurable k
- **MMR Search**: Max Marginal Relevance for diverse results
- **Hybrid Search**: Combine full-text search (FTS) with vector similarity
- **Score Fusion**: Weighted sum and Reciprocal Rank Fusion (RRF)
- **Query-Time Tuning**: Adjust beam size for accuracy/speed tradeoff

### 🏗️ Production-Ready
- **SERIALIZABLE Isolation**: Built for CockroachDB's default isolation level
- **Multi-Tenancy**: Index prefix columns for efficient tenant isolation
- **Connection Pooling**: Configurable connection pools with health checks
- **Horizontal Scalability**: Designed for distributed deployments
- **Chat History**: Persistent conversation storage

## 📦 Installation

```bash
pip install langchain-cockroachdb
```

## 🛠️ Development

This project uses a Makefile for common tasks:

```bash
make help                # Show all available commands
make dev                 # Setup development environment
make test                # Run tests
make lint                # Run linter
make examples            # Run all examples
```

See [MAKEFILE.md](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/MAKEFILE.md) for complete command reference.

Or with uv (recommended):

```bash
uv pip install langchain-cockroachdb
```

## 🚀 Quick Start

### Basic Vector Store

```python
from langchain_cockroachdb import CockroachDBVectorStore, CockroachDBEngine
from langchain_openai import OpenAIEmbeddings

# Initialize connection
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://user:password@localhost:26257/defaultdb?sslmode=require"
)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="my_documents",
)

# Initialize table (first time only)
engine.init_vectorstore_table(
    table_name="my_documents",
    vector_dimension=1536,  # OpenAI embedding dimension
)

# Add documents
texts = [
    "CockroachDB is a distributed SQL database",
    "LangChain helps build LLM applications",
    "Vector search enables semantic similarity",
]
ids = vectorstore.add_texts(texts)

# Search
results = vectorstore.similarity_search("What is CockroachDB?", k=3)
for doc in results:
    print(doc.page_content)
```

### Async Operations

```python
import asyncio
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

async def main():
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://user:password@localhost:26257/defaultdb"
    )
    
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="async_docs",
    )
    
    # Initialize table
    await engine.ainit_vectorstore_table("async_docs", vector_dimension=1536)
    
    # Add documents
    ids = await vectorstore.aadd_texts(texts)
    
    # Search with scores
    results = await vectorstore.asimilarity_search_with_score(
        "distributed databases", k=5
    )
    for doc, score in results:
        print(f"Score: {score:.4f} - {doc.page_content}")

asyncio.run(main())
```

## 📚 Advanced Usage

### Vector Indexes (C-SPANN)

CockroachDB's C-SPANN (CockroachDB SPANN) indexes provide fast approximate nearest neighbor search optimized for distributed environments. C-SPANN enables real-time indexing of billions of vectors with automatic sharding, rebalancing, and fresh results.

```python
from langchain_cockroachdb import CSPANNIndex, DistanceStrategy

# Create optimized index
index = CSPANNIndex(
    distance_strategy=DistanceStrategy.COSINE,
    min_partition_size=100,
    max_partition_size=1000,
)

# Apply index (async)
await vectorstore.aapply_vector_index(index)

# Or sync
vectorstore.apply_vector_index(index)
```

### Query-Time Tuning

Adjust search accuracy vs speed with beam size:

```python
from langchain_cockroachdb import CSPANNQueryOptions

# Higher beam size = more accurate, slower
results = await vectorstore.asimilarity_search(
    "my query",
    k=10,
    query_options=CSPANNQueryOptions(beam_size=200),  # Default is ~100
)

# Lower beam size = faster, less accurate
results = await vectorstore.asimilarity_search(
    "my query",
    k=10,
    query_options=CSPANNQueryOptions(beam_size=50),
)
```

### Metadata Filtering

Rich metadata filtering with SQL-like operators:

```python
# Add documents with metadata
vectorstore.add_texts(
    texts=["Document 1", "Document 2", "Document 3"],
    metadatas=[
        {"source": "web", "year": 2024, "category": "tech"},
        {"source": "pdf", "year": 2023, "category": "science"},
        {"source": "web", "year": 2024, "category": "science"},
    ],
)

# Simple equality filter
results = vectorstore.similarity_search(
    "query",
    filter={"source": "web"}
)

# Comparison operators
results = vectorstore.similarity_search(
    "query",
    filter={"year": {"$gte": 2024}}
)

# Complex filters with AND/OR
results = vectorstore.similarity_search(
    "query",
    filter={
        "$and": [
            {"source": {"$in": ["web", "pdf"]}},
            {"year": {"$gte": 2023}},
            {"category": "science"}
        ]
    }
)

# Range queries
results = vectorstore.similarity_search(
    "query",
    filter={"year": {"$between": [2020, 2024]}}
)
```

**Supported Filter Operators:**
- Equality: `{"field": "value"}`
- Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- Lists: `$in`, `$nin`
- Text: `$like`, `$ilike`
- Range: `$between`
- Existence: `$exists`
- Logical: `$and`, `$or`, `$not`

### Hybrid Search (FTS + Vector)

Combine full-text search with vector similarity for best results:

```python
from langchain_cockroachdb import HybridSearchConfig

# Enable hybrid search
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="hybrid_docs",
    hybrid_search_config=HybridSearchConfig(
        fts_weight=0.3,      # 30% full-text search
        vector_weight=0.7,   # 70% vector similarity
        fusion_type="weighted_sum",
    ),
)

# Initialize with FTS support
await engine.ainit_vectorstore_table(
    table_name="hybrid_docs",
    vector_dimension=1536,
    create_tsvector=True,  # Enable full-text search
)

# Search automatically uses both FTS and vectors
results = await vectorstore.asimilarity_search(
    "distributed SQL database with vector support",
    k=5,
)
```

**Fusion Strategies:**
- `weighted_sum`: Combine scores with configurable weights
- `reciprocal_rank_fusion` (RRF): Rank-based fusion (more robust to score scale differences)

### Max Marginal Relevance (MMR)

Get diverse results with MMR:

```python
# MMR balances relevance and diversity
results = await vectorstore.amax_marginal_relevance_search(
    query="machine learning",
    k=5,              # Return 5 results
    fetch_k=20,       # Fetch 20 candidates
    lambda_mult=0.5,  # 0=max diversity, 1=max relevance
)
```

### Multi-Tenant Deployments

Use prefix columns for efficient tenant isolation:

```python
# Create index with prefix columns
index = CSPANNIndex(
    distance_strategy=DistanceStrategy.COSINE,
)

await vectorstore.aapply_vector_index(
    index,
    prefix_columns=["tenant_id", "namespace"],  # Partition by tenant
)

# Queries automatically benefit from index prefixes
results = await vectorstore.asimilarity_search(
    "query",
    filter={"tenant_id": "acme-corp", "namespace": "prod"},
    k=10,
)
```

### Chat Message History

Persist conversation history in CockroachDB:

```python
from langchain_cockroachdb import CockroachDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create history
history = CockroachDBChatMessageHistory(
    session_id="user_123_conversation_456",
    connection_string="cockroachdb://...",
    table_name="chat_history",
)

# Create table (first time)
history.create_table_if_not_exists()

# Add messages
history.add_user_message("Hello!")
history.add_ai_message("Hi! How can I help you today?")

# Retrieve messages
messages = history.messages
for msg in messages:
    print(f"{msg.type}: {msg.content}")

# Clear history
history.clear()
```

### Batch Operations

Optimize bulk inserts with configurable batch sizes:

```python
# Large dataset
texts = ["Document " + str(i) for i in range(10000)]

# CockroachDB-optimized batch size (default: 100)
await vectorstore.aadd_texts(texts, batch_size=100)

# Adjust based on vector dimension and network latency
await vectorstore.aadd_texts(
    large_texts,
    batch_size=50,  # Smaller batches for high-dimensional vectors
)
```

### Custom Table Schema

Customize table and column names:

```python
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="custom_vectors",
    schema="public",
    content_column="text_content",
    embedding_column="vector_embedding",
    metadata_column="doc_metadata",
    id_column="document_id",
)
```

## 🏗️ CockroachDB-Specific Optimizations

### Why CockroachDB for Vector Search?

1. **Distributed by Design**: Horizontal scalability for massive vector datasets
2. **ACID Guarantees**: Full transactional consistency (SERIALIZABLE by default)
3. **High Availability**: Multi-region deployments with automatic failover
4. **Native Vectors**: First-class `VECTOR` type, not a plugin
5. **C-SPANN Indexes**: Purpose-built for distributed vector search
6. **No Single Point of Failure**: Unlike pgvector on a single Postgres instance

### Batch Size Recommendations

CockroachDB has specific guidance for vector operations:

| Vector Dimension | Recommended Batch Size |
|------------------|------------------------|
| < 512            | 200-500                |
| 512-1536         | 100-200                |
| > 1536           | 50-100                 |

```python
# Adjust batch size based on your embeddings
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=100,  # Default, optimized for OpenAI embeddings (1536 dims)
)
```

### Index Tuning

C-SPANN partition sizes affect performance:

```python
# For small datasets (< 10k vectors)
index = CSPANNIndex(
    min_partition_size=50,
    max_partition_size=500,
)

# For medium datasets (10k - 1M vectors)
index = CSPANNIndex(
    min_partition_size=100,
    max_partition_size=1000,
)

# For large datasets (> 1M vectors)
index = CSPANNIndex(
    min_partition_size=200,
    max_partition_size=2000,
)
```

### Connection Pooling

Optimize connection pooling for distributed workloads:

```python
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=10,          # Connections per engine
    max_overflow=20,       # Additional connections under load
    pool_pre_ping=True,    # Verify connections before use
)
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/cockroachdb/langchain-cockroachdb.git
cd langchain-cockroachdb

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
uv pip install -e ".[dev]"

# Start CockroachDB
docker-compose up -d

# Run tests
pytest tests/unit -v              # Unit tests (fast, no DB)
pytest tests/integration -v       # Integration tests (requires DB)
pytest tests --cov=langchain_cockroachdb  # With coverage
```

### Code Quality

```bash
# Linting
ruff check langchain_cockroachdb tests

# Auto-fix
ruff check langchain_cockroachdb tests --fix

# Type checking
mypy langchain_cockroachdb
```

## 📖 Documentation

- [Contributing Guidelines](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/CONTRIBUTING.md)
- [Development Guide](DEVELOPMENT.md)
- [Changelog](CHANGELOG.md)
- [CockroachDB Vector Indexes](https://www.cockroachlabs.com/docs/stable/vector-indexes)
- [LangChain Documentation](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/CONTRIBUTING.md) for guidelines.

### Quick Contribution Checklist

- [ ] Tests pass: `pytest tests -v`
- [ ] Linting passes: `ruff check langchain_cockroachdb tests`
- [ ] Type checking passes: `mypy langchain_cockroachdb`
- [ ] Documentation updated
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

## 📊 Performance

Benchmarks comparing CockroachDB with pgvector (coming soon):

- Insert throughput
- Query latency (p50, p95, p99)
- Index build time
- Scaling characteristics

## 🐛 Troubleshooting

### Connection Issues

```python
# Test connection
async with engine.engine.connect() as conn:
    result = await conn.execute(text("SELECT version()"))
    print(result.scalar())
```

### Enable SQL Logging

```python
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Check Vector Count

```python
async with engine.engine.connect() as conn:
    result = await conn.execute(
        text(f"SELECT COUNT(*) FROM {vectorstore.collection_name}")
    )
    print(f"Vectors: {result.scalar()}")
```

## 📝 License

Apache License 2.0 - see [LICENSE](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [CockroachDB](https://www.cockroachlabs.com/)

## 🔗 Links

- [CockroachDB Documentation](https://www.cockroachlabs.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)

---

**Built with ❤️ for the CockroachDB and LangChain communities**

## ⚙️ Configuration

### Production-Ready Retry Logic

All operations automatically retry on transient errors (40001 serialization failures, connection errors):

```python
# Engine-level retry configuration
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://...",
    retry_max_attempts=10,           # Max retries (default: 5)
    retry_initial_backoff=0.1,       # Initial delay in seconds (default: 0.1)
    retry_max_backoff=30.0,          # Max delay in seconds (default: 10.0)
    retry_backoff_multiplier=2.0,    # Exponential multiplier (default: 2.0)
    retry_jitter=True,               # Add randomization (default: True)
)

# VectorStore-level retry configuration (per-batch)
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    retry_max_attempts=5,            # Per-batch retries (default: 3)
    retry_initial_backoff=0.1,
    retry_max_backoff=5.0,
)
```

**Retry Logic Features:**
- ✅ Automatic detection of retryable errors (40001, connection failures)
- ✅ Exponential backoff with jitter (prevents thundering herd)
- ✅ Configurable per engine and per vectorstore
- ✅ Based on CockroachDB best practices

### Connection Pool Configuration

```python
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://...",
    pool_size=20,                    # Base connections (default: 10)
    max_overflow=40,                 # Additional connections (default: 20)
    pool_pre_ping=True,              # Health checks (default: True)
    pool_recycle=1800,               # Recycle after 30min (default: 3600)
    pool_timeout=30.0,               # Connection timeout (default: 30.0)
)
```

### Batch Size Configuration

```python
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=100,                  # Default batch size (default: 100)
)

# Override at runtime
ids = await vectorstore.aadd_texts(texts, batch_size=50)
```

**Batch Size Guidelines:**
- Small embeddings (< 512 dims): 100-500
- Large embeddings (> 1024 dims): 50-100
- High write contention: 10-50
- CockroachDB works best with smaller batches compared to single-node databases

### Configuration Presets

| Workload | Pool | Overflow | Retries | Backoff | Batch |
|----------|------|----------|---------|---------|-------|
| **Development** | 5 | 10 | 3 | 0.1s → 1s | 100 |
| **Production Web** | 20 | 40 | 10 | 0.1s → 30s | 100 |
| **Batch Jobs** | 3 | 5 | 20 | 0.5s → 60s | 50 |
| **Multi-Region** | 15 | 30 | 15 | 0.2s → 60s | 50 |
| **Low Latency** | 5 | 10 | 2 | 0.05s → 1s | 100 |

See `examples/retry_configuration.py` for detailed examples.

## 🔄 Async vs Sync

### When to Use Async (Recommended)

```python
from langchain_cockroachdb import AsyncCockroachDBVectorStore

# Async provides 10-100x throughput for distributed databases
vectorstore = AsyncCockroachDBVectorStore(...)
results = await vectorstore.asimilarity_search("query")
```

**Use async when:**
- Building production web applications
- High concurrent operations
- Distributed/multi-region CockroachDB
- Modern LLM apps (OpenAI/Anthropic APIs are async)

**Benefits:**
- Non-blocking I/O (handle 1000s of requests)
- Better throughput on network-bound operations
- Natural fit for async LLM APIs
- Efficient connection pool usage

### When to Use Sync

```python
from langchain_cockroachdb import CockroachDBVectorStore

# Sync wrapper for simple scripts
vectorstore = CockroachDBVectorStore(...)
results = vectorstore.similarity_search("query")
```

**Use sync when:**
- Simple scripts or batch jobs
- Sequential processing
- Legacy code without async/await
- Simpler mental model needed

See `examples/sync_usage.py` for complete example.

## 🧪 Testing

Comprehensive test coverage (92 tests across 9 test files):

```bash
# All tests
make test

# Unit tests only (fast, no Docker required)
python -m pytest tests/unit -v

# Integration tests (requires Docker)
python -m pytest tests/integration -v

# Specific test categories
python -m pytest tests/unit/test_retry.py -v          # Retry logic
python -m pytest tests/integration/test_sync_wrapper.py -v  # Sync wrapper
python -m pytest tests/integration/test_configuration.py -v # Configuration
```

### Test Structure

**Unit Tests (37 tests) - No database required:**
- `test_hybrid_search.py` (8 tests) - Hybrid search fusion logic
- `test_indexes.py` (12 tests) - Index SQL generation
- `test_retry.py` (17 tests) - Retry decorators, error detection, backoff

**Integration Tests (55 tests) - Requires Docker:**
- `test_engine.py` (8 tests) - Engine initialization, connection pooling
- `test_vectorstore.py` (14 tests) - Vector store operations
- `test_chat_history.py` (6 tests) - Chat message persistence
- `test_retry_behavior.py` (5 tests) - Real database retry scenarios
- `test_sync_wrapper.py` (12 tests) - Sync wrapper API
- `test_configuration.py` (10 tests) - Configuration parameters

### Test Categories by Feature

| Feature | Unit Tests | Integration Tests | Total |
|---------|-----------|-------------------|-------|
| Retry Logic | 17 | 5 | 22 |
| Sync Wrapper | 0 | 12 | 12 |
| Configuration | 0 | 10 | 10 |
| Vector Operations | 0 | 14 | 14 |
| Indexes | 12 | 0 | 12 |
| Hybrid Search | 8 | 0 | 8 |
| Engine | 0 | 8 | 8 |
| Chat History | 0 | 6 | 6 |
| **Total** | **37** | **55** | **92** |

**Requirements:**
- Unit tests: Just Python (no Docker)
- Integration tests: Docker Desktop running (uses testcontainers)

