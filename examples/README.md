# Examples

Comprehensive examples demonstrating langchain-cockroachdb features.

## Running Examples

### Prerequisites

Start CockroachDB locally:

```bash
docker-compose up -d
```

Set connection string (optional):

```bash
export COCKROACHDB_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
```

### Run Examples

```bash
# Quickstart - basic vector store usage
python examples/quickstart.py

# Sync usage - synchronous wrapper for simple scripts
python examples/sync_usage.py

# Retry configuration - production configuration presets
python examples/retry_configuration.py

# Vector indexes - C-SPANN index creation and tuning
python examples/vector_indexes.py

# Hybrid search - combining FTS and vector search
python examples/hybrid_search.py

# Metadata filtering - advanced filter queries
python examples/metadata_filtering.py

# Chat history - conversation persistence
python examples/chat_history.py
```

## Examples Overview

### 1. quickstart.py (Async)

Basic async vector store operations:
- Connecting to CockroachDB
- Creating tables
- Adding documents
- Similarity search
- Filtering by metadata

**Good for**: First-time users, getting started

### 2. sync_usage.py (Sync Wrapper)

Synchronous wrapper for simple scripts:
- CockroachDBVectorStore (sync API)
- All CRUD operations in sync mode
- When to use sync vs async
- Sync wrapper benefits and tradeoffs

**Good for**: Simple scripts, batch jobs, legacy code without async/await

### 3. retry_configuration.py (Configuration)

Production-ready configuration examples:
- Default configuration (development)
- High-performance configuration (production web apps)
- Low-latency configuration (single-region)
- Batch job configuration (resilient long-running)
- Multi-region configuration (high latency tolerance)
- Runtime configuration override
- Configuration guidelines by workload

**Good for**: Production deployments, performance tuning, different workload patterns

### 4. vector_indexes.py

Vector index management:
- Creating C-SPANN indexes
- Configuring partition sizes
- Query-time beam size tuning
- Multiple distance strategies
- Multi-tenant prefix columns

**Good for**: Performance optimization, production deployments

### 5. hybrid_search.py

Combining FTS and vector search:
- TSVECTOR column creation
- Weighted score fusion
- Reciprocal Rank Fusion (RRF)
- Fusion strategy comparison

**Good for**: Best search quality, combining keyword + semantic

### 6. metadata_filtering.py

Advanced metadata queries:
- Equality and comparison operators
- IN/NOT IN operators
- Logical AND/OR combinations
- Nested filter expressions
- Complex filter patterns

**Good for**: Multi-tenant applications, filtered search

### 7. chat_history.py

Persistent conversation storage:
- Session management
- Message persistence
- Session isolation
- Integration with LangChain chains

**Good for**: Chatbots, conversational AI

## Next Steps

After running examples, explore:

- [DEVELOPMENT.md](../DEVELOPMENT.md) - Architecture and patterns
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [README.md](../README.md) - Full documentation
