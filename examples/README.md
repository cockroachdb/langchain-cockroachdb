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

# Resilience - real CockroachDB faults, retried automatically
python examples/resilience.py

# Vector indexes - C-SPANN index creation and tuning
python examples/vector_indexes.py

# Hybrid search - combining FTS and vector search
python examples/hybrid_search.py

# Metadata filtering - advanced filter queries
python examples/metadata_filtering.py

# Chat history - conversation persistence
python examples/chat_history.py

# Checkpointer - LangGraph state persistence
python examples/checkpointer.py

# Long-term memory - LangGraph store with semantic search and TTL
python examples/long_term_memory.py

# Multi-tenancy - namespace-based tenant isolation
python examples/multi_tenancy.py
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

Shows RetryPolicy presets, per-component overrides and the multi-region failover policy:
- Default configuration (development)
- High-performance configuration (production web apps)
- Low-latency configuration (single-region)
- Batch job configuration (resilient long-running)
- Multi-region configuration (high latency tolerance)
- Runtime configuration override
- Configuration guidelines by workload

**Good for**: Production deployments, performance tuning, different workload patterns

### 4. resilience.py

Injects real CockroachDB errors (40001, 40003, 57P01) and kills pooled connections while the
vector store, chat history and checkpointer keep working. Shows the WARNING line each retry logs.
Needs a local cluster you can run admin statements on.

**Good for**: Understanding retry behavior, testing fault tolerance

### 5. vector_indexes.py

Vector index management:
- Creating C-SPANN indexes
- Configuring partition sizes
- Query-time beam size tuning
- Multiple distance strategies
- Multi-tenant prefix columns

**Good for**: Performance optimization, production deployments

### 6. hybrid_search.py

Combining FTS and vector search:
- TSVECTOR column creation
- Weighted score fusion
- Reciprocal Rank Fusion (RRF)
- Fusion strategy comparison

**Good for**: Best search quality, combining keyword + semantic

### 7. metadata_filtering.py

Advanced metadata queries:
- Equality and comparison operators
- IN/NOT IN operators
- Logical AND/OR combinations
- Nested filter expressions
- Complex filter patterns

**Good for**: Multi-tenant applications, filtered search

### 8. chat_history.py

Persistent conversation storage:
- Session management
- Message persistence
- Session isolation
- Integration with LangChain chains

**Good for**: Chatbots, conversational AI

### 9. checkpointer.py

LangGraph checkpointer with CockroachDB:
- Sync and async checkpointer usage
- Connection pool mode
- Saving, retrieving, and listing checkpoints
- Pending sends migration
- Thread deletion

**Good for**: LangGraph agents, short-term memory, human-in-the-loop workflows

### 10. multi_tenancy.py

Namespace-based multi-tenancy:
- Opt-in namespace column creation
- Per-tenant vector store instances
- Search, delete, and get_by_ids isolation
- Admin view (no namespace) for cross-tenant access

**Good for**: SaaS applications, multi-tenant RAG pipelines

### 11. long_term_memory.py

LangGraph long-term memory store with CockroachDB:
- A LangGraph agent that saves a memory in one thread and recalls it in another
- Semantic search on a C-SPANN index, with a scripted model and fake embeddings (no API key)
- Row-level TTL, and items that never expire
- Store and checkpointer on one connection pool
- Listing namespaces and deleting a user with `delete_namespace`
- Sync and async stores

**Good for**: LangGraph and Deep Agents memory, per-user data, requests to forget a user

## Next Steps

After running examples, explore:

- [DEVELOPMENT.md](../DEVELOPMENT.md) - Architecture and patterns
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [README.md](../README.md) - Full documentation
