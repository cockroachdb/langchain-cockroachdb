# Examples

Working code examples demonstrating langchain-cockroachdb features.

## Available Examples

All examples are available in the [`examples/`](https://github.com/cockroachdb/langchain-cockroachdb/tree/main/examples) directory of the repository.

### Basic Usage

| Example | Description | Features |
|---------|-------------|----------|
| [quickstart.py](basic-usage.md#quickstart) | Get started in 5 minutes | Basic CRUD, search |
| [sync_usage.py](basic-usage.md#sync-wrapper) | Synchronous API | Sync wrapper, simple scripts |

### Advanced Features

| Example | Description | Features |
|---------|-------------|----------|
| [vector_indexes.py](index-optimization.md) | Optimize query performance | C-SPANN indexes, tuning |
| [hybrid_search.py](advanced-filtering.md#hybrid-search) | Combine FTS + vectors | Fusion strategies |
| [metadata_filtering.py](advanced-filtering.md) | Complex metadata queries | Advanced filters |

### Configuration and Resilience

| Example | Description | Features |
|---------|-------------|----------|
| [retry_configuration.py](../getting-started/configuration.md#configuration-presets) | Production configurations | Retry logic, connection pooling |
| [resilience.py](../guides/resilience.md#trying-it) | Fault injection and retries | Real CockroachDB errors, retry behavior |

### Chat Applications

| Example | Description | Features |
|---------|-------------|----------|
| [chat_history.py](basic-usage.md#chat-history) | Persistent conversations | Session management |

### LangGraph Checkpointer

| Example | Description | Features |
|---------|-------------|----------|
| [checkpointer.py](checkpointer.md) | LangGraph state persistence | Short-term memory, human-in-the-loop |

### LangGraph Long-Term Memory

| Example | Description | Features |
|---------|-------------|----------|
| [long_term_memory.py](long-term-memory.md) | Agent memory across threads | Semantic search, TTL, `delete_namespace` |

### Multi-Tenancy

| Example | Description | Features |
|---------|-------------|----------|
| [multi_tenancy.py](../guides/multi-tenancy.md) | Namespace-based tenant isolation | Per-tenant stores, admin view |

## Running Examples

### Prerequisites

1. **CockroachDB running:**
   ```bash
   docker-compose up -d
   ```

2. **Install package:**
   ```bash
   pip install langchain-cockroachdb langchain-openai langchain
   ```

3. **Set connection string (optional):**
   ```bash
   export COCKROACHDB_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
   ```

4. **Set OpenAI key (for some examples):**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

### Run an Example

```bash
cd examples
python quickstart.py
```

## Example Categories

### Getting Started
- [Basic Usage](basic-usage.md) - CRUD operations, simple search
- Quick start guides for common scenarios

### Advanced Techniques
- [Advanced Filtering](advanced-filtering.md) - Complex queries, hybrid search
- [Index Optimization](index-optimization.md) - Performance tuning

### Production Patterns
- Configuration for different workloads
- Error handling and retry logic
- Multi-tenancy patterns

## Next Steps

- [Guides](../guides/vector-store.md) - Learn key concepts
- [API Reference](../api/vectorstore.md) - Detailed API docs
- [Configuration](../getting-started/configuration.md) - Optimize for your workload
