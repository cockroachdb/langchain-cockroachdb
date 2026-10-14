# Architecture

Understanding the internal structure of langchain-cockroachdb.

## Package Structure

```
langchain_cockroachdb/
├── __init__.py              # Public API exports
├── engine.py                # Connection management
├── async_vectorstore.py     # Async vector store implementation
├── vectorstores.py          # Sync wrapper
├── indexes.py               # Index abstractions
├── hybrid_search_config.py  # Hybrid search configuration
├── chat_message_histories.py # Chat persistence
├── retry.py                 # Retry logic
├── _guard.py                # Hang guard (attempt timeout)
├── _loop.py                 # Background event loop for sync methods
├── _connection.py           # Connection helpers
├── _sql.py                  # SQL generation
└── _filters.py              # Metadata filter translation
```

## Core Components

### CockroachDBEngine

**Purpose:** Connection and transaction management

**Key Features:**
- AsyncEngine from SQLAlchemy
- Connection pooling
- Retry configuration
- Table initialization

**Pattern:**
```python
engine = CockroachDBEngine.from_connection_string(url)
# Engine manages connection lifecycle
await engine.aclose()
```

### AsyncCockroachDBVectorStore

**Purpose:** Primary vector store implementation

**Key Features:**
- Async-first design
- Batch operations
- Metadata filtering
- Index management

**Pattern:**
```python
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
)
await vectorstore.aadd_texts(texts)
```

### CockroachDBVectorStore

**Purpose:** Synchronous wrapper for async operations

**Implementation:**
- Runs async methods on one background event loop
- Same API as async version (without `a` prefix)
- Safe from any number of threads and from inside a running event loop

**Pattern:**
```python
vectorstore = CockroachDBVectorStore(...)  # Sync
vectorstore.add_texts(texts)  # No await
```

## Design Patterns

### Async-First Architecture

All I/O operations are async by default:

```python
# Core operations
async def aadd_texts(texts: list[str]) -> list[str]:
    async with self.engine.engine.connect() as conn:
        # Async database operations
        ...
```

**Benefits:**
- High concurrency
- Better resource utilization
- Natural fit for modern Python

### Retry Decorators

Each idempotent statement is wrapped by `RetryPolicy.arun`. Connection checkout uses `engine._connect`, which provides the hang guard (attempt timeout):

```python
async def aadd_texts(texts: list[str]) -> list[str]:
    async with self.engine._connect() as conn:
        await self.retry_policy.arun(
            self._insert_batch(conn, batch),
            label="insert into documents",
        )
```

**Benefits:**
- Transparent retry logic
- Configurable per component
- Handles transient CockroachDB errors automatically

### Dependency Injection

Engine is injected into vector store:

```python
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,  # Injected dependency
    embeddings=embeddings,
    collection_name="docs",
)
```

**Benefits:**
- Testable (can mock engine)
- Flexible configuration
- Connection reuse

## Database Schema

### Vector Store Table

```sql
CREATE TABLE {table_name} (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(n),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### Chat History Table

```sql
CREATE TABLE message_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    message JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON message_store (session_id);
```

## SQL Generation

### Filter Translation

Metadata filters are translated to SQL:

```python
filter = {"category": "tech", "year": {"$gte": 2023}}

# Becomes:
WHERE metadata->>'category' = 'tech'
  AND CAST(metadata->>'year' AS INTEGER) >= 2023
```

### Distance Operators

Different distance strategies use different operators:

| Strategy | Operator | SQL |
|----------|----------|-----|
| COSINE | `<=>` | `ORDER BY embedding <=> :query` |
| EUCLIDEAN | `<->` | `ORDER BY embedding <-> :query` |
| INNER_PRODUCT | `<#>` | `ORDER BY embedding <#> :query` |

## Error Handling

### Retryable Errors

| SQLSTATE | Meaning | Why a retry is safe |
|---|---|---|
| `40001` | Serialization failure ("restart transaction") | The transaction did not commit |
| `40003` | Result is ambiguous | Every write the library issues is idempotent |
| `08xxx` | Connection exceptions, including CockroachDB sqlproxy's `08C00` | Idempotent writes, and the pool replaces the connection |
| `57P01` | Node draining or shutting down | Idempotent writes |
| `58C00` | Range unavailable (CockroachDB) | Usually clears once leases move |
| `58C01` | Connection failure between CockroachDB nodes | Usually clears once leases move |
| `53300` | Too many connections | Connection slots free up |

Errors without a SQLSTATE and CockroachDB's uncategorized codes (`XXUUU`, `XX000`) are matched against known connection-loss phrases.

See [Retries and Resilience](../guides/resilience.md) for full details.

## Testing Strategy

### Unit Tests

Test logic without database:
- Filter SQL generation
- Index configuration
- Retry logic

### Integration Tests

Test with real CockroachDB:
- Vector operations
- Index creation
- Concurrent operations

### Testcontainers

Use Docker containers for integration tests:
```python
@pytest.fixture
async def crdb_container():
    container = CockroachDBContainer()
    container.start()
    yield container.get_connection_url()
    container.stop()
```

## Performance Considerations

### Connection Pooling

- Pool size: 10 (default)
- Max overflow: 20 (default)
- Pre-ping: Enabled

### Batch Operations

- Default batch size: 100
- Configurable per operation
- Smaller batches for large vectors

### Query Optimization

- Use indexes for large datasets
- Filter before vector search when possible
- Tune beam size for accuracy/speed tradeoff

## Extension Points

### Custom Distance Functions

Extend with custom distance strategies:
```python
class CustomDistance(DistanceStrategy):
    name = "custom"
    operator = "custom_op"
```

### Custom Filters

Extend filter translator for custom operators:
```python
def _translate_filter(self, filter: dict) -> str:
    # Custom filter logic
```

## Next Steps

- [Contributing](contributing.md) - Join development
- [Testing](testing.md) - Write tests
- [API Reference](../api/engine.md) - Detailed API
