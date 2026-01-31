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
└── retry.py                 # Retry logic
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
- Runs async methods in background event loop
- Same API as async version (without `a` prefix)
- Uses `asyncio.run()` internally

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

All operations use retry decorators:

```python
@async_retry_with_backoff(
    max_attempts=self.retry_max_attempts,
    initial_backoff=self.retry_initial_backoff,
    ...
)
async def _insert_batch(batch: list) -> None:
    # Operation with automatic retry
```

**Benefits:**
- Transparent retry logic
- Configurable per operation
- Handles 40001 errors automatically

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

Automatically retry on:
- `40001`: Serialization errors
- Connection errors
- Timeout errors

### Non-Retryable Errors

Fail immediately on:
- Schema errors
- Permission errors
- Data validation errors

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
