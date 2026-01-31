# Development Guide

This guide provides detailed information for developers working on langchain-cockroachdb.

## Architecture Overview

### Package Structure

```
langchain-cockroachdb/
├── langchain_cockroachdb/
│   ├── __init__.py              # Public API exports
│   ├── engine.py                # CockroachDBEngine - connection management
│   ├── async_vectorstore.py     # AsyncCockroachDBVectorStore - core async implementation
│   ├── vectorstores.py          # CockroachDBVectorStore - sync wrapper
│   ├── indexes.py               # C-SPANN index abstractions
│   ├── hybrid_search_config.py  # Hybrid search configuration
│   └── chat_message_histories.py # Chat message persistence
├── tests/
│   ├── unit/                    # Unit tests (no database)
│   └── integration/             # Integration tests (with CockroachDB)
├── examples/                    # Usage examples
└── docs/                        # Documentation
```

### Design Philosophy

1. **Mirror LangChain Patterns**: Follow LangChain's `VectorStore` interface for consistency
2. **CockroachDB-Optimized**: Leverage CockroachDB-specific features (C-SPANN, SERIALIZABLE)
3. **Async-First**: Primary implementation is async, sync is a thin wrapper
4. **Type Safe**: Full type hints throughout
5. **Well-Tested**: Comprehensive unit and integration tests

## Key Components

### CockroachDBEngine

Manages async SQLAlchemy engine and provides helper methods.

**Key Features:**
- Connection pooling with health checks
- Table creation helpers
- Async context manager support
- Sync/async wrapper pattern

**Implementation Notes:**
- Uses `create_async_engine` from SQLAlchemy
- Configures pool for CockroachDB defaults
- Provides `ainit_vectorstore_table()` for schema setup

### AsyncCockroachDBVectorStore

Core vector store implementation using CockroachDB native vectors.

**Key Features:**
- Native `VECTOR` type support
- C-SPANN vector indexes
- Metadata filtering with complex operators
- Hybrid search (FTS + vector)
- MMR (Max Marginal Relevance) search
- Batched inserts optimized for CockroachDB

**Implementation Notes:**
- Batch size defaults to 100 (CockroachDB recommendation for vectors)
- Uses parameterized queries for safety
- Supports multiple distance strategies (cosine, L2, inner product)
- Filter translation to SQL WHERE clauses

### C-SPANN Indexes

C-SPANN (CockroachDB SPANN) is CockroachDB's vector indexing algorithm for real-time indexing of billions of vectors at scale. Key characteristics:

CockroachDB's native vector index implementation.

**Key Features:**
- Configurable partition sizes
- Multiple distance strategies
- Prefix column support for multi-tenancy
- Query-time beam size tuning

**Implementation Notes:**
- Generates `CREATE INDEX ... USING VECTOR`
- Maps distance strategies to operator classes
- Supports session-level search tuning

## Development Patterns

### Async/Sync Duality

The package provides both async and sync interfaces:

```python
# Async (primary implementation)
class AsyncCockroachDBVectorStore:
    async def aadd_texts(self, texts: list[str]) -> list[str]:
        # Implementation

# Sync (wrapper)
class CockroachDBVectorStore(AsyncCockroachDBVectorStore):
    def add_texts(self, texts: list[str]) -> list[str]:
        return asyncio.run(self.aadd_texts(texts))
```

**Why:**
- Some users prefer sync APIs for simplicity
- Async is more efficient for I/O-bound operations
- Matches LangChain patterns

### Transaction Retries

CockroachDB uses SERIALIZABLE isolation by default, which may cause retries.

**Best Practices:**
- Keep transactions small and fast
- Use single-statement operations when possible
- Implement retry logic for multi-statement transactions
- Consider using `run_transaction()` helper

Example:
```python
# Good: Single statement
async with engine.begin() as conn:
    await conn.execute(text(insert_sql))

# For multi-statement: implement retry logic
from sqlalchemy_cockroachdb import run_transaction

async def insert_with_retry():
    await run_transaction(engine, lambda conn: conn.execute(...))
```

### Batch Size Tuning

CockroachDB has specific guidance for vector inserts.

**Recommendations:**
- Default batch size: 100-500 vectors
- Smaller batches for large dimensions (>1000)
- Configurable via `batch_size` parameter
- Monitor performance and adjust

Example:
```python
# Default batch size
await vectorstore.aadd_texts(texts)  # Uses self.batch_size (100)

# Custom batch size
await vectorstore.aadd_texts(texts, batch_size=250)
```

### Filter Translation

Metadata filters are translated to SQL WHERE clauses.

**Supported Operators:**
- Equality: `{"field": "value"}`
- Comparison: `{"field": {"$gt": 10}}`
- List: `{"field": {"$in": [1, 2, 3]}}`
- Logical: `{"$and": [...]}`, `{"$or": [...]}`

**Implementation:**
```python
def _build_filter_clause(self, filter: dict) -> str:
    # Recursive parsing of filter dict
    # Generates parameterized SQL WHERE clause
    # Handles nested $and/$or operators
```

## Testing Strategy

### Unit Tests

Test individual components in isolation.

**Location:** `tests/unit/`

**Characteristics:**
- No database required
- Fast execution (<1s total)
- Test logic, not I/O
- Mock external dependencies

**Examples:**
- Index SQL generation
- Filter clause building
- Hybrid search score fusion
- Configuration validation

### Integration Tests

Test with real CockroachDB database.

**Location:** `tests/integration/`

**Characteristics:**
- Requires CockroachDB instance
- Tests full workflows
- Validates SQL correctness
- Tests CRUD operations

**Examples:**
- Vector insertion and retrieval
- Index creation and usage
- Filter queries
- Transaction behavior

### Test Fixtures

Common fixtures defined in `conftest.py`:

```python
@pytest.fixture
def connection_string() -> str:
    """Database connection string."""

@pytest_asyncio.fixture
async def cockroachdb_engine(connection_string) -> CockroachDBEngine:
    """Initialized engine."""

@pytest_asyncio.fixture
async def vectorstore(cockroachdb_engine) -> AsyncCockroachDBVectorStore:
    """Ready-to-use vector store."""
```

## Performance Optimization

### Vector Insertion

**Optimization Strategies:**
- Use smaller batches (100-500)
- Insert in parallel when possible
- Pre-create indexes after bulk insert
- Monitor batch time and adjust

**Benchmarking:**
```python
import time

start = time.time()
await vectorstore.aadd_texts(texts, batch_size=100)
duration = time.time() - start
print(f"Inserted {len(texts)} vectors in {duration:.2f}s")
```

### Query Performance

**Optimization Strategies:**
- Create vector indexes for >10k vectors
- Tune beam_size for accuracy/speed tradeoff
- Use prefix columns for multi-tenant scenarios
- Filter before vector search when possible

**Beam Size Impact:**
```python
# Faster, less accurate
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=10)
)

# Slower, more accurate
results = await vectorstore.asimilarity_search(
    "query", k=10, query_options=CSPANNQueryOptions(beam_size=500)
)
```

### Index Configuration

**Partition Size Guidelines:**
- `min_partition_size`: 10-100 (smaller = more granular)
- `max_partition_size`: 500-2000 (larger = fewer partitions)
- Tune based on data distribution

```python
# For 100k vectors
index = CSPANNIndex(
    distance_strategy=DistanceStrategy.COSINE,
    min_partition_size=50,
    max_partition_size=1000,
)
```

## Debugging Tips

### Connection Issues

```python
# Test connection
async with engine.engine.connect() as conn:
    result = await conn.execute(text("SELECT version()"))
    print(result.scalar())
```

### Query Debugging

```python
# Enable SQL logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Vector Search Issues

```python
# Check if vectors are inserted
async with engine.engine.connect() as conn:
    result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
    print(f"Vector count: {result.scalar()}")

# Verify index exists
result = await conn.execute(text("""
    SELECT indexname FROM pg_indexes 
    WHERE tablename = :table
"""), {"table": table_name})
print(f"Indexes: {list(result)}")
```

## Release Process

1. **Update version** in `__init__.py` and `pyproject.toml`
2. **Update CHANGELOG.md** with changes
3. **Run full test suite**
   ```bash
   pytest tests -v
   ruff check langchain_cockroachdb tests
   mypy langchain_cockroachdb
   ```
4. **Build package**
   ```bash
   uv build
   ```
5. **Create git tag**
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```
6. **Publish to PyPI**
   ```bash
   uv publish
   ```

## Troubleshooting

### Common Issues

**Import errors after changes:**
```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

**Tests hang:**
- Check if CockroachDB is running
- Verify connection string
- Check for unclosed connections

**Type errors:**
```bash
# Update type stubs
uv pip install --upgrade types-all
```

## Resources

- [CockroachDB Docs - Vector Indexes](https://www.cockroachlabs.com/docs/stable/vector-indexes)
- [CockroachDB Docs - SERIALIZABLE Isolation](https://www.cockroachlabs.com/docs/stable/transactions)
- [LangChain VectorStore Interface](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)

## Getting Help

- Open an issue for bugs or questions
- Tag `@cockroachdb` maintainers for urgent issues
- Check existing issues and discussions first
