# Testing Guide

Learn how to run and write tests for langchain-cockroachdb.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Fast tests, no database
│   ├── test_indexes.py
│   ├── test_hybrid_search.py
│   └── test_retry.py
└── integration/             # Tests with CockroachDB
    ├── test_engine.py
    ├── test_vectorstore.py
    ├── test_chat_history.py
    ├── test_retry_behavior.py
    ├── test_sync_wrapper.py
    └── test_configuration.py
```

## Running Tests

### All Tests

```bash
make test
```

### Unit Tests Only

```bash
pytest tests/unit -v
```

### Integration Tests Only

```bash
pytest tests/integration -v
```

### Specific Test File

```bash
pytest tests/unit/test_indexes.py -v
```

### Specific Test

```bash
pytest tests/unit/test_indexes.py::TestCSPANNIndex::test_index_creation -v
```

### With Coverage

```bash
pytest tests --cov=langchain_cockroachdb --cov-report=html
open htmlcov/index.html
```

## Requirements

### Unit Tests
- No external dependencies
- Run anywhere
- Very fast (<1s)

### Integration Tests
- Docker running
- Testcontainers library
- Takes longer (~1min)

## Writing Tests

### Unit Test Example

```python
import pytest
from langchain_cockroachdb import CSPANNIndex, DistanceStrategy

class TestCSPANNIndex:
    def test_index_creation(self):
        """Test creating C-SPANN index configuration."""
        index = CSPANNIndex(
            name="test_index",
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        assert index.name == "test_index"
        assert index.distance_strategy == DistanceStrategy.COSINE
    
    def test_index_sql_generation(self):
        """Test SQL generation for index creation."""
        index = CSPANNIndex()
        sql = index.get_create_sql("table", "embedding")
        
        assert "CREATE VECTOR INDEX" in sql
        assert "table" in sql
        assert "embedding" in sql
```

### Integration Test Example

```python
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

class TestVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self, connection_string: str):
        """Test adding documents and searching."""
        # Setup
        engine = CockroachDBEngine.from_connection_string(connection_string)
        await engine.ainit_vectorstore_table(
            table_name="test_docs",
            vector_dimension=384,
            drop_if_exists=True,
        )
        
        embeddings = DeterministicFakeEmbedding(size=384)
        vectorstore = AsyncCockroachDBVectorStore(
            engine=engine,
            embeddings=embeddings,
            collection_name="test_docs",
        )
        
        # Test
        texts = ["doc1", "doc2", "doc3"]
        ids = await vectorstore.aadd_texts(texts)
        
        assert len(ids) == 3
        
        results = await vectorstore.asimilarity_search("doc1", k=2)
        
        assert len(results) == 2
        assert results[0].page_content in texts
        
        # Cleanup
        await engine.aclose()
```

## Fixtures

### Common Fixtures

Available in `conftest.py`:

```python
@pytest.fixture
def connection_string() -> str:
    """Get CockroachDB connection string."""
    return os.getenv(
        "COCKROACHDB_URL",
        "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
    )

@pytest.fixture
async def engine(connection_string: str):
    """Get CockroachDB engine."""
    engine = CockroachDBEngine.from_connection_string(connection_string)
    yield engine
    await engine.aclose()

@pytest.fixture
async def vectorstore(engine: CockroachDBEngine):
    """Get vector store with test table."""
    await engine.ainit_vectorstore_table(
        table_name="test_vectors",
        vector_dimension=384,
        drop_if_exists=True,
    )
    
    embeddings = DeterministicFakeEmbedding(size=384)
    vs = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="test_vectors",
    )
    
    yield vs
```

## Test Guidelines

### 1. Use Descriptive Names

```python
# Good
async def test_add_texts_with_metadata_returns_correct_ids():
    ...

# Bad
async def test_add():
    ...
```

### 2. Test One Thing

```python
# Good: Tests one behavior
async def test_add_texts_returns_ids():
    ids = await vectorstore.aadd_texts(["doc1"])
    assert len(ids) == 1

# Bad: Tests multiple behaviors
async def test_vectorstore():
    ids = await vectorstore.aadd_texts(["doc1"])
    results = await vectorstore.asimilarity_search("doc1")
    await vectorstore.adelete(ids)
    # Too much in one test
```

### 3. Use Fixtures

```python
# Good: Uses fixture
async def test_search(vectorstore):
    await vectorstore.aadd_texts(["doc1"])
    results = await vectorstore.asimilarity_search("doc1")
    assert len(results) > 0

# Bad: Setup in test
async def test_search():
    engine = CockroachDBEngine.from_connection_string(...)
    await engine.ainit_vectorstore_table(...)
    vectorstore = AsyncCockroachDBVectorStore(...)
    # Repeated setup
```

### 4. Clean Up Resources

```python
@pytest.fixture
async def resource():
    # Setup
    r = create_resource()
    
    yield r
    
    # Cleanup (always runs)
    await r.close()
```

## Test Coverage

### Current Coverage

```bash
pytest tests --cov=langchain_cockroachdb --cov-report=term

# Shows:
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
langchain_cockroachdb/__init__.py          10      0   100%
langchain_cockroachdb/engine.py            89      5    94%
langchain_cockroachdb/vectorstore.py      245     12    95%
...
-----------------------------------------------------------
TOTAL                                    1234     45    96%
```

### Coverage Goals

- Overall: >90%
- Core modules: >95%
- New features: 100%

## Continuous Integration

Tests run automatically on:
- Every push
- Every pull request
- Python 3.10, 3.11, 3.12

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Run tests
        run: make test
```

## Troubleshooting

### Docker Not Running

```bash
# Start Docker Desktop
# Then:
docker ps  # Verify Docker works
```

### Connection Refused

```bash
# Start CockroachDB
docker-compose up -d

# Wait for it to be ready
docker logs cockroachdb
```

### Slow Tests

```bash
# Run only fast unit tests
pytest tests/unit -v

# Or run specific test
pytest tests/integration/test_vectorstore.py::TestVectorStore::test_add_texts -v
```

## Next Steps

- [Contributing](contributing.md) - Submit changes
- [Architecture](architecture.md) - Understand design
- [API Reference](../api/vectorstore.md) - API details
