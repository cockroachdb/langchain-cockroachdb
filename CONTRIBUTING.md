# Contributing to langchain-cockroachdb

Thank you for your interest in contributing to langchain-cockroachdb! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an open and welcoming environment.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) for package management
- Docker for running CockroachDB locally
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/cockroachdb/langchain-cockroachdb.git
   cd langchain-cockroachdb
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Start CockroachDB locally**
   ```bash
   docker-compose up -d
   ```

4. **Verify setup**
   ```bash
   pytest tests/unit -v
   ```

## Development Workflow

### Running Tests

```bash
# Unit tests (fast, no database required)
pytest tests/unit -v

# Integration tests (requires CockroachDB)
docker-compose up -d
pytest tests/integration -v

# All tests
pytest tests -v

# With coverage
pytest tests --cov=langchain_cockroachdb --cov-report=html
```

### Code Quality

Before submitting a PR, ensure your code passes all checks:

```bash
# Linting
ruff check langchain_cockroachdb tests

# Auto-fix linting issues
ruff check langchain_cockroachdb tests --fix

# Type checking
mypy langchain_cockroachdb

# Format code (if using black)
black langchain_cockroachdb tests
```

### Running Specific Tests

```bash
# Run single test file
pytest tests/unit/test_indexes.py -v

# Run single test
pytest tests/unit/test_indexes.py::TestCSPANNIndex::test_create_index_sql_basic -v

# Run tests matching pattern
pytest tests -k "vector" -v
```

## Contribution Guidelines

### Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- CockroachDB version
- Python version
- Relevant code snippets or error messages

### Submitting Pull Requests

1. **Fork and create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise code
   - Add tests for new functionality
   - Update documentation as needed
   - Follow existing code style

3. **Test thoroughly**
   ```bash
   pytest tests -v
   ruff check langchain_cockroachdb tests
   mypy langchain_cockroachdb
   ```

4. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "feat: add support for X
   
   - Implement feature X
   - Add tests for X
   - Update documentation
   
   Closes #123"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: add support for metadata filtering in vector search

- Implement _build_filter_clause for complex filters
- Support $and, $or, $gt, $lt operators
- Add comprehensive tests
- Update documentation

Closes #42
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings for all public functions/classes

### Docstring Format

```python
async def asimilarity_search(
    self,
    query: str,
    k: int = 4,
    filter: Optional[dict] = None,
    **kwargs: Any,
) -> list[Document]:
    """Search for similar documents.
    
    Args:
        query: Query text
        k: Number of results to return
        filter: Optional metadata filter
        **kwargs: Additional arguments
        
    Returns:
        List of matching documents
        
    Raises:
        ValueError: If query is empty
        
    Example:
        ```python
        results = await store.asimilarity_search("database", k=5)
        ```
    """
```

### Testing Guidelines

- Write tests for all new features
- Maintain >80% code coverage
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Follow AAA pattern: Arrange, Act, Assert
- Use fixtures for common setup

Example:
```python
async def test_asimilarity_search_with_filter_returns_filtered_results(
    vectorstore: AsyncCockroachDBVectorStore,
    sample_texts: list[str],
    sample_metadatas: list[dict],
) -> None:
    """Test that similarity search respects metadata filters."""
    # Arrange
    await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)
    filter_dict = {"category": {"$eq": "database"}}
    
    # Act
    results = await vectorstore.asimilarity_search("query", k=5, filter=filter_dict)
    
    # Assert
    assert len(results) > 0
    for doc in results:
        assert doc.metadata.get("category") == "database"
```

## CockroachDB-Specific Considerations

### Transaction Retries

When implementing features that use transactions:

- Use smaller transactions when possible
- Implement retry logic for serializable isolation
- Consider using `run_transaction()` helper
- Test with induced contention

### Vector Operations

- Default to smaller batch sizes (100-500) for vector inserts
- Use C-SPANN indexes with appropriate partition sizes
- Test with various distance strategies (cosine, L2, inner product)
- Consider prefix columns for multi-tenant scenarios

### Performance Testing

When adding performance-critical features:

- Benchmark with realistic data volumes
- Test with different index configurations
- Document performance characteristics
- Compare with and without indexes

## Documentation

### What to Document

- All public APIs
- Configuration options
- Usage examples
- Performance considerations
- Migration guides (if applicable)

### Where to Add Documentation

- Docstrings in code
- README.md for quick start
- Examples in `examples/` directory
- Architecture decisions in code comments

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in GitHub Discussions
- Tag maintainers for urgent issues
- Check existing issues and PRs first

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
