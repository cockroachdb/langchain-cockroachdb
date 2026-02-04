# Contributing

Thank you for your interest in contributing to langchain-cockroachdb!

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/langchain-cockroachdb.git
cd langchain-cockroachdb
```

### 2. Install Dependencies

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or with uv (recommended)
uv pip install -e ".[dev]"
```

### 3. Start CockroachDB

```bash
docker-compose up -d
```

### 4. Run Tests

```bash
make test
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the [code style guidelines](#code-style) below.

### 3. Run Checks

```bash
# Run all checks
make all

# Or individually:
make lint      # Linting
make format    # Formatting
make typecheck # Type checking
make test      # Tests
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use async/await for I/O operations

```python
# Good
async def add_documents(
    self,
    texts: list[str],
    metadatas: list[dict] | None = None,
) -> list[str]:
    """Add documents to the vector store.
    
    Args:
        texts: List of document texts
        metadatas: Optional metadata for each document
        
    Returns:
        List of document IDs
    """
    # Implementation
```

### Linting

```bash
# Auto-fix most issues
make format

# Check linting
make lint
```

### Type Checking

```bash
make typecheck
```

## Testing

### Test Structure

```
tests/
├── unit/           # Fast tests, no database
└── integration/    # Tests with CockroachDB
```

### Writing Tests

```python
import pytest
from langchain_cockroachdb import AsyncCockroachDBVectorStore

class TestVectorStore:
    @pytest.mark.asyncio
    async def test_add_texts(self, vectorstore):
        """Test adding texts to vector store."""
        texts = ["doc1", "doc2"]
        ids = await vectorstore.aadd_texts(texts)
        
        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/unit -v

# Specific test
pytest tests/unit/test_indexes.py::TestCSPANNIndex::test_index_creation -v

# With coverage
pytest tests --cov=langchain_cockroachdb --cov-report=html
```

### Test Requirements

- ✅ Tests must pass
- ✅ Coverage should not decrease
- ✅ Use fixtures for common setup
- ✅ Mock external services when possible

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 10) -> bool:
    """Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description
        arg2: Description (default: 10)
        
    Returns:
        Description
        
    Raises:
        ValueError: Description
        
    Example:
        ```python
        result = function("value", arg2=20)
        ```
    """
```

### Documentation Site

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally
mkdocs serve

# Build
mkdocs build
```

### Adding Documentation

- Update relevant guide in `docs/guides/`
- Add examples to `examples/`
- Update README.md if needed

## Pull Request Guidelines

### PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow Conventional Commits

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How were the changes tested?

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
```

## Release Process

Maintainers only:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push to GitHub
5. GitHub Actions builds and publishes to PyPI

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/cockroachdb/langchain-cockroachdb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cockroachdb/langchain-cockroachdb/discussions)

## Code of Conduct

Be respectful and inclusive. See [CODE_OF_CONDUCT.md](https://github.com/cockroachdb/langchain-cockroachdb/blob/main/CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
