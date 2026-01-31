# Makefile Commands Reference

This project includes a comprehensive Makefile to streamline common development tasks.

## Quick Start

```bash
# See all available commands
make help

# Setup development environment (starts DB + installs deps)
make dev

# Run tests
make test

# Run CI checks (lint, format, type-check, test)
make ci
```

## Installation

```bash
make install-uv          # Install uv package manager
make install             # Install package and dependencies
```

## Testing

```bash
make test                # Run all tests
make test-unit           # Run unit tests only
make test-integration    # Run integration tests only
make test-watch          # Run tests in watch mode
make coverage            # Run tests with coverage report
```

## Code Quality

```bash
make lint                # Run linter (ruff)
make lint-fix            # Run linter with auto-fix
make format              # Format code with ruff
make format-check        # Check formatting without changes
make type-check          # Run type checker (mypy)
make check               # Run all quality checks (lint + format + type)
```

## Database Management

```bash
make start-db            # Start CockroachDB using docker-compose
make stop-db             # Stop CockroachDB
make restart-db          # Restart CockroachDB
make db-shell            # Open CockroachDB SQL shell
make db-logs             # Show CockroachDB logs
```

## Examples

```bash
make examples            # Run all examples sequentially
make example-quickstart  # Run quickstart example
make example-indexes     # Run vector indexes example
make example-hybrid      # Run hybrid search example
make example-filters     # Run metadata filtering example
make example-chat        # Run chat history example
```

## Building & Publishing

```bash
make build               # Build package distributions
make clean               # Clean build artifacts and cache files
make version             # Show current version
make bump-patch          # Bump patch version (0.0.X)
make bump-minor          # Bump minor version (0.X.0)
make bump-major          # Bump major version (X.0.0)
make publish-test        # Publish to test PyPI
make publish             # Publish to production PyPI
```

## Development Workflows

### New Developer Setup

```bash
# 1. Install uv (if needed)
make install-uv

# 2. Setup environment
make dev

# 3. Verify setup
make test
```

### Daily Development

```bash
# Start database
make start-db

# Run tests while developing
make test-watch

# Before committing
make ci
```

### Pre-Release Checklist

```bash
# 1. Run full CI
make ci

# 2. Test examples
make examples

# 3. Update version
make bump-minor  # or bump-patch, bump-major

# 4. Build
make build

# 5. Test on test PyPI
make publish-test

# 6. Publish to production
make publish
```

## CI/CD Integration

The Makefile commands are used by GitHub Actions workflows:

```yaml
# Example: .github/workflows/_test.yml
- run: make install
- run: make test
```

## Composite Commands

Some commands are compositions of other commands:

- `make dev` = `make start-db` + `make install`
- `make check` = `make lint` + `make format-check` + `make type-check`
- `make ci` = `make check` + `make test`
- `make all` = `make clean` + `make install` + `make check` + `make test` + `make build`

## Troubleshooting

### Database Connection Issues

```bash
# Check if database is running
make db-shell

# Restart database
make restart-db

# Check logs
make db-logs
```

### Test Failures

```bash
# Run with verbose output
pytest tests -v

# Run specific test
pytest tests/integration/test_vectorstore.py -v

# Run with debugger
pytest tests -v --pdb
```

### Build Issues

```bash
# Clean and rebuild
make clean
make build
```

## Adding Custom Commands

To add a new command, edit the Makefile:

```makefile
my-command: ## Description of my command
	@echo "Running my command..."
	# Your commands here
```

Then run:

```bash
make my-command
```

## Tips

1. **Use tab completion**: Most shells support tab completion for Makefile targets
2. **Chain commands**: `make clean && make build && make publish-test`
3. **Parallel tests**: Use `pytest -n auto` for parallel test execution
4. **Watch mode**: Use `make test-watch` for automatic re-running during development

## Color Output

The Makefile uses colored output for better readability:
- 🔵 Blue: Info messages
- 🟢 Green: Success messages
- 🟡 Yellow: Warning messages

## Dependencies

The Makefile assumes these tools are available:
- `uv` - Package manager (installed via `make install-uv`)
- `docker` and `docker-compose` - For CockroachDB
- `python3` - Python interpreter

## Environment Variables

Some commands respect environment variables:

```bash
# Skip testcontainers and use existing database
USE_TESTCONTAINER=false make test-integration

# Use custom connection string
export COCKROACHDB_URL="cockroachdb://user@host:port/db"
make examples
```

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development documentation
- [README.md](README.md) - Project overview
