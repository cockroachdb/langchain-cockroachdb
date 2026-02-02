.PHONY: help install install-uv ensure-uv ensure-venv \
        test test-unit test-integration test-watch coverage \
        lint lint-fix format format-check type-check check \
        clean build start-db stop-db restart-db db-shell db-logs \
        examples example-quickstart example-indexes example-hybrid example-filters example-chat \
        dev ci publish-test publish docs version bump-patch bump-minor bump-major all

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# -------------------------
# Guardrails / Preflight
# -------------------------

ensure-uv: ## Verify uv is installed
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(YELLOW)❌ 'uv' is not installed or not on PATH.$(NC)"; \
		echo "$(YELLOW)👉 Run: make install-uv$(NC)"; \
		exit 1; \
	}

ensure-venv: ensure-uv ## Ensure .venv exists (creates it if missing)
	@if [ ! -d ".venv" ]; then \
		echo "$(BLUE)Creating virtual environment (.venv)...$(NC)"; \
		uv venv; \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	fi

# -------------------------
# Core commands
# -------------------------

help: ## Show this help message
	@echo "$(BLUE)langchain-cockroachdb - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ensure-venv ## Install package and dependencies into .venv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@uv pip install -e ".[dev]"
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-uv: ## Install uv package manager
	@echo "$(BLUE)Installing uv...$(NC)"
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "$(GREEN)✓ uv installed$(NC)"
	@echo "$(YELLOW)Note: You may need to restart your shell for 'uv' to be on PATH.$(NC)"

test: ensure-venv ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	@.venv/bin/pytest tests -v

test-unit: ensure-venv ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@.venv/bin/pytest tests/unit -v

test-integration: ensure-venv ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@.venv/bin/pytest tests/integration -v

test-watch: ensure-venv ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@.venv/bin/pytest-watch tests -v

coverage: ensure-venv ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@.venv/bin/pytest tests --cov=langchain_cockroachdb --cov-report=term --cov-report=html --cov-report=xml
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

lint: ensure-venv ## Run linter (ruff)
	@echo "$(BLUE)Running linter...$(NC)"
	@.venv/bin/ruff check langchain_cockroachdb tests examples

lint-fix: ensure-venv ## Run linter with auto-fix
	@echo "$(BLUE)Running linter with auto-fix...$(NC)"
	@.venv/bin/ruff check --fix langchain_cockroachdb tests examples

format: ensure-venv ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	@.venv/bin/ruff format langchain_cockroachdb tests examples

format-check: ensure-venv ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@.venv/bin/ruff format --check langchain_cockroachdb tests examples

type-check: ensure-venv ## Run type checker (mypy)
	@echo "$(BLUE)Running type checker...$(NC)"
	@.venv/bin/mypy langchain_cockroachdb --ignore-missing-imports --no-site-packages 2>&1 | grep -v "_pytest" || true

check: lint format-check type-check ## Run all code quality checks

clean: ## Clean build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

build: ensure-venv clean ## Build package distributions
	@echo "$(BLUE)Building package...$(NC)"
	@uv build
	@echo "$(GREEN)✓ Build complete: dist/$(NC)"

# -------------------------
# CockroachDB (docker-compose)
# -------------------------

start-db: ## Start CockroachDB using docker-compose
	@echo "$(BLUE)Starting CockroachDB...$(NC)"
	docker-compose up -d
	@echo "$(YELLOW)Waiting for CockroachDB to be ready...$(NC)"
	@sleep 5
	@docker exec langchain-cockroachdb ./cockroach sql --insecure -e "SELECT 1" > /dev/null 2>&1 && \
		echo "$(GREEN)✓ CockroachDB is ready$(NC)" || \
		echo "$(YELLOW)⚠ CockroachDB may still be starting...$(NC)"

stop-db: ## Stop CockroachDB
	@echo "$(BLUE)Stopping CockroachDB...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ CockroachDB stopped$(NC)"

restart-db: stop-db start-db ## Restart CockroachDB

db-shell: ## Open CockroachDB SQL shell
	@echo "$(BLUE)Opening CockroachDB SQL shell...$(NC)"
	docker exec -it langchain-cockroachdb ./cockroach sql --insecure

db-logs: ## Show CockroachDB logs
	docker-compose logs -f cockroachdb

# -------------------------
# Examples
# -------------------------

examples: ensure-venv ## Run all examples
	@echo "$(BLUE)Running examples...$(NC)"
	@for file in examples/*.py; do \
		echo "$(YELLOW)Running $$file...$(NC)"; \
		.venv/bin/python $$file || exit 1; \
		echo "$(GREEN)✓ $$file completed$(NC)"; \
		echo ""; \
	done
	@echo "$(GREEN)✓ All examples completed successfully$(NC)"

example-quickstart: ensure-venv ## Run quickstart example
	@echo "$(BLUE)Running quickstart example...$(NC)"
	@.venv/bin/python examples/quickstart.py

example-indexes: ensure-venv ## Run vector indexes example
	@echo "$(BLUE)Running vector indexes example...$(NC)"
	@.venv/bin/python examples/vector_indexes.py

example-hybrid: ensure-venv ## Run hybrid search example
	@echo "$(BLUE)Running hybrid search example...$(NC)"
	@.venv/bin/python examples/hybrid_search.py

example-filters: ensure-venv ## Run metadata filtering example
	@echo "$(BLUE)Running metadata filtering example...$(NC)"
	@.venv/bin/python examples/metadata_filtering.py

example-chat: ensure-venv ## Run chat history example
	@echo "$(BLUE)Running chat history example...$(NC)"
	@.venv/bin/python examples/chat_history.py

# -------------------------
# Workflows
# -------------------------

dev: start-db install ## Setup development environment
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo "$(YELLOW)Run 'make test' to verify setup$(NC)"

ci: check test ## Run CI checks (lint, format, type-check, test)
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

publish-test: build ## Publish to test PyPI
	@echo "$(BLUE)Publishing to test PyPI...$(NC)"
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	@echo "$(YELLOW)⚠ Publishing to production PyPI$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload dist/*; \
		echo "$(GREEN)✓ Published to PyPI$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

docs: ensure-venv ## Build documentation site with mkdocs
	@echo "$(BLUE)Building documentation...$(NC)"
	@if [ ! -f .venv/bin/mkdocs ]; then \
		echo "$(YELLOW)Installing docs dependencies...$(NC)"; \
		.venv/bin/pip install -e ".[docs]" > /dev/null; \
	fi
	@.venv/bin/mkdocs build --strict
	@echo "$(GREEN)✓ Documentation built: site/$(NC)"

docs-serve: ensure-venv ## Serve documentation locally (with live reload)
	@echo "$(BLUE)Starting documentation server...$(NC)"
	@if [ ! -f .venv/bin/mkdocs ]; then \
		echo "$(YELLOW)Installing docs dependencies...$(NC)"; \
		.venv/bin/pip install -e ".[docs]"; \
	fi
	@echo "$(YELLOW)Opening http://127.0.0.1:8000$(NC)"
	@.venv/bin/mkdocs serve

version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	@grep '^version' pyproject.toml | head -1 | cut -d'"' -f2

bump-patch: ## Bump patch version (0.0.X)
	@echo "$(BLUE)Bumping patch version...$(NC)"
	@python -c "import re; \
		content = open('pyproject.toml').read(); \
		match = re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"', content); \
		if match: \
			major, minor, patch = map(int, match.groups()); \
			new_version = f'{major}.{minor}.{patch+1}'; \
			content = re.sub(r'version = \"\d+\.\d+\.\d+\"', f'version = \"{new_version}\"', content); \
			open('pyproject.toml', 'w').write(content); \
			print(f'Version bumped to {new_version}');"

bump-minor: ## Bump minor version (0.X.0)
	@echo "$(BLUE)Bumping minor version...$(NC)"
	@python -c "import re; \
		content = open('pyproject.toml').read(); \
		match = re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"', content); \
		if match: \
			major, minor, patch = map(int, match.groups()); \
			new_version = f'{major}.{minor+1}.0'; \
			content = re.sub(r'version = \"\d+\.\d+\.\d+\"', f'version = \"{new_version}\"', content); \
			open('pyproject.toml', 'w').write(content); \
			print(f'Version bumped to {new_version}');"

bump-major: ## Bump major version (X.0.0)
	@echo "$(BLUE)Bumping major version...$(NC)"
	@python -c "import re; \
		content = open('pyproject.toml').read(); \
		match = re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"', content); \
		if match: \
			major, minor, patch = map(int, match.groups()); \
			new_version = f'{major+1}.0.0'; \
			content = re.sub(r'version = \"\d+\.\d+\.\d+\"', f'version = \"{new_version}\"', content); \
			open('pyproject.toml', 'w').write(content); \
			print(f'Version bumped to {new_version}');"

all: clean install check test build ## Run full build pipeline
	@echo "$(GREEN)✓ Full build pipeline completed successfully$(NC)"

