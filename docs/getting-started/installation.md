# Installation

## Requirements

- Python 3.10 or higher
- CockroachDB 24.2 or higher (for native vector support)

## Install from PyPI

```bash
pip install langchain-cockroachdb
```

Or with uv (recommended):

```bash
uv pip install langchain-cockroachdb
```

## Install from Source

```bash
git clone https://github.com/cockroachdb/langchain-cockroachdb.git
cd langchain-cockroachdb
pip install -e .
```

## Development Installation

For contributing or development:

```bash
git clone https://github.com/cockroachdb/langchain-cockroachdb.git
cd langchain-cockroachdb
pip install -e ".[dev]"
```

This installs additional dependencies:
- pytest (testing)
- pytest-asyncio (async testing)
- ruff (linting/formatting)
- mypy (type checking)
- testcontainers (integration testing)

## CockroachDB Setup

### Option 1: CockroachDB Cloud (Recommended)

1. Sign up at [cockroachlabs.cloud](https://cockroachlabs.cloud)
2. Create a free cluster
3. Get your connection string
4. Download the SSL certificate

### Option 2: Docker (Development)

```bash
docker run -d \
  --name cockroachdb \
  -p 26257:26257 \
  -p 8080:8080 \
  cockroachdb/cockroach:latest \
  start-single-node --insecure
```

### Option 3: Local Binary

Download from [cockroachlabs.com/docs/releases](https://www.cockroachlabs.com/docs/releases/)

```bash
cockroach start-single-node --insecure --listen-addr=localhost:26257
```

## Verify Installation

```python
import asyncio
from langchain_cockroachdb import CockroachDBEngine

async def test():
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
    )
    async with engine.engine.connect() as conn:
        from sqlalchemy import text
        result = await conn.execute(text("SELECT version()"))
        print(f"✅ Connected: {result.scalar()}")
    await engine.aclose()

asyncio.run(test())
```

## Next Steps

- [Quick Start Guide](quick-start.md) - Build your first application
- [Configuration](configuration.md) - Learn about configuration options
