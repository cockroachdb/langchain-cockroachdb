# CockroachDBEngine API

Connection and transaction management.

::: langchain_cockroachdb.engine.CockroachDBEngine
    options:
      show_root_heading: true
      show_source: false
      members:
        - from_connection_string
        - from_engine
        - ainit_vectorstore_table
        - init_vectorstore_table
        - aclose
        - close

## Overview

The `CockroachDBEngine` manages database connections, connection pooling, and table initialization.

## Class Methods

### from_connection_string

Create engine from connection string.

```python
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://user:pass@host:26257/db",
    pool_size=10,
    max_overflow=20,
)
```

### from_engine

Create from existing SQLAlchemy engine.

```python
from sqlalchemy.ext.asyncio import create_async_engine

async_engine = create_async_engine("cockroachdb://...")
engine = CockroachDBEngine.from_engine(async_engine)
```

## Instance Methods

### ainit_vectorstore_table

Create vector store table (async).

```python
await engine.ainit_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
    drop_if_exists=False,
)
```

### init_vectorstore_table

Create vector store table (sync).

```python
engine.init_vectorstore_table(
    table_name="documents",
    vector_dimension=1536,
)
```

### aclose / close

Close engine and connections.

```python
await engine.aclose()  # Async
engine.close()          # Sync
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size` | int | 10 | Base connection pool size |
| `max_overflow` | int | 20 | Additional connections allowed |
| `pool_pre_ping` | bool | True | Health check before use |
| `pool_recycle` | int | 3600 | Recycle connections after (seconds) |
| `pool_timeout` | float | 30.0 | Wait timeout for connection |
| `retry_max_attempts` | int | 5 | Maximum retry attempts |
| `retry_initial_backoff` | float | 0.1 | Initial retry delay |
| `retry_max_backoff` | float | 10.0 | Maximum retry delay |
| `retry_backoff_multiplier` | float | 2.0 | Backoff multiplier |
| `retry_jitter` | bool | True | Add randomization to backoff |

## Examples

See [Configuration Guide](../getting-started/configuration.md) for detailed examples.
