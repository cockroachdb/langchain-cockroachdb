# Configuration

Learn how to configure langchain-cockroachdb for different workloads.

## Engine Configuration

### Basic Configuration

```python
from langchain_cockroachdb import CockroachDBEngine

engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://user:pass@host:26257/db",
    pool_size=10,                    # Base connections
    max_overflow=20,                 # Additional connections
    pool_pre_ping=True,              # Health check before use
    pool_recycle=3600,               # Recycle after 1 hour
    pool_timeout=30.0,               # Wait 30s for connection
    connect_timeout=10,              # Connection timeout (default: 10)
)
```

### Retry Configuration

Set a `RetryPolicy` on the engine to control which errors are retried and how:

```python
from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBChatMessageHistory,
    CockroachDBEngine,
    RetryPolicy,
)

engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
    retry_policy=RetryPolicy(max_attempts=8, max_backoff=5.0),
)
store = AsyncCockroachDBVectorStore(engine, embeddings, "documents")  # inherits
history = CockroachDBChatMessageHistory("session-1", engine=engine)  # inherits
fast_store = AsyncCockroachDBVectorStore(
    engine, embeddings, "documents", retry_policy=RetryPolicy(max_attempts=2)
)
```

See [Retries and Resilience](../guides/resilience.md) for what is retried and every option.

## Vector Store Configuration

### Basic Settings

```python
from langchain_cockroachdb import AsyncCockroachDBVectorStore

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    
    # Column names
    content_column="content",
    embedding_column="embedding",
    metadata_column="metadata",
    id_column="id",
    
    # Distance metric
    distance_strategy=DistanceStrategy.COSINE,  # or L2, IP
    
    # Batch size for inserts
    batch_size=100,
)
```

### Retry Settings

The vector store inherits the engine's retry policy. Each insert batch is retried on its own. Override the policy with `retry_policy` or shorthand `retry_*` arguments:

```python
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
    retry_policy=RetryPolicy(max_attempts=3, max_backoff=5.0),
)
```

## Configuration Presets

### Development

```python
# Optimized for local development
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=5,
    max_overflow=10,
    retry_policy=RetryPolicy(max_attempts=3, max_backoff=1.0),
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=100,
)
```

### Web Applications

```python
# High concurrency, aggressive retries
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=20,
    max_overflow=40,
    retry_policy=RetryPolicy(max_attempts=10, max_backoff=30.0),
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=100,
    retry_policy=RetryPolicy(max_attempts=5),
)
```

### Batch Jobs

```python
# Resilient, patient retries
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=3,
    max_overflow=5,
    retry_policy=RetryPolicy(max_attempts=20, max_backoff=60.0),
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=500,  # Larger batches for throughput
    retry_policy=RetryPolicy(max_attempts=10),
)
```

### Multi-Region

```python
from langchain_cockroachdb import RETRYABLE_SQLSTATES, RetryPolicy

# Tolerates failovers and high latency
failover = RetryPolicy(
    max_attempts=8,
    max_backoff=5.0,
    retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"},
    attempt_timeout=10,
)

engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=15,
    max_overflow=30,
    pool_timeout=60.0,  # Longer timeout
    retry_policy=failover,
)

vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="docs",
    batch_size=200,  # Moderate batches
)
```

## Batch Size Guidelines

Choose batch size based on embedding dimensions and workload:

| Embedding Size | Recommended Batch Size |
|----------------|------------------------|
| < 512 dims     | 200-500               |
| 512-1536 dims  | 100-200               |
| > 1536 dims    | 50-100                |

CockroachDB works best with smaller batches compared to single-node databases.

## Connection Pool Sizing

General guidelines:

| Workload Type | pool_size | max_overflow |
|---------------|-----------|--------------|
| Development   | 5         | 10           |
| Web (low traffic) | 10    | 20           |
| Web (high traffic) | 20-50 | 40-100      |
| Batch jobs    | 3-5       | 5-10         |
| Analytics     | 10-20     | 20-40        |

**Formula:** `max_connections = pool_size + max_overflow`

## Monitoring Configuration

### Enable SQL Logging

```python
import logging

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Enable Pool Logging

```python
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    echo=True,        # Log all SQL
    echo_pool=True,   # Log pool events
)
```

## Environment Variables

Store sensitive configuration in environment variables:

```bash
export COCKROACHDB_URL="cockroachdb://user:pass@host:26257/db?sslmode=verify-full"
export COCKROACHDB_POOL_SIZE="20"
export COCKROACHDB_MAX_OVERFLOW="40"
```

```python
import os

engine = CockroachDBEngine.from_connection_string(
    os.getenv("COCKROACHDB_URL"),
    pool_size=int(os.getenv("COCKROACHDB_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("COCKROACHDB_MAX_OVERFLOW", "20")),
)
```

## SSL Configuration

### CockroachDB Cloud

```python
connection_string = (
    "cockroachdb://user:pass@cluster.cloud:26257/db"
    "?sslmode=verify-full"
    "&sslrootcert=/path/to/root.crt"
)
```

### Self-Hosted with Custom CA

```python
connection_string = (
    "cockroachdb://user@host:26257/db"
    "?sslmode=verify-full"
    "&sslcert=/path/to/client.crt"
    "&sslkey=/path/to/client.key"
    "&sslrootcert=/path/to/ca.crt"
)
```

### Insecure (Development Only)

```python
connection_string = "cockroachdb://root@localhost:26257/db?sslmode=disable"
```

## Next Steps

- [Vector Indexes](../guides/vector-indexes.md) - Optimize queries
- [Async vs Sync](../guides/async-vs-sync.md) - Choose the right API
- [Examples](../examples/index.md) - See configurations in action
