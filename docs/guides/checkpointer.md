# LangGraph Checkpointer Guide

Persist LangGraph workflow state in CockroachDB for short-term memory, human-in-the-loop patterns, and fault-tolerant agent execution.

## Overview

The CockroachDB checkpointer provides:

- **Short-term memory** for multi-turn conversations in LangGraph agents
- **Human-in-the-loop** support with interrupt/resume capabilities
- **Fault tolerance** with persistent state across process restarts
- **Distributed durability** backed by CockroachDB's transaction isolation (SERIALIZABLE default, READ COMMITTED also supported)

The implementation mirrors [langgraph-checkpoint-postgres](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver) with adaptations for CockroachDB compatibility.

## Installation

```bash
pip install langchain-cockroachdb
```

The checkpointer requires `langgraph-checkpoint>=2.1.2` and `psycopg[binary]>=3.0.0`, which are included as dependencies.

## Quick Start

### Synchronous Usage

```python
from langchain_cockroachdb import CockroachDBSaver

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

with CockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    # Compile your LangGraph workflow with the checkpointer
    graph = workflow.compile(checkpointer=checkpointer)

    # Invoke with a thread_id for persistent state
    config = {"configurable": {"thread_id": "user-123"}}
    result = graph.invoke({"messages": [("user", "Hello!")]}, config)

    # Continue the conversation -- state is persisted
    result = graph.invoke({"messages": [("user", "What did I just say?")]}, config)
```

### Asynchronous Usage

```python
from langchain_cockroachdb import AsyncCockroachDBSaver

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

async with AsyncCockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()

    graph = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-456"}}
    result = await graph.ainvoke({"messages": [("user", "Hello!")]}, config)
```

## Connection Modes

### Direct Connection

Best for scripts or single-process applications:

```python
from psycopg import Connection
from psycopg.rows import dict_row
from langchain_cockroachdb import CockroachDBSaver

conn = Connection.connect(
    "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
    autocommit=True,
    prepare_threshold=5,
    row_factory=dict_row,
)
saver = CockroachDBSaver(conn)
saver.setup()
```

!!! note
    Use `postgresql://` (not `cockroachdb://`) when creating raw psycopg connections directly. The `from_conn_string()` factory handles this conversion automatically.

### Connection Pool

Best for web applications and high-concurrency workloads:

```python
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langchain_cockroachdb import CockroachDBSaver

pool = ConnectionPool(
    "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
    max_size=20,
    kwargs={"autocommit": True, "row_factory": dict_row},
)
saver = CockroachDBSaver(pool)
saver.setup()
```

### Async Connection Pool

```python
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langchain_cockroachdb import AsyncCockroachDBSaver

pool = AsyncConnectionPool(
    "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
    max_size=20,
    kwargs={"autocommit": True, "row_factory": dict_row},
)
saver = AsyncCockroachDBSaver(pool)
await saver.setup()
```

### Factory Method (Recommended)

The simplest approach -- handles connection string conversion, pool setup, and enables prepared statement caching (`prepare_threshold=5`) for query plan reuse:

```python
# Sync
with CockroachDBSaver.from_conn_string("cockroachdb://root@localhost:26257/defaultdb") as saver:
    saver.setup()

# Async
async with AsyncCockroachDBSaver.from_conn_string("cockroachdb://root@localhost:26257/defaultdb") as saver:
    await saver.setup()
```

## Core Operations

### Saving Checkpoints

Checkpoints are saved automatically by LangGraph during graph execution. You can also save manually:

```python
from langgraph.checkpoint.base import empty_checkpoint

config = {"configurable": {"thread_id": "my-thread", "checkpoint_ns": ""}}
checkpoint = empty_checkpoint()
metadata = {"source": "input", "step": 1}

saved_config = saver.put(config, checkpoint, metadata, {})
```

### Retrieving Checkpoints

```python
# Get the latest checkpoint for a thread
config = {"configurable": {"thread_id": "my-thread", "checkpoint_ns": ""}}
result = saver.get_tuple(config)

if result:
    print(result.checkpoint)   # The checkpoint data
    print(result.metadata)     # Associated metadata
    print(result.config)       # Config with actual checkpoint_id
```

### Listing Checkpoints

```python
# List all checkpoints (across all threads)
for checkpoint_tuple in saver.list(None):
    print(checkpoint_tuple.config)

# Filter by thread
thread_config = {"configurable": {"thread_id": "my-thread"}}
for checkpoint_tuple in saver.list(thread_config):
    print(checkpoint_tuple.metadata)

# Filter by metadata
for checkpoint_tuple in saver.list(None, filter={"source": "loop"}):
    print(checkpoint_tuple.config)

# Limit results
for checkpoint_tuple in saver.list(None, limit=5):
    print(checkpoint_tuple.config)
```

### Deleting Threads

```python
saver.delete_thread("my-thread")
```

### Storing Intermediate Writes

```python
saver.put_writes(config, [("channel_name", value)], task_id="task-1")
```

## LangGraph Integration

### Chatbot with Memory

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_cockroachdb import CockroachDBSaver

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

llm = ChatOpenAI(model="gpt-4o-mini")


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

with CockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

    # First turn
    config = {"configurable": {"thread_id": "session-1"}}
    result = app.invoke(
        {"messages": [("user", "My name is Alice")]},
        config,
    )
    print(result["messages"][-1].content)

    # Second turn -- the bot remembers the name
    result = app.invoke(
        {"messages": [("user", "What is my name?")]},
        config,
    )
    print(result["messages"][-1].content)
```

### Human-in-the-Loop

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.base import empty_checkpoint
from langchain_cockroachdb import CockroachDBSaver

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"


def step_1(state: MessagesState):
    return {"messages": [("assistant", "Ready for approval")]}


def step_2(state: MessagesState):
    return {"messages": [("assistant", "Approved and executed")]}


graph = StateGraph(MessagesState)
graph.add_node("step_1", step_1)
graph.add_node("step_2", step_2)
graph.add_edge(START, "step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", END)

with CockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["step_2"],
    )

    config = {"configurable": {"thread_id": "approval-1"}}

    # Runs step_1 then pauses before step_2
    result = app.invoke(
        {"messages": [("user", "Start workflow")]},
        config,
    )
    print("Paused for approval:", result["messages"][-1])

    # Resume after human approval
    result = app.invoke(None, config)
    print("Completed:", result["messages"][-1])
```

## Schema and Migrations

The `setup()` method creates three tables and associated indexes:

| Table | Purpose |
|-------|---------|
| `checkpoints` | Stores checkpoint data and metadata per thread/namespace |
| `checkpoint_blobs` | Stores serialized channel values (binary blobs) |
| `checkpoint_writes` | Stores intermediate writes for pending sends |

Migrations are applied incrementally and idempotently -- calling `setup()` multiple times is safe.

## Row-Level TTL

CockroachDB supports automatic deletion of expired rows via [Row-Level TTL](https://www.cockroachlabs.com/docs/stable/row-level-ttl). The checkpointer exposes this as an opt-in feature for automatic cleanup of old checkpoint data.

### Enabling TTL

```python
# Sync
with CockroachDBSaver.from_conn_string(DB_URI) as saver:
    saver.setup()

    # Expire checkpoints older than 30 days, clean up daily
    saver.enable_ttl(ttl_interval="30 days", cron="@daily")

# Async
async with AsyncCockroachDBSaver.from_conn_string(DB_URI) as saver:
    await saver.setup()
    await saver.aenable_ttl(ttl_interval="7 days", cron="@hourly")
```

### Disabling TTL

```python
saver.disable_ttl()       # sync
await saver.adisable_ttl()  # async
```

### How It Works

- `setup()` adds a `created_at TIMESTAMPTZ` column (with `DEFAULT now()`) to all three checkpoint tables via migration.
- `enable_ttl()` uses CockroachDB's `ttl_expiration_expression` (recommended over `ttl_expire_after` to avoid full table rewrites).
- CockroachDB runs a background job on the specified cron schedule to delete rows where `created_at + interval` is in the past.
- `enable_ttl()` is idempotent -- calling it again updates the interval and cron.
- `disable_ttl()` removes TTL from all three tables.

!!! note
    TTL deletion is **eventual** -- expired rows remain queryable until the background job runs. CockroachDB rate-limits deletions to minimize impact on foreground queries.

## Performance

The CockroachDB checkpointer includes several performance optimizations over the base PostgresSaver approach:

### Separate Lightweight Queries

Instead of correlated subqueries with `jsonb_agg`/`jsonb_build_object`, the checkpointer issues separate simple queries for checkpoints, blobs, and writes. Each query uses primary key lookups.

### Batch Fetching in list()

When listing multiple checkpoints, blobs and writes for **all** returned checkpoints are fetched in just 2 batch queries (instead of 2 per checkpoint). This reduces database round trips from 2N+1 to 3 total, regardless of how many checkpoints are returned.

### Raw BYTEA Deserialization

Blob data is read as raw BYTEA via psycopg3's binary cursor protocol, eliminating the overhead of hex-encoding in SQL and decoding in Python.

### Prepared Statement Caching

The `from_conn_string()` factory sets `prepare_threshold=5`, enabling server-side prepared statement caching after 5 executions of the same query. This eliminates repeated query parsing and planning overhead for hot paths.

## CockroachDB-Specific Adaptations

This checkpointer is modeled on `langgraph-checkpoint-postgres` (PostgresSaver) with the following adaptations for CockroachDB compatibility:

### Separate Queries Instead of Multidimensional Arrays

CockroachDB does not support multidimensional arrays. The PostgresSaver uses correlated subqueries with `array_agg(array[...])`. The CockroachDB checkpointer instead fetches blobs and writes via separate lightweight primary key lookups, with batch variants for `list()`.

### Index Creation Without CONCURRENTLY

CockroachDB creates indexes in a non-blocking manner by default, so the `CONCURRENTLY` keyword used by PostgresSaver is omitted.

### Connection String Sanitization

The `from_conn_string()` factory automatically converts `cockroachdb://` URLs to `postgresql://` for psycopg compatibility.

### Explicit Table Function Aliases

CockroachDB requires explicit aliases for set-returning functions like `jsonb_each_text()`:

```sql
-- Required for CockroachDB:
FROM jsonb_each_text(checkpoint -> 'channel_versions') AS jt
-- Not: FROM jsonb_each_text(checkpoint -> 'channel_versions')
```

## Comparison with PostgresSaver

| Feature | PostgresSaver | CockroachDBSaver |
|---------|---------------|------------------|
| Connection library | psycopg3 | psycopg3 |
| Connection modes | Connection, Pool, Pipeline | Connection, Pool, Pipeline |
| Isolation level | READ COMMITTED | SERIALIZABLE (default), READ COMMITTED supported |
| Blob/write fetching | Correlated subqueries | Separate PK lookups + batch fetching |
| Index creation | `CREATE INDEX CONCURRENTLY` | `CREATE INDEX` (non-blocking by default) |
| Blob encoding | Native bytea arrays | Raw BYTEA via binary cursor |
| Prepared statements | `prepare_threshold=0` (disabled) | `prepare_threshold=5` (enabled) |
| Row-level TTL | Not supported | `enable_ttl()` / `disable_ttl()` |
| Async support | Yes | Yes |
| Thread deletion | Yes | Yes |
| Pipeline support | Yes | Yes |

## Next Steps

- [Checkpointer Example](../examples/checkpointer.md) - Working code examples
- [API Reference](../api/checkpointer.md) - Detailed API documentation
- [Chat History Guide](chat-history.md) - For long-term conversation storage
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - LangGraph framework docs
