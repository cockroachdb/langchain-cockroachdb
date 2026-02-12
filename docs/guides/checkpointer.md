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
    prepare_threshold=0,
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

The simplest approach -- handles connection string conversion and pool setup:

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

## CockroachDB-Specific Adaptations

This checkpointer is modeled on `langgraph-checkpoint-postgres` (PostgresSaver) with the following adaptations for CockroachDB compatibility:

### JSONB Instead of Multidimensional Arrays

CockroachDB does not support multidimensional arrays. The PostgresSaver uses:

```sql
-- PostgresSaver (not compatible with CockroachDB):
SELECT array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob]) ...
```

The CockroachDB checkpointer uses JSONB aggregation instead:

```sql
-- CockroachDBSaver:
SELECT jsonb_agg(jsonb_build_object(
    'channel', bl.channel,
    'type', bl.type,
    'blob', encode(bl.blob, 'hex')
)) ...
```

Blob data is hex-encoded within the JSONB structure and decoded during deserialization.

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
| Array aggregation | `array_agg(array[...])` | `jsonb_agg(jsonb_build_object(...))` |
| Index creation | `CREATE INDEX CONCURRENTLY` | `CREATE INDEX` (non-blocking by default) |
| Blob encoding | Native bytea arrays | Hex-encoded in JSONB |
| Async support | Yes | Yes |
| Thread deletion | Yes | Yes |
| Pipeline support | Yes | Yes |

## Next Steps

- [Checkpointer Example](../examples/checkpointer.md) - Working code examples
- [API Reference](../api/checkpointer.md) - Detailed API documentation
- [Chat History Guide](chat-history.md) - For long-term conversation storage
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - LangGraph framework docs
