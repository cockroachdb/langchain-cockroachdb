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

Best for web applications and high-concurrency workloads. A saver on a pool runs operations concurrently, each on its own pooled connection, so one saver can serve many agents, threads or tasks. A saver on a single connection runs operations one at a time, so long-running services should use a pool. A shared pooled saver runs as many operations at once as the pool has connections, so set `max_size` for the number of agents you expect to run at the same time. psycopg_pool's default `max_size` equals `min_size`, which defaults to 4. When every connection is busy, an operation waits up to the pool's `timeout` and then raises `PoolTimeout`, which is not retried. For example:

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

Pools you build yourself do not get the library's `gssencmode=disable` default. On macOS with prefork servers, add `gssencmode=disable` to the pool's conninfo or kwargs.

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

## Retries and Connection Pools

A single connection cannot recover once it drops, so the savers do not retry a dead single connection. Give them a pool, which replaces broken connections:

```python
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import CockroachDBSaver, RetryPolicy

with ConnectionPool(
    "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
    kwargs={"autocommit": True, "row_factory": dict_row},
    open=True,
) as pool:
    checkpointer = CockroachDBSaver(pool, retry_policy=RetryPolicy(max_attempts=8))
    checkpointer.setup()  # safe to run from several processes at once
```

A pipelined saver (`from_conn_string(..., pipeline=True)`) uses one connection, so it also cannot recover once that connection drops. Use a pool for long-running services.

`setup()` is safe to run from several processes at once and resumes where it stopped if it is interrupted.

`from_conn_string` accepts `retry_policy`, `connect_timeout`, `prepare_threshold` and `connect_kwargs` (for example `{"application_name": "my-service"}`).

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

### Atomic Writes

`put` with channel blobs, `put_writes` with several writes, and `delete_thread` send BEGIN, their statements and COMMIT as one batch, so each write is saved completely or not at all, even when the caller is cancelled or fails partway. A write still takes one round trip, and CockroachDB can still retry it on the server. Single-statement writes are unchanged.

On a single connection that the caller also uses, a write made inside a transaction the caller opened joins it as a savepoint, so the caller's commit or rollback decides. The saver never rolls back a transaction it did not open, and an error inside the caller's transaction is raised at once, without retries, so the caller's own retry loop can restart the whole transaction.

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

### Crash Recovery and Durability Modes

LangGraph decides when a checkpoint is written through the `durability`
argument of `invoke`, `stream` and their async versions. See LangGraph's
[durability modes](https://docs.langchain.com/oss/python/langgraph/checkpointers#durability-modes).

- `"sync"`: each checkpoint is written before the next step starts.
- `"async"`: each checkpoint is written in the background while the next step
  runs. This is LangGraph's default.
- `"exit"`: state is written only when the run ends.

With `"async"`, a process that dies shortly after a step can lose that step's
checkpoint, and with it the record of a tool call the model made. After a
restart the graph resumes from an older checkpoint, so the model can plan the
same call again under a new tool-call ID. An idempotency key built from the
tool-call ID does not catch that repeat, because the ID it was built from was
never saved.

When a crash between steps matters, for example when tools charge cards, send
messages or change other systems, pass `durability="sync"`:

```python
result = app.invoke(inputs, config, durability="sync")
```

`"sync"` adds a write to CockroachDB before each step. Keep `"async"` where
losing the last step is acceptable, such as a chat turn the user can repeat.

Even with `"sync"`, a node that was interrupted or did not finish runs again
from its first line when the graph resumes, so a tool that reached an external
service before the crash can run twice. Make such tools idempotent. Derive a
key from state that was saved before the tool ran, such as the thread ID and the
tool-call ID, and have the receiving service ignore a repeated key and return the
original result.

### Application-process Recovery

For application-crash recovery coverage, see the
[process-recovery tests](../development/testing.md#application-process-recovery).
They validate approval resume and pending-write reuse with a fresh process and
pool, using `durability="sync"` on both sync and async graph calls. On Python
3.10, approval resume works only with the sync API, because an async graph's
`interrupt()` needs Python 3.11 or newer. Interrupted
or unfinished nodes can execute code again, so external tool effects need
application-level idempotency or reconciliation.

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

    # Expire threads idle for 30 days, clean up daily
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

### What TTL Expires

Each row expires by its own `created_at`, which `setup()` adds to all three tables with `DEFAULT now()`.

- A checkpoint row, and the writes saved with it, expire `ttl_interval` after they were written. The older checkpoints of an active thread expire too, so its history gets shorter while its latest checkpoints stay.
- A blob row holds one channel's value at one version. It is written once, when the channel changes, and every later checkpoint in which the channel kept that version uses the same row. A blob row expires `ttl_interval` plus a tenth of it after its `created_at`.

A channel that has not changed since an early step, such as a profile set on the first turn, keeps its first blob row. So while TTL is on, each `put` brings `created_at` forward to now on the blob rows its checkpoint uses but does not write, once they are older than a tenth of the interval. It does this in the same transaction as the checkpoint row. The extra tenth is the grace: after a put, each blob row the new checkpoint uses expires no earlier than the checkpoint.

So a thread keeps its whole state while it is active, and expires once it has been idle for `ttl_interval`. A blob row that no checkpoint uses any more stays at most a tenth of the interval longer than the checkpoints did. While TTL is on, a put that uses blob rows it does not write sends one more `UPDATE` in its transaction, and a blob row is rewritten at most once per tenth of the interval.

The saver learns whether TTL is on, and the interval, from the settings of `checkpoint_blobs` in the catalog. It reads them on the first put that needs them and keeps them for `TTL_STATE_CACHE_SECONDS` (60 seconds), so it sees a change another process makes within a minute. Its own `enable_ttl()` and `disable_ttl()` take effect for it at once.

### How It Works

- `enable_ttl()` uses CockroachDB's `ttl_expiration_expression`, which avoids the full table rewrite that `ttl_expire_after` causes.
- CockroachDB runs a background job on the cron schedule. It deletes checkpoint and write rows whose `created_at + interval` has passed, and blob rows whose `created_at + interval + interval / 10` has.
- `enable_ttl()` is idempotent. Calling it again updates the interval and cron.
- `disable_ttl()` removes TTL from all three tables.
- A TTL setting on `checkpoint_blobs` that `enable_ttl()` did not write, such as one set by hand, turns the refresh off. The saver logs that at INFO.

!!! note
    TTL deletion is **eventual**: expired rows remain queryable until the background job runs. CockroachDB rate-limits deletions to minimize impact on foreground queries.

### Upgrading from 0.5

In 0.5, blob rows got no grace, and puts did not refresh them. After upgrading, run `enable_ttl()` once more with the same interval and cron to add the grace. Until then, puts refresh blob rows, but an idle thread's blob rows can expire up to a tenth of the interval before its last checkpoint.

### Graphs with DeltaChannel

LangGraph's `DeltaChannel`, which Deep Agents uses for its messages and files, stores each step's change and a full snapshot only every so many steps. To load the state, the saver rebuilds such a channel from older checkpoints and their writes, back to the last snapshot. TTL still expires those older checkpoints by age, even while the thread is active. So for these graphs, pick a TTL longer than a thread stays active, or delete finished threads with `delete_thread()`.

When the history of a delta channel stops at a missing checkpoint before its snapshot, `get_delta_channel_history` logs a WARNING. It names the thread and the missing checkpoint, and says that the state of those channels may be incomplete. What it returns is unchanged.

## Performance

The CockroachDB checkpointer includes several performance optimizations over the base PostgresSaver approach:

### Separate Lightweight Queries

Instead of correlated subqueries with `jsonb_agg`/`jsonb_build_object`, the checkpointer issues separate simple queries for checkpoints, blobs, and writes. Each query uses primary key lookups.

### Batch Fetching in list()

When listing multiple checkpoints, blobs and writes for **all** returned checkpoints are fetched in just 2 batch queries (instead of 2 per checkpoint). This reduces database round trips from 2N+1 to 3 total, regardless of how many checkpoints are returned.

### Raw BYTEA Deserialization

Blob data is read as raw BYTEA via psycopg3's binary cursor protocol, eliminating the overhead of hex-encoding in SQL and decoding in Python.

### Prepared Statement Caching

The `from_conn_string()` factory sets `prepare_threshold=5` by default, enabling server-side prepared statement caching after 5 executions of the same query. This eliminates repeated query parsing and planning overhead for hot paths. `prepare_threshold` is configurable via `from_conn_string(prepare_threshold=...)`, and `None` turns prepared statements off.

Statements sent in a pipeline are never prepared: the writes (`put`, `put_writes`, `delete_thread`) and, with `pipeline=True`, every statement. In pipeline mode psycopg records a statement as prepared before the server confirms it. If an earlier statement in the same batch fails, the server never prepares it, and later runs on that connection could fail with "prepared statement ... does not exist".

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
| Prepared statements | `prepare_threshold=0` (each query from its first run) | `prepare_threshold=5` (after five runs, outside pipelines) |
| Row-level TTL | Not supported | `enable_ttl()` / `disable_ttl()` |
| Async support | Yes | Yes |
| Thread deletion | Yes | Yes |
| Pipeline support | Yes | Yes |

## Next Steps

- [Checkpointer Example](../examples/checkpointer.md) - Working code examples
- [API Reference](../api/checkpointer.md) - Detailed API documentation
- [Chat History Guide](chat-history.md) - For long-term conversation storage
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - LangGraph framework docs
