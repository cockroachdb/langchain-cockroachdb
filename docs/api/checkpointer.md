# Checkpointer API

LangGraph checkpoint savers for CockroachDB.

## CockroachDBSaver

Synchronous checkpointer for LangGraph workflows.

::: langchain_cockroachdb.checkpointer.saver.CockroachDBSaver
    options:
      show_root_heading: true
      show_source: false

## AsyncCockroachDBSaver

Asynchronous checkpointer for LangGraph workflows.

::: langchain_cockroachdb.checkpointer.async_saver.AsyncCockroachDBSaver
    options:
      show_root_heading: true
      show_source: false

## Key Methods

### Shared (Sync & Async)

| Method | Async Method | Description |
|--------|-------------|-------------|
| `setup()` | `setup()` | Create tables and run migrations |
| `put()` | `aput()` | Save a checkpoint |
| `get_tuple()` | `aget_tuple()` | Retrieve a checkpoint |
| `list()` | `alist()` | List checkpoints with optional filters |
| `put_writes()` | `aput_writes()` | Store intermediate writes |
| `delete_thread()` | `adelete_thread()` | Delete all checkpoints for a thread |

### Factory Methods

| Method | Description |
|--------|-------------|
| `from_conn_string(conn_string)` | Create saver from a connection string (context manager) |

## Connection Types

Both savers accept these connection types:

| Type | Description |
|------|-------------|
| `psycopg.Connection` / `AsyncConnection` | Single connection (simple scripts) |
| `psycopg_pool.ConnectionPool` / `AsyncConnectionPool` | Connection pool (production) |

## Examples

See [Checkpointer Guide](../guides/checkpointer.md) for comprehensive usage examples.
