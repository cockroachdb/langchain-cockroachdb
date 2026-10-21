# Store API

LangGraph long-term memory stores for CockroachDB.

## CockroachDBStore

Synchronous store.

::: langchain_cockroachdb.store.store.CockroachDBStore
    options:
      show_root_heading: true
      show_source: false

## AsyncCockroachDBStore

Asynchronous store.

::: langchain_cockroachdb.store.aio.AsyncCockroachDBStore
    options:
      show_root_heading: true
      show_source: false

## CockroachDBIndexConfig

Semantic search settings, passed as `index`.

::: langchain_cockroachdb.store.config.CockroachDBIndexConfig
    options:
      show_root_heading: true
      show_source: false

## CSPANNConfig

C-SPANN partition sizes, passed as `index["cspann"]`.

::: langchain_cockroachdb.store.config.CSPANNConfig
    options:
      show_root_heading: true
      show_source: false

## CockroachDBTTLConfig

Expiry settings, passed as `ttl`.

::: langchain_cockroachdb.store.config.CockroachDBTTLConfig
    options:
      show_root_heading: true
      show_source: false

## PoolConfig

Connection pool settings, passed as `pool_config` to `from_conn_string`.

::: langchain_cockroachdb.store.config.PoolConfig
    options:
      show_root_heading: true
      show_source: false

## Imports

| Name | `langchain_cockroachdb.store` | `langchain_cockroachdb` |
|------|-------------------------------|-------------------------|
| `CockroachDBStore` | Yes | Yes |
| `AsyncCockroachDBStore` | Yes | Yes |
| `CockroachDBIndexConfig` | Yes | Yes |
| `CockroachDBTTLConfig` | Yes | Yes |
| `PoolConfig` | Yes | No |
| `CSPANNConfig` | No, import it from `langchain_cockroachdb.store.config` | No |

## Key Methods

| Method | Async Method | Description |
|--------|--------------|-------------|
| `setup()` | `setup()` | Create the tables and run migrations, build the vector index with `index`, turn on TTL with `ttl` |
| `put()` | `aput()` | Store an item, or replace it |
| `get()` | `aget()` | Read an item |
| `search()` | `asearch()` | Search a namespace prefix, with a filter and an optional semantic `query` |
| `delete()` | `adelete()` | Delete an item |
| `list_namespaces()` | `alist_namespaces()` | List namespaces by prefix, suffix and depth |
| `batch()` | `abatch()` | Run several operations, with the writes in one transaction |
| `delete_namespace()` | `adelete_namespace()` | Delete a namespace and its descendants, return the item count |
| `sweep_ttl()` | `sweep_ttl()` | Delete expired items now, return the count |
| `start_ttl_sweeper()` | `start_ttl_sweeper()` | Sweep periodically in a background thread or task |
| `stop_ttl_sweeper()` | `stop_ttl_sweeper()` | Stop the sweeper |
| `disable_ttl()` | `disable_ttl()` | Turn off row-level TTL on the store's tables |
| `drop_vector_index()` | `adrop_vector_index()` | Drop the vector index, to rebuild it with `setup()` |

On `AsyncCockroachDBStore`, `setup()`, `sweep_ttl()`, `start_ttl_sweeper()`, `stop_ttl_sweeper()` and `disable_ttl()` are coroutines.

### Factory Methods

| Method | Description |
|--------|-------------|
| `from_conn_string(conn_string, ...)` | Create a store on a new connection, or on a new pool with `pool_config` (context manager) |

## Connection Types

Both stores accept these connection types, in autocommit mode with `dict_row`:

| Type | Description |
|------|-------------|
| `psycopg.Connection` / `AsyncConnection` | Single connection (simple scripts) |
| `psycopg_pool.ConnectionPool` / `AsyncConnectionPool` | Connection pool (production) |

## Examples

See the [Long-Term Memory Guide](../guides/long-term-memory.md) for usage.
