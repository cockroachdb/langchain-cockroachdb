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

## HistoryConfig

Version history settings, passed as `history`.

::: langchain_cockroachdb.store.config.HistoryConfig
    options:
      show_root_heading: true
      show_source: false

## PoolConfig

Connection pool settings, passed as `pool_config` to `from_conn_string`.

::: langchain_cockroachdb.store.config.PoolConfig
    options:
      show_root_heading: true
      show_source: false

## Reconciliation Types

The types that `reconcile`, `find_conflicts`, `resolve`, `record`, `confirm`, `put_if`, `delete_if` and `history` take and return. Import them from `langchain_cockroachdb.store`. See [Keeping Memories True](../guides/long-term-memory.md#keeping-memories-true) for how they fit together.

## Proposal

The new memory that a policy, or a judge in the two-step flow, decides about.

::: langchain_cockroachdb.store.reconcile.Proposal
    options:
      show_root_heading: true
      show_source: false

## Candidate

A live memory that the new memory may conflict with.

::: langchain_cockroachdb.store.reconcile.Candidate
    options:
      show_root_heading: true
      show_source: false

## Decision

What a policy or a judge decides: what to do with the new memory and with each candidate.

::: langchain_cockroachdb.store.reconcile.Decision
    options:
      show_root_heading: true
      show_source: false

## Action

What a decision does with one candidate.

::: langchain_cockroachdb.store.reconcile.Action
    options:
      show_root_heading: true
      show_source: false

## Conflicts

What `find_conflicts` returns, for `resolve`.

::: langchain_cockroachdb.store.reconcile.Conflicts
    options:
      show_root_heading: true
      show_source: false

## ConflictsChanged

Raised by `resolve` when a memory the decision depends on changed after `find_conflicts`.

::: langchain_cockroachdb.store.reconcile.ConflictsChanged
    options:
      show_root_heading: true
      show_source: false

## ReconcileResult

What `reconcile` or `resolve` did.

::: langchain_cockroachdb.store.reconcile.ReconcileResult
    options:
      show_root_heading: true
      show_source: false

## MemoryRecord

A live memory with its version, provenance, valid time and confirmations, as `record` returns it.

::: langchain_cockroachdb.store.reconcile.MemoryRecord
    options:
      show_root_heading: true
      show_source: false

## MemoryVersion

An old version of a memory, as `history` returns it.

::: langchain_cockroachdb.store.reconcile.MemoryVersion
    options:
      show_root_heading: true
      show_source: false

## ConditionalWrite

What `put_if` or `delete_if` did.

::: langchain_cockroachdb.store.reconcile.ConditionalWrite
    options:
      show_root_heading: true
      show_source: false

## RECONCILE_SERDE_TYPES

The classes a checkpoint serializer must allow, so reconcile values can wait in graph state.

::: langchain_cockroachdb.store.reconcile.RECONCILE_SERDE_TYPES
    options:
      show_root_heading: true
      show_source: false

## Policies

A policy is a function of the `Proposal` and the sequence of candidates that returns a `Decision`, typed as `Policy` (`Callable[[Proposal, Sequence[Candidate]], Decision]`). It runs inside the reconcile's transaction, and again on every retry, so it must be a plain function, fast and deterministic, and must not call the store. These are the built-in ones, and `validate_decision`, which checks a decision as `reconcile` and `resolve` do.

::: langchain_cockroachdb.store.reconcile.supersede_same_slot
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.store.reconcile.skip_duplicates
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.store.reconcile.supersede_similar
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.store.reconcile.keep_all
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.store.reconcile.first_of
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.store.reconcile.validate_decision
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
| `HistoryConfig` | Yes | No |
| `Proposal` | Yes | No |
| `Candidate` | Yes | No |
| `Decision` | Yes | No |
| `Action` | Yes | No |
| `Conflicts` | Yes | No |
| `ConflictsChanged` | Yes | No |
| `ReconcileResult` | Yes | No |
| `MemoryRecord` | Yes | No |
| `MemoryVersion` | Yes | No |
| `ConditionalWrite` | Yes | No |
| `RECONCILE_SERDE_TYPES` | Yes | No |
| `Policy` | Yes | No |
| `supersede_same_slot` | Yes | No |
| `skip_duplicates` | Yes | No |
| `supersede_similar` | Yes | No |
| `keep_all` | Yes | No |
| `first_of` | Yes | No |
| `validate_decision` | Yes | No |
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
| `delete_namespace()` | `adelete_namespace()` | Delete a namespace and its descendants with their vectors and history, return the item count |
| `reconcile()` | `areconcile()` | Write a memory and settle the memories it may conflict with, under a policy, in one transaction |
| `find_conflicts()` | `afind_conflicts()` | Find the memories a new one may conflict with, without writing anything |
| `resolve()` | `aresolve()` | Apply a decision about what `find_conflicts` found, if nothing it depends on changed |
| `record()` | `arecord()` | Read a memory with its version, provenance, valid time and confirmations |
| `confirm()` | `aconfirm()` | Confirm a memory: renew its expiry and count the confirmation |
| `put_if()` | `aput_if()` | Write a memory only if it is still the one `record` returned |
| `delete_if()` | `adelete_if()` | Delete a memory only if it is still the one `record` returned |
| `history()` | `ahistory()` | List a memory's old versions, newest first, on a store with history |
| `purge()` | `apurge()` | Remove a memory with its vectors and its history |
| `sweep_ttl()` | `sweep_ttl()` | Delete expired items now, return the count |
| `start_ttl_sweeper()` | `start_ttl_sweeper()` | Sweep periodically in a background thread or task |
| `stop_ttl_sweeper()` | `stop_ttl_sweeper()` | Stop the sweeper |
| `disable_ttl()` | `adisable_ttl()` | Turn off row-level TTL on the store's tables |
| `drop_vector_index()` | `adrop_vector_index()` | Drop the vector index, to rebuild it with `setup()` |

On `AsyncCockroachDBStore`, `setup()`, `sweep_ttl()`, `start_ttl_sweeper()` and `stop_ttl_sweeper()` are coroutines without the `a` prefix, as in AsyncPostgresStore.

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

See the [Long-Term Memory Guide](../guides/long-term-memory.md) for usage, and [Keeping Memories True](../guides/long-term-memory.md#keeping-memories-true) and the [Memory Reconciliation Example](../examples/memory-reconciliation.md) for the reconciliation methods.
