# Retries and Resilience

CockroachDB runs every transaction at SERIALIZABLE isolation and keeps serving through node restarts, drains and region failures. Clients are expected to retry a few kinds of errors, and langchain-cockroachdb does it on every database call: the vector store, chat history and both LangGraph checkpointers.

## What gets retried

| SQLSTATE | Meaning | Why a retry is safe |
|---|---|---|
| `40001` | Serialization failure ("restart transaction") | The transaction did not commit |
| `40003` | Result is ambiguous | Every write the library issues is idempotent |
| `08xxx` | Connection exceptions, including CockroachDB sqlproxy's `08C00` | Idempotent writes, and the pool replaces the connection |
| `57P01` | Node draining or shutting down | Idempotent writes |
| `58C00` | Range unavailable (CockroachDB) | Usually clears once leases move |
| `58C01` | Connection failure between CockroachDB nodes | Usually clears once leases move |
| `53300` | Too many connections | Connection slots free up |

Errors that arrive without a SQLSTATE, which is how the driver reports most connection failures, are retried when SQLAlchemy marks the connection invalidated or when the message matches a known connection-loss phrase. CockroachDB's uncategorized codes (`XXUUU`, `XX000`) are matched against known phrases too, for example "read tcp ...: i/o timeout" that can follow a canceled session. Failed logins and SSL or configuration mistakes are never retried, even though the driver reports them without a code, so a bad password does not trip an account lockout.

Everything else is raised right away, including statement timeouts (`57014`) and `XXA00` (the data was committed but a schema change failed).

## Configuring retries

Set a `RetryPolicy` once on the engine. The vector store and chat history inherit it; each can override it, and the checkpointers take their own.

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

| Field | Default | What it does |
|---|---|---|
| `max_attempts` | `5` | Total attempts, including the first |
| `initial_backoff` | `0.1` | Seconds before the first retry |
| `max_backoff` | `10.0` | Cap for any delay |
| `backoff_multiplier` | `2.0` | Growth between delays |
| `jitter` | `0.5` | Share of each delay that is randomized (0 turns it off) |
| `retryable_sqlstates` | `RETRYABLE_SQLSTATES` | Codes to retry |
| `retryable_sqlstate_classes` | `{"08"}` | Two-character classes to retry |
| `uncategorized_sqlstates` | `{"XXUUU", "XX000"}` | Codes too generic to decide on; message phrases decide |
| `transient_messages` | `TRANSIENT_MESSAGES` | Phrases that mark a code-less error as transient |
| `non_transient_messages` | `NON_TRANSIENT_MESSAGES` | Phrases that rule a retry out |
| `retry_on` | `None` | Your own predicate, replacing the rules above |
| `attempt_timeout` | `None` | Seconds before a hung query or write is abandoned and retried |
| `ddl_attempt_timeout` | `None` | The same guard, for each schema change |

Policies are plain data, so they can come from configuration files or environment variables:

```python
policy = RetryPolicy(**{"max_attempts": 6, "retryable_sqlstates": ["40001", "40003", "57P01"]})
```

The older `retry_max_attempts`, `retry_initial_backoff`, `retry_max_backoff`, `retry_backoff_multiplier` and `retry_jitter` arguments still work as shorthand.

## Idempotency

A `40001` means the transaction did not commit. After a `40003` or a dropped connection it may have. Retrying is only safe when running the work twice has the same effect as running it once, and every write the library makes is built that way: vector inserts are UPSERTs, deletes are deletes, chat messages carry ids chosen before the first attempt and are written with `ON CONFLICT DO NOTHING`, checkpoints are UPSERTs, and schema changes use `IF [NOT] EXISTS`.

For your own code, `async_retry_with_backoff` and `sync_retry_with_backoff` use the same rules. If the work is not idempotent, retry serialization failures only:

```python
from langchain_cockroachdb import async_retry_with_backoff, get_sqlstate


@async_retry_with_backoff(retry_on=lambda error: get_sqlstate(error) == "40001")
async def transfer(conn):
    ...
```

## Riding through a regional failover

While leases move after a node or region failure, statements can wait several seconds. If your application sets a `statement_timeout`, those statements fail with `57014`. Add the code to the policy, and cut off attempts that hang:

```python
from langchain_cockroachdb import RETRYABLE_SQLSTATES, RetryPolicy

failover = RetryPolicy(
    max_attempts=8,
    max_backoff=5.0,
    retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"},
    attempt_timeout=10,
)
```

Statement timeouts are not retried by default because they also fire for genuinely slow queries, and retrying those multiplies load on the queries that are already slow.

## Cutting off hung attempts

`attempt_timeout` gives each attempt a deadline. When it passes, the library shuts down the connection's socket, so the stuck call fails as a lost connection and the normal retry runs on a fresh connection. It starts once a connection is checked out, so `pool_pre_ping` and connection setup are bounded by `connect_timeout` and TCP keepalives, not by `attempt_timeout`. This works even when the server ignores cancel requests, which CockroachDB can do for a `COMMIT` while a range it wrote to has lost quorum, and it ends reads stalled by a network partition long before the operating system's TCP timeouts would.

It is off by default. Good starting points are 10 to 30 seconds for interactive paths and 60 to 120 seconds for batch work. Schema changes use `ddl_attempt_timeout`, also off by default, which bounds each schema change: one statement in the vector store, chat history and engine setup, and one migration step or one TTL statement in the checkpointers. Building an index on a large table can legitimately take minutes.

## Checkpointers: use a connection pool

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

`from_conn_string` accepts `retry_policy`, `connect_timeout`, `prepare_threshold` and `connect_kwargs` (for example `{"application_name": "my-service"}`).

## Connection settings

- `connect_timeout` defaults to 10 seconds. A value in the URL or `connect_args` wins; pass `connect_timeout=None` to leave it unset.
- `pool_pre_ping` (on by default) checks a pooled connection before use, so connections that died while idle are replaced without an error.
- TCP keepalives detect dead peers on idle connections, for example `?keepalives_idle=30&keepalives_interval=10&keepalives_count=3`.
- `gssencmode` defaults to `disable` for new connections. CockroachDB has no GSS encryption, and this also avoids a Kerberos lookup that can crash forked workers on macOS. Values in the URL or `connect_args` override it.

## Sync and async

Async methods run on your event loop. Sync methods run their database work on one background event loop, so they are safe from any number of threads and from inside a running event loop. Use one style per engine; the engine logs a warning if it sees both at the same time. Engines that use the built-in background loop are also safe across `os.fork()`: a child process gets a fresh pool and loop.

## Watching retries

Each retry logs one WARNING on the `langchain_cockroachdb.retry` logger:

```text
WARNING langchain_cockroachdb.retry: Retrying insert into public.resilience_1ceb80 after SerializationFailure (SQLSTATE 40001): restart transaction: demo fault 40001 (attempt 1 of 6 failed, next try in 0.10s)
```

and one more when it gives up. Raise that logger's level to hide them, or route them to your metrics.

## Trying it

`examples/resilience.py` makes CockroachDB raise real `40001`, `40003` and `57P01` errors and kills pooled connections while the vector store, chat history and checkpointer keep working.
