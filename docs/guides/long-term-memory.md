# LangGraph Long-Term Memory

Keep memories that outlive a conversation thread in CockroachDB: user preferences, facts an agent learned, profiles.

## Overview

A LangGraph checkpointer keeps the state of one thread, which is short-term memory. A store keeps what every thread can read, which is long-term memory. `CockroachDBStore` and `AsyncCockroachDBStore` implement LangGraph's `BaseStore`, so they work wherever LangGraph takes a store: `create_agent(store=...)`, `StateGraph.compile(store=...)` and Deep Agents.

The API follows `PostgresStore` from [langgraph-checkpoint-postgres](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres): `from_conn_string`, `setup()`, the `index` and `ttl` arguments and the TTL sweeper methods work the same way. What this store adds or does differently is listed in [Differences from PostgresStore](#differences-from-postgresstore).

Memories can stop being true. [Keeping Memories True](#keeping-memories-true) shows how the store settles a new memory against the ones it may conflict with, keeps a history of old versions with their provenance, lets expiry follow confirmations, and keeps two writers from losing an update.

## Installation

```bash
pip install langchain-cockroachdb
```

The store needs `langgraph-checkpoint>=2.1.2` and psycopg 3, which are dependencies. `create_agent` comes from the `langchain` package. Semantic search needs an embedding model, for example `OpenAIEmbeddings` from `langchain-openai`.

## Quick Start

### Synchronous Usage

```python
from langchain_cockroachdb import CockroachDBStore

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

with CockroachDBStore.from_conn_string(DB_URI) as store:
    store.setup()

    namespace = ("user", "alice", "memories")
    store.put(namespace, "seat", {"text": "Prefers window seats"})

    item = store.get(namespace, "seat")
    print(item.value)  # {'text': 'Prefers window seats'}

    for item in store.search(("user", "alice")):
        print(item.namespace, item.key, item.value)
```

### Asynchronous Usage

```python
from langchain_cockroachdb import AsyncCockroachDBStore

async with AsyncCockroachDBStore.from_conn_string(DB_URI) as store:
    await store.setup()

    namespace = ("user", "alice", "memories")
    await store.aput(namespace, "seat", {"text": "Prefers window seats"})
    item = await store.aget(namespace, "seat")
```

`setup()` must run before the store is used for the first time. It creates the tables and runs the migrations. It is idempotent and safe to run from several processes at once.

## Connections and Pools

### Factory Method (Recommended)

`from_conn_string` opens one connection, or a pool when you pass `pool_config`. Use a pool for services: it replaces broken connections, while a single connection cannot recover once it drops.

```python
from langchain_cockroachdb import CockroachDBStore

with CockroachDBStore.from_conn_string(
    DB_URI,
    pool_config={"min_size": 2, "max_size": 20},
) as store:
    store.setup()
```

`pool_config` is a `PoolConfig`, with the same meaning as in PostgresStore:

| Key | Default | What it does |
|---|---|---|
| `min_size` | `1` | Connections the pool keeps open |
| `max_size` | `None` | Most connections the pool opens. `None` means `min_size` |
| `kwargs` | `{}` | Extra psycopg connection options for every pooled connection. The store sets `autocommit`, `row_factory` and `prepare_threshold` itself, and they raise `ValueError` here |

Without `max_size`, the pool has `min_size` connections, so set it for the concurrency you need.

All `from_conn_string` options:

| Parameter | Default | What it does |
|---|---|---|
| `pipeline` | `False` | Pipeline mode on the single connection. Ignored with `pool_config` |
| `pool_config` | `None` | Open a pool with these settings instead of one connection |
| `index` | `None` | Semantic search settings, see [Semantic Search](#semantic-search) |
| `ttl` | `None` | Expiry settings, see [TTL](#ttl) |
| `history` | `None` | Version history: `True`, or a `HistoryConfig` such as `{"ttl_days": 365}`. See [History](#history) |
| `reconcile_scan_limit` | `2000` | The most vectors a namespace may hold for its similar candidates to be ranked exactly. See [The Scan Limit](#the-scan-limit) |
| `retry_policy` | `RetryPolicy()` | How operations retry |
| `connect_timeout` | `10.0` | Seconds to wait when connecting, or `None` |
| `prepare_threshold` | `5` | psycopg prepared statement threshold, or `None` to disable. Statements sent in a pipeline, which are nearly all of the store's, are never prepared |
| `connect_kwargs` | `None` | Extra psycopg connection options, for example `{"application_name": "my-service"}` |
| `table_name` | `"store"` | Name of the items table |
| `schema` | `"public"` | Schema of the store's tables |

### Your Own Pool or Connection

The constructor takes a psycopg connection or a `psycopg_pool` pool, plus `pipe`, `index`, `ttl`, `history`, `reconcile_scan_limit`, `retry_policy`, `table_name` and `schema`. The connection or pool must use autocommit and `dict_row`. One pool can serve the store and the checkpointer:

```python
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import CockroachDBSaver, CockroachDBStore

with ConnectionPool(
    "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
    max_size=20,
    kwargs={"autocommit": True, "row_factory": dict_row},
    open=True,
) as pool:
    store = CockroachDBStore(pool)
    store.setup()
    checkpointer = CockroachDBSaver(pool)
    checkpointer.setup()
```

Use `postgresql://` for raw psycopg connections; `from_conn_string` converts `cockroachdb://` for you. Pools you build yourself do not get the library's `gssencmode=disable` default. On macOS with prefork servers, add `gssencmode=disable` to the pool's conninfo or kwargs.

### Concurrency

A `CockroachDBStore` on a pool runs batches concurrently, each on its own pooled connection. On a single connection, including `pipeline=True`, it runs one batch at a time.

`AsyncCockroachDBStore` sends `aget`, `aput`, `asearch` and its other awaited calls through LangGraph's `AsyncBatchedBaseStore` queue. The queue runs one `abatch` at a time and merges the ops queued in the same event loop tick into one batch. So a larger pool does not run those calls in parallel, though it still replaces broken connections. Direct `abatch` calls, and `batch` calls from other threads, run concurrently on a pool.

Create an `AsyncCockroachDBStore` inside a running event loop. Its sync methods (`get`, `put` and the others) are for other threads while that loop runs. They run the async methods on it, so `get`, `put`, `search`, `delete` and `list_namespaces` join the same queue. Called from the loop's own thread they raise `asyncio.InvalidStateError`. If the loop is closed or not running, or the call comes from a forked child process, they raise `RuntimeError` instead of waiting forever, as the sync methods of `AsyncCockroachDBSaver` do.

## Retries and Atomic Writes

Every call retries transient CockroachDB errors under the store's `RetryPolicy`. Pass `retry_policy=` to change it; see [Retries and Resilience](resilience.md) for the fields. Every write is an UPSERT, a DELETE, an UPDATE that pushes back an expiry or confirms a memory, or an INSERT into history that skips versions already there. Each is safe to run twice, so a retry after an ambiguous result is safe. For the rare windows where a retry after a lost commit cannot tell its own earlier commit from another writer's work, see [Retries and Rebuilt Results](#retries-and-rebuilt-results) in the reconciliation section.

A `batch()` call validates every op and serializes every value before any statement runs. It then runs its reads, one round trip each for its gets, its searches and its listings, and commits all its puts and deletes together in one transaction. Semantic searches under a namespace add a round trip before the searches, and sometimes one after them, as [How Searches Use the Index](#how-searches-use-the-index) explains. Embeddings are computed once, before the first attempt, so a retry does not call the embedding model again. When one batch puts the same key twice, the last put wins.

On CockroachDB v25.4, a TTL change can meet a schema change that another process runs on the same table, such as a vector index another `setup()` builds over many vectors. The TTL steps of `setup()`, `disable_ttl()` and `adisable_ttl()` retry that error for up to ten minutes by default, for each table, as described in [Configuring retries](resilience.md#configuring-retries). `setup()` also reads each table's TTL settings first and changes them only when they differ, so the processes that start after the first one change nothing.

## Namespaces and Items

An item is a JSON object, addressed by a namespace (a tuple of labels) and a key. LangGraph's `BaseStore` requires every label to be a non-empty string without `.`, in every store, and reserves `"langgraph"` as the first label. This store keeps a namespace as its labels joined with `.`, as PostgresStore does, and checks the labels again for ops passed straight to `batch()`.

```python
store.put(("user", "alice", "memories"), "seat", {"text": "Prefers window seats", "stars": 5})
store.put(("user", "alice", "profile"), "name", {"name": "Alice"})

item = store.get(("user", "alice", "profile"), "name")
print(item.value, item.created_at, item.updated_at)

store.delete(("user", "alice", "profile"), "name")
```

A value that cannot be stored as JSON, including NaN and infinity, raises `ValueError` before anything is written.

### Listing Namespaces

With items in `("user", "alice", "memories")`, `("user", "alice", "profile")` and `("user", "bob", "memories")`:

```python
store.list_namespaces(prefix=("user", "alice"))
# [('user', 'alice', 'memories'), ('user', 'alice', 'profile')]

store.list_namespaces(prefix=("user", "*"), max_depth=2)
# [('user', 'alice'), ('user', 'bob')]

store.list_namespaces(suffix=("memories",))
```

`*` matches one label. `max_depth` cuts each namespace to its first `max_depth` labels. Results are distinct and sorted by their text with the labels joined by `.`, as in PostgresStore, and `limit` (default 100) and `offset` page through them. That is the order of the tuples unless a label holds a character that sorts before `.`, such as a space, `-` or `#`: `("a#b", "c")` comes before `("a", "b")`, where `InMemoryStore` puts it after. A `prefix` that starts with plain labels, such as `("user", "alice")` or `("user", "*", "memories")`, reads only the rows under those labels.

### Searching and Filtering

`search` returns the items in a namespace and its descendants, so `("user", "alice")` also finds items in `("user", "alice", "memories")`. Without a `query`, the newest items come first.

```python
results = store.search(("user", "alice"), filter={"stars": {"$gte": 4}}, limit=10, offset=0)
```

| Filter | Matches |
|---|---|
| `{"topic": "travel"}` | `topic` equals `"travel"`. A list compares as JSON |
| `{"topic": {"$eq": "travel"}}` | The same |
| `{"topic": {"$ne": "travel"}}` | `topic` is something else, or is missing |
| `{"stars": {"$gt": 3}}` | `stars` is a number greater than 3. Also `$gte`, `$lt` and `$lte` |
| `{"name": {"$lt": "m"}}` | The text of `name` sorts before `"m"` |
| `{"address": {"city": "Lisbon"}}` | The nested key `address.city` equals `"Lisbon"` |

Every condition must match. A comparison with a number only matches numeric values, and a comparison with a string compares text. Comparison operators do not take booleans, and an unknown operator raises `ValueError`.

## Semantic Search

Pass `index` to embed items when they are put and to rank searches by similarity:

```python
from langchain_openai import OpenAIEmbeddings

from langchain_cockroachdb import CockroachDBStore

with CockroachDBStore.from_conn_string(
    DB_URI,
    index={
        "dims": 1536,
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "fields": ["text"],
    },
) as store:
    store.setup()
    store.put(("user", "alice", "memories"), "seat", {"text": "Prefers window seats"})

    results = store.search(("user", "alice", "memories"), query="Where does Alice sit on planes?")
    for item in results:
        print(item.score, item.value["text"])
```

With an index, `setup()` also creates the vectors table and, unless `flat` is set, a C-SPANN vector index on it.

### Index Settings

`index` is a `CockroachDBIndexConfig`: LangGraph's `IndexConfig` with CockroachDB options added.

| Key | Default | What it does |
|---|---|---|
| `dims` | required | Vector dimensions, fixed when the vectors table is created |
| `embed` | required | An `Embeddings` object, a sync or async embedding function, or a provider string such as `"openai:text-embedding-3-small"` (needs the `langchain` package) |
| `fields` | `["$"]` | Paths in each value to embed. `"$"` embeds the whole value as JSON, and `"messages[*].content"` embeds each message on its own |
| `distance_type` | `"cosine"` | `"cosine"`, `"l2"` or `"inner_product"` |
| `cspann` | `None` | C-SPANN partition sizes, `{"min_partition_size": 10, "max_partition_size": 100}`. Give both or neither; the maximum must be at least 4 times the minimum. Without it the index uses CockroachDB's defaults |
| `flat` | `False` | Do not create a vector index. Searches rank every matching vector exactly, in one round trip, which suits small stores |
| `beam_size` | `None` | C-SPANN beam width. A larger beam finds more of the true nearest neighbors and reads more of the index. A small one can return fewer candidates than a search asks for. If that leaves a page short, the search runs its second, exact query |

A put can override `fields`: `index=False` stores the item without vectors, and `index=["text", "summary"]` embeds those paths instead.

```python
store.put(("user", "alice", "profile"), "name", {"name": "Alice"}, index=False)
```

Each embedded text is one vector row, so an item can have several. Putting an item again replaces all its vectors, and deleting it deletes them. A search returns each item once, with the score of its best match. Higher scores are better: the score is `1 - distance` for cosine, and the negated distance for `l2` and `inner_product`. Results are ordered by score, then by namespace and key.

### How Searches Use the Index

The vector index has the namespace as a prefix column, because most memory searches name one user's namespace. A search with a `query` under a namespace prefix runs in two steps. First a short read lists the namespaces at or under the prefix that have vectors. Then the ranked search runs on the index, limited to those namespaces. For one user's namespace that list has one entry, so the search reads only that namespace's part of the index.

The index finds approximate nearest neighbors, which `beam_size` tunes. It gives the search a window of candidate vectors, twice as many as the page needs and at least 41, which cost the index about the same as fewer. The filter and the expiry check then apply to those candidates. A selective filter, or many expired items that the TTL job has not deleted yet, can leave fewer matches in the window than the page asks for. Then the search runs a second, exact query over the namespaces the first step found: it applies the filter and the expiry check first, ranks every match and returns that page instead. A short first page counts as exact only when the index returned every vector those namespaces hold, which the store checks with a count of their vectors that stops at the window's size. That is the usual case when the namespaces hold fewer vectors than the window. A small `beam_size` or small partitions can cap what the index returns, and then the second query runs as well. For one user's memories the second query reads little, but it costs a round trip.

Some searches skip the index and rank every match exactly in one query: a search of the whole store, with an empty prefix; every search of a store with `flat` set, which skips the first step too; and a search under a prefix where more than 1000 namespaces have vectors (`MAX_SEARCH_NAMESPACES` in `langchain_cockroachdb.store.sql`). Like PostgresStore, they apply the filter and the expiry check before they rank.

`beam_size` is applied to each search with `SET LOCAL`, inside the search's own transaction, so it does not stay on a pooled connection.

Expired items are never returned, but their vectors stay in the table until the TTL job or `sweep_ttl()` deletes them, and until then they take up places in the window. With TTLs of minutes and the default hourly job, they can fill it, and more searches then need the second query. Set `sweep_interval_minutes` (or `ttl_job_cron`) lower when TTLs are minutes.

A `query` on a store without `index` runs a plain search: it filters and returns the newest items first, not the most similar ones. The store logs one WARNING about it, once per store.

### Changing the Index Configuration

`setup()` compares the configuration with the existing vectors table and index:

| Change after the first `setup()` | What happens |
|---|---|
| `dims` | `setup()` raises `ValueError`. Use the existing dims, or another `table_name` |
| `distance_type` | `setup()` raises `ValueError`. Drop the index and run `setup()` again to build it with the new distance |
| `cspann` sizes | `setup()` logs a WARNING and keeps the existing index. Drop it and run `setup()` again to use the new sizes |
| `flat` turned on | `setup()` logs a WARNING and keeps the index. Searches no longer use it, but puts still update it. Drop it if no other store needs it |
| `fields` | Applies to items put from then on. Existing items keep their vectors until they are put again |

```python
store.drop_vector_index()  # async: await store.adrop_vector_index()
store.setup()  # builds the index again with the store's current settings
```

Searches scan the vectors until the new index is built. Adding `index` to a store that already holds items creates the vectors table and index, and semantic search finds the existing items once they are put again.

## TTL

Pass `ttl` to give items an expiry. `default_ttl` and the `ttl` argument of `put` are in minutes.

```python
from langchain_cockroachdb import CockroachDBStore

with CockroachDBStore.from_conn_string(
    DB_URI,
    ttl={"default_ttl": 60 * 24 * 30, "refresh_on_read": True},
) as store:
    store.setup()  # turns on CockroachDB row-level TTL

    memories = ("user", "alice", "memories")
    store.put(memories, "seat", {"text": "Prefers window seats"})  # expires in 30 days
    store.put(memories, "code", {"text": "Door code 4321"}, ttl=10)  # expires in 10 minutes
    store.put(("user", "alice", "profile"), "name", {"name": "Alice"}, ttl=None)  # never expires
```

`ttl` is a `CockroachDBTTLConfig`: LangGraph's `TTLConfig` with CockroachDB options added.

| Key | Default | What it does |
|---|---|---|
| `default_ttl` | `None` | Minutes until an item expires when `put` gets no `ttl`. `None` means no expiry |
| `refresh_on_read` | `True` | Whether `get` and `search` push back the expiry of the items they return |
| `omit_expired` | `True` | Whether reads, searches and listings leave out expired items |
| `sweep_interval_minutes` | `None` | How often CockroachDB's TTL job runs: every N minutes below 60 (N must be a whole number), hourly at 60 or more. Also the default interval of `start_ttl_sweeper()` |
| `ttl_job_cron` | `None` | A cron expression for the TTL job, such as `"@daily"`. Overrides `sweep_interval_minutes` |

### The TTL Job

CockroachDB deletes expired rows itself, with [row-level TTL](https://www.cockroachlabs.com/docs/stable/row-level-ttl). With a `ttl` configuration, `setup()` sets `ttl_expiration_expression` to the `expires_at` column on the items table, and on the vectors table when there is an index. The job runs on the cron that `sweep_interval_minutes` gives, or `ttl_job_cron`, and hourly (`@hourly`) by default. A changed `sweep_interval_minutes` or `ttl_job_cron` takes effect at the next `setup()`, which leaves settings that already match alone. Rows without an expiry are never deleted. Vector rows carry their item's expiry, so they expire with it.

The job runs on a schedule, so an expired item stays in the table until a run deletes it. Reads, searches and listings leave expired items out anyway, because `omit_expired` is on by default. That holds even for a store built without `ttl`, since another process may write items with an expiry to the same tables. Pass `ttl={"omit_expired": False}` to see expired items until the job deletes them. Their vectors stay as well, and make semantic searches run their second query more often, so with TTLs of minutes set `sweep_interval_minutes` lower than the default hour, as [How Searches Use the Index](#how-searches-use-the-index) explains.

### Refresh on Read

With `refresh_on_read`, a `get` or a `search` sets the expiry of the items it returns to now plus their TTL. Only items that have a TTL change, and vectors are refreshed with their item. The refresh is an UPDATE inside the read statement, so reading items that have a TTL is also a write. For memory that is read very often, set `refresh_on_read` to `False` or use a longer TTL. One call can decide for itself with `refresh_ttl`:

```python
store.get(("user", "alice", "memories"), "seat", refresh_ttl=False)
```

With `omit_expired` off and refresh on, reading an expired item that the job has not deleted yet gives it a new expiry, so it stays.

### Stores Without a TTL Configuration

A store built without `ttl` applies no default expiry and never refreshes one. An explicit `put(..., ttl=10)` still writes an expiry. Its `setup()` does not turn on the TTL job, and it leaves an existing TTL setting alone, so a process that starts without `ttl` cannot switch off another's expiry. The items it writes with a TTL are hidden once they expire, and are deleted by `sweep_ttl()`, or by the TTL job if another store turned it on.

### Sweeping and Disabling TTL

`sweep_ttl()` deletes expired items now and returns how many it deleted. `start_ttl_sweeper()` runs it in a background thread, every `sweep_interval_minutes` or else every 5 minutes, and `stop_ttl_sweeper()` stops it, as in PostgresStore. They are optional here: the TTL job already deletes expired rows, and the sweeper only deletes them sooner. Without a `ttl` configuration, `start_ttl_sweeper()` does nothing, as in PostgresStore.

```python
deleted = store.sweep_ttl()

store.start_ttl_sweeper(sweep_interval_minutes=5)
...
store.stop_ttl_sweeper()
```

`disable_ttl()` (async: `adisable_ttl()`) turns off row-level TTL on the store's tables. `setup()` never does, even without a `ttl` configuration. Expired items stay hidden while `omit_expired` is on.

```python
store.disable_ttl()  # async: await store.adisable_ttl()
```

On `AsyncCockroachDBStore`, `sweep_ttl()`, `start_ttl_sweeper()` (it returns an `asyncio.Task`) and `stop_ttl_sweeper()` are coroutines with the same names, as in AsyncPostgresStore. Turning TTL off is `await store.adisable_ttl()`.

## Deleting a Namespace

```python
deleted = store.delete_namespace(("user", "alice"))  # async: await store.adelete_namespace(...)
```

`delete_namespace` deletes every item in a namespace and its descendants, with their vectors and any history they have, and returns the number of items deleted. `("user", "alice")` covers `("user", "alice", "memories")` but not `("user", "alice2")`. It runs as one statement, so it commits completely or not at all. If a retry follows a commit whose result was lost, the count is that of the retry, usually 0. An empty prefix raises `InvalidNamespaceError`. `BaseStore` has no such method; it is specific to this store.

A store created without history also deletes the namespace's rows in a history table left from when history was on, as [Forgetting](#forgetting) explains. A role that did not run `setup()` then needs SELECT and DELETE on that table. Without them the whole call fails with a permission error and deletes nothing, rather than forget only part of the namespace.

## Tables

| Table | Holds |
|---|---|
| `store` | Items: `prefix` (the namespace), `key`, `value` (JSONB), `created_at`, `updated_at`, `expires_at` and `ttl_minutes`, and from migration 2 `version`, `provenance`, `confirmed_at`, `confirmations`, `write_id`, `valid_at` and `confirm_id` |
| `store_vectors` | One row per embedded text, with the item's expiry. Created only with `index` |
| `store_history` | Old versions of items: `prefix`, `key`, `version`, `value`, `provenance`, `valid_at`, `invalid_at`, `written_at`, `ended_at`, `reason`, `superseded_by`, `confirmations`, `confirmed_at`, `write_id` and `expires_at`. Created only with `history` |
| `store_migrations` | The migration versions applied |

`table_name="memories"` names the tables `memories`, `memories_vectors`, `memories_history` and `memories_migrations`, so several stores can share a database. `schema=` puts them in another schema, which must exist. The default schema is `public`.

The store creates plain tables, with no multi-region locality such as `REGIONAL BY ROW` or `GLOBAL`. You can set one yourself with `ALTER TABLE`; `setup()` leaves it as you set it.

## Use with LangGraph

### With create_agent

The tools annotate their runtime as `ToolRuntime[Context, dict]`, which names both the context type and the agent state type. Python 3.10 cannot fill in the default of the second, so `ToolRuntime[Context]` fails there. A plain `ToolRuntime` expects no context, and pydantic warns about the `Context` it gets.

Tools reach the store through `ToolRuntime`, and the user id comes from the run's context:

```python
import uuid
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_openai import OpenAIEmbeddings

from langchain_cockroachdb import CockroachDBSaver, CockroachDBStore

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"


@dataclass
class Context:
    user_id: str


@tool
def save_memory(text: str, runtime: ToolRuntime[Context, dict]) -> str:
    """Save a fact about the user for later conversations."""
    assert runtime.store is not None
    namespace = ("user", runtime.context.user_id, "memories")
    runtime.store.put(namespace, str(uuid.uuid4()), {"text": text})
    return f"Saved: {text}"


@tool
def recall(query: str, runtime: ToolRuntime[Context, dict]) -> str:
    """Search the user's memories."""
    assert runtime.store is not None
    namespace = ("user", runtime.context.user_id, "memories")
    items = runtime.store.search(namespace, query=query, limit=3)
    return "\n".join(item.value["text"] for item in items) or "No memories found."


with (
    CockroachDBStore.from_conn_string(
        DB_URI,
        pool_config={"max_size": 10},
        index={
            "dims": 1536,
            "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
            "fields": ["text"],
        },
        ttl={"default_ttl": 60 * 24 * 30},
    ) as store,
    CockroachDBSaver.from_conn_string(DB_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()
    agent = create_agent(
        "openai:gpt-4o-mini",
        tools=[save_memory, recall],
        context_schema=Context,
        checkpointer=checkpointer,
        store=store,
    )

    alice = Context(user_id="alice")
    agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that I prefer window seats."}]},
        {"configurable": {"thread_id": "thread-1"}},
        context=alice,
    )
    # A new thread starts with no messages, but the store still has the memory
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Where do I like to sit on a plane?"}]},
        {"configurable": {"thread_id": "thread-2"}},
        context=alice,
    )
    print(result["messages"][-1].content)
```

With `AsyncCockroachDBStore`, write the tools as async functions that call `runtime.store.aput` and `runtime.store.asearch`, and call `agent.ainvoke`. [`examples/long_term_memory.py`](../examples/long-term-memory.md) runs both versions with a scripted model.

### In a StateGraph

A node gets the store from its `Runtime`, and `compile` takes it next to the checkpointer:

```python
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime

from langchain_cockroachdb import CockroachDBSaver, CockroachDBStore

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
llm = ChatOpenAI(model="gpt-4o-mini")
index = {
    "dims": 1536,
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "fields": ["text"],
}


@dataclass
class Context:
    user_id: str


def chatbot(state: MessagesState, runtime: Runtime[Context]):
    assert runtime.store is not None
    namespace = ("user", runtime.context.user_id, "memories")
    memories = runtime.store.search(namespace, query=state["messages"][-1].content, limit=3)
    known = "\n".join(item.value["text"] for item in memories)
    system = {"role": "system", "content": f"What you know about the user:\n{known}"}
    return {"messages": [llm.invoke([system, *state["messages"]])]}


builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")

with (
    CockroachDBStore.from_conn_string(DB_URI, index=index) as store,
    CockroachDBSaver.from_conn_string(DB_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer, store=store)
    graph.invoke(
        {"messages": [{"role": "user", "content": "Where do I like to sit on a plane?"}]},
        {"configurable": {"thread_id": "thread-3"}},
        context=Context(user_id="alice"),
    )
```

### With Deep Agents

[Deep Agents](https://docs.langchain.com/oss/python/deepagents/long-term-memory) keeps files in a store when you route a path to a `StoreBackend`:

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from langchain_cockroachdb import CockroachDBStore

with CockroachDBStore.from_conn_string(DB_URI, pool_config={"max_size": 10}) as store:
    store.setup()
    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
        backend=CompositeBackend(
            default=StateBackend(),
            routes={"/memories/": StoreBackend(namespace=lambda rt: ("memories",))},
        ),
        store=store,
    )
```

Files under `/memories/` go to the store and outlive the thread. Other files stay in the thread's state. Deep Agents accepts `.` in namespace labels, but LangGraph's `BaseStore` rejects such a label with `InvalidNamespaceError` in every store, `InMemoryStore` and `PostgresStore` included. Build labels from ids without dots: replace the dots in an email address, for example.

This snippet was checked against the Deep Agents documentation (deepagents 0.5 and later), not run in this project's tests.

## Keeping Memories True

Semantic search finds memories that are similar to a query, not memories that are still true. A preference the user changed months ago can keep steering an agent, and two memories saved under different keys can contradict each other unnoticed. The store has methods that help keep memory true. They find the memories a new one may conflict with, let your code decide what to do with each, apply the decision in one serializable transaction, and keep a history of what the agent believed, when, and why.

Deciding what contradicts what stays with your application, which may use rules or an LLM. The store never calls an LLM itself.

| Method | What it does |
|---|---|
| `reconcile` | Writes a memory and settles the memories it may conflict with, under a policy, in one transaction |
| `find_conflicts` | Finds the memories a new one may conflict with, without writing anything |
| `resolve` | Applies a decision about what `find_conflicts` found, if nothing it depends on changed since |
| `record` | Reads a memory with its version, provenance, valid time and confirmations |
| `confirm` | Marks a memory as confirmed: renews its expiry and counts the confirmation |
| `put_if`, `delete_if` | Write or delete a memory only if it is still the one you read |
| `history` | Lists the old versions of a memory, or of every memory in a namespace, on a store with history |
| `purge` | Removes a memory with its vectors and its history |

`AsyncCockroachDBStore` has each of them as a coroutine with an `a` prefix: `areconcile`, `afind_conflicts`, `aresolve`, `arecord`, `aconfirm`, `aput_if`, `adelete_if`, `ahistory` and `apurge`. They run on the caller's event loop, not through the store's batch queue, so on a pool they run concurrently. The types and the built-in policies are imported from `langchain_cockroachdb.store`. [`examples/memory_reconciliation.py`](../examples/memory-reconciliation.md) runs through all of this with a scripted judge.

### Turning It On

`setup()` adds the columns these methods need to the items table. To keep old versions, pass `history=True`, or a `HistoryConfig` such as `{"ttl_days": 365}` that gives them a TTL, and `setup()` creates the history table too.

```python
from langchain_openai import OpenAIEmbeddings

from langchain_cockroachdb import CockroachDBStore

with CockroachDBStore.from_conn_string(
    DB_URI,
    index={
        "dims": 1536,
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "fields": ["text"],
    },
    ttl={"default_ttl": 60 * 24 * 30, "refresh_on_read": False},
    history={"ttl_days": 365},
) as store:
    store.setup()
```

The examples below use this `store`, one after another. A store upgraded from 0.5 needs `setup()` once, as [Upgrading From 0.5](#upgrading-from-05) explains.

### Reconciling a Memory

A slot names the fact a memory states. Here each value's `"slot"` field holds it, and `slot="slot"` tells `reconcile` which field to compare, so a new seat preference finds the old one, with or without an index:

```python
from langchain_cockroachdb.store import first_of, skip_duplicates, supersede_same_slot

memories = ("user", "alice", "memories")
policy = first_of(skip_duplicates(), supersede_same_slot())

store.reconcile(
    memories, "seat-1", {"slot": "seat", "text": "Prefers window seats"}, slot="slot", policy=policy
)
aisle = store.reconcile(
    memories,
    None,  # a new uuid4 key
    {"slot": "seat", "text": "Prefers aisle seats now"},
    slot="slot",
    policy=policy,
    provenance={"source": "chat", "thread": "thread-7"},
)
print(aisle.new, aisle.version, aisle.superseded)  # written 1 ('seat-1',)
```

`reconcile` reads the candidates inside its transaction and calls the policy with the new memory, as a `Proposal`, and the candidates. It applies the `Decision` the policy returns, then commits. There is no default policy, so `reconcile` never removes a memory your code did not decide to remove.

It returns a `ReconcileResult`. `key` is the new memory's key, and `new` says whether it was `"written"`, `"skipped"` or `"superseded"`. `version` is the version written, or None. `superseded`, `deleted`, `confirmed` and `kept` list the candidates by what the decision did with them.

`reconcile` also takes `slot`, `threshold`, `limit` and `filter`, which [Candidates](#candidates) describes, `provenance` and `valid_at`, which [Provenance and Valid Time](#provenance-and-valid-time) describes, and `index` and `ttl`, as `put` does. The value's index fields are embedded once, before the first attempt.

### Candidates

Candidates are live memories in the new memory's namespace, not in its descendants. A memory is a candidate for one of three reasons, strongest first:

| Reason | The memory |
|---|---|
| `"key"` | Is under the new memory's key, whatever `filter` says |
| `"slot"` | Holds the same value in the `slot` field, compared as JSON. This needs no index |
| `"similar"` | Has a vector that scores at or above `threshold` against the new memory's, on a store with an index |

The slot is a top-level field of the value that holds a string, number or boolean. Candidates come strongest reason first, then best score, then key. `limit` (default 10) caps the similar ones. Key and slot matches come on top of it, and `limit=0` keeps only them. `filter` narrows the slot and similar candidates by value, as in `search`. Expired memories are never candidates, and finding candidates never pushes back an expiry.

Each `Candidate` has the memory as `item`, a `SearchItem` with its `score`, along with its `reason`, `version`, `valid_at`, `provenance`, `confirmed_at` and `confirmations`. `key` and `score` are shortcuts. A key or slot match carries its best score when it has vectors, and None otherwise.

Scores come from the store's score function, as in `search`: 1 minus the distance on a cosine store, and the negated distance on an `l2` or `inner_product` store. On a cosine store `threshold` defaults to 0.8 and must lie between -1 and 1. On an `l2` or `inner_product` store no default fits, so pass one. The new memory is embedded as a document, with `embed_documents`, while `search` embeds its query with `embed_query`. Some models embed the two differently, so search scores are a poor guide here. Pick `threshold` from the scores that `find_conflicts` gives memories you know to be related.

In a namespace with many vectors, similar candidates come from the C-SPANN index instead of an exact scan, as [The Scan Limit](#the-scan-limit) explains.

### Policies

A policy is a function of the `Proposal` and the candidates that returns a `Decision`. `Policy` is its type. The built-in policies are explicit about what they supersede:

| Policy | What it decides |
|---|---|
| `supersede_same_slot()` | Skips the new memory if the memory under its key has a later valid time. Otherwise, if a slot candidate's valid time is later than the new memory's, records the new memory as superseded by the latest such candidate and keeps the rest. If none is later, supersedes every slot candidate |
| `skip_duplicates(threshold=0.97)` | Skips a new memory, one whose key holds no live memory, if a candidate has an equal value or scores at or above `threshold`, and confirms the best such candidate. With a `slot`, only candidates in the same slot count: a repeat outside the slot states another fact, so it lets the next policy supersede an older memory in the slot. Saving a live memory's value again under its key skips it and confirms that memory. Any other value under a live key is an update, which it never skips: it writes the new memory and keeps everything, so `first_of` tries the next policy |
| `supersede_similar(threshold=0.9)` | Works like `supersede_same_slot`, for the slot candidates and the candidates that score at or above `threshold`. Like it, it skips the new memory if the memory under its key has a later valid time, whatever that memory scores. It is blunt: similar is not contradictory |
| `keep_all()` | Writes the new memory and keeps every candidate |
| `first_of(*policies)` | Tries the policies in order and uses the first decision that does more than write the new memory and keep everything. A decision whose actions only `KEEP` candidates keeps everything too |

Confirming a duplicate instead of writing it again is how repeated observations reinforce a fact instead of piling up.

A `Decision` says what to do with the new memory and with each candidate:

| Field | Meaning |
|---|---|
| `new` | `"write"`, the default, writes the new memory. `"skip"` writes nothing. `"superseded"` records it as already superseded by a candidate whose fact is newer: with history on it goes straight into history, and without history it is skipped |
| `superseded_by` | With `"superseded"`, the key of that candidate. The decision must keep or confirm it |
| `actions` | Maps candidate keys to an `Action`, or to its value such as `"delete"`. A candidate left out is kept |

| Action | With history | Without history |
|---|---|---|
| `KEEP` | Leaves the memory as it is | The same |
| `SUPERSEDE` | Ends the memory as superseded by the new one, at the new memory's `valid_at` if one was given or else at the write, and keeps it in history | Deletes the memory and its vectors |
| `DELETE` | Ends the memory as deleted, at the write, and keeps it in history | Deletes the memory and its vectors |
| `CONFIRM` | Sets `confirmed_at`, counts the confirmation and renews the expiry from the memory's own TTL. The version and `updated_at` stay as they are | The same |

A decision that does not fit its candidates raises `ValueError` before anything is written, for example:

- an action for a key that is not a candidate;
- an action other than `KEEP` for the memory the new one overwrites;
- `SUPERSEDE` while skipping the new memory (use `DELETE`);
- a `superseded_by` that is not a kept or confirmed candidate.

Your own policy can use anything the candidates carry. This one weighs confidence from provenance:

```python
from langchain_cockroachdb.store import Action, Decision


def trust_confidence(proposal, candidates):
    """Supersede the memories in the same slot, unless one comes from a more confident source."""
    confidence = (proposal.provenance or {}).get("confidence", 0)
    same_slot = [c for c in candidates if c.reason == "slot"]
    if any((c.provenance or {}).get("confidence", 0) > confidence for c in same_slot):
        return Decision(new="skip")
    return Decision(actions={c.key: Action.SUPERSEDE for c in same_slot})


store.reconcile(
    memories,
    None,
    {"slot": "home_city", "text": "Lives in Porto"},
    slot="slot",
    policy=trust_confidence,
    provenance={"source": "chat", "confidence": 0.9},
)
```

A policy runs inside the transaction, and again on every retry, so it must be fast and deterministic. A slow one holds the transaction open and can run past the attempt timeout. A policy must never use the store, directly or through another thread, since the store holds its connection while the policy runs. A direct call raises `RuntimeError` at once, and a call handed to another thread can deadlock a store on a single connection. If the policy raises, the transaction rolls back and the exception propagates unchanged, without a retry.

Policies are plain functions on the async store too. `areconcile` refuses an `async def` policy with `ValueError`, before any I/O. A decision that needs async work, such as an LLM call, belongs in the two-step flow below. On the async store the policy runs on the event loop, which runs nothing else meanwhile. A policy that waits there for store work done through another thread, or through `asyncio.run_coroutine_threadsafe`, blocks the loop. Nothing can detect that, and the store's hang guard cannot fire while the loop is blocked. All of this holds for a `confirm` TTL function as well.

### The Two-Step Flow

An LLM call, or a person, is too slow to wait for inside a transaction. `find_conflicts` takes the arguments of `reconcile` except the policy. It reads the candidates without writing anything and returns a `Conflicts`: the new memory as a `Proposal`, its `candidates`, and the `target_version` and `target_updated_at` of the memory under the new key. Your code decides while no transaction is open. Then `resolve(conflicts, decision)` applies the decision in one transaction, as `reconcile` applies a policy's. It writes the memory that `find_conflicts` planned, with the embedding computed then, so the flow embeds once.

Here an LLM decides through structured output. Any chat model that supports `with_structured_output` works:

```python
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from langchain_cockroachdb.store import Conflicts, ConflictsChanged, Decision, ReconcileResult


class Verdict(BaseModel):
    key: str = Field(description="The key of an existing memory")
    action: Literal["keep", "supersede", "delete", "confirm"]


class Judgement(BaseModel):
    """What to do with a new memory and the existing memories it may conflict with."""

    new: Literal["write", "skip", "superseded"] = Field(
        description="Write the new memory, skip it as a repeat, or record it as superseded"
    )
    superseded_by: str | None = Field(
        default=None, description="With new='superseded', the key of the newer memory"
    )
    verdicts: list[Verdict] = Field(default_factory=list)


judge = init_chat_model("openai:gpt-4o-mini").with_structured_output(Judgement)

INSTRUCTIONS = """You keep a user's memories true. Given a new memory and existing ones,
supersede the memories the new one replaces, delete the ones that were wrong, and keep the rest.
If an existing memory says the same thing, skip the new one and confirm that memory.
If an existing memory is newer and contradicts the new one, record the new one as superseded."""


def describe(conflicts: Conflicts) -> str:
    """The new memory and its candidates, for the model to read."""
    new = conflicts.proposal
    lines = [f"New memory, true as of {new.valid_at:%Y-%m-%d}: {new.value['text']}"]
    lines += [
        f"Existing memory {c.key}, true as of {c.valid_at:%Y-%m-%d}: {c.item.value['text']}"
        for c in conflicts.candidates
    ]
    return "\n".join(lines)


def remember(namespace: tuple[str, ...], text: str, attempts: int = 3) -> ReconcileResult:
    for _ in range(attempts):
        conflicts = store.find_conflicts(namespace, None, {"text": text})
        decision = Decision()  # nothing to compare with: write it
        if conflicts.candidates:
            judgement = judge.invoke(
                [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": describe(conflicts)},
                ]
            )
            decision = Decision(
                new=judgement.new,
                superseded_by=judgement.superseded_by if judgement.new == "superseded" else None,
                actions={verdict.key: verdict.action for verdict in judgement.verdicts},
            )
        try:
            return store.resolve(conflicts, decision)
        except ConflictsChanged:
            pass  # a memory changed while the model decided: look again
    raise RuntimeError(f"The memories kept changing while deciding about {text!r}")


remember(memories, "No longer allergic to peanuts")
```

`Decision` takes the action names as strings and rejects an unknown `new`. A decision that does not fit the candidates, such as one that names a key that is not a candidate, makes `resolve` raise `ValueError` before any I/O, so you can ask the model again. With `AsyncCockroachDBStore`, call `afind_conflicts`, `await judge.ainvoke(...)` and `aresolve`.

In its transaction, before it writes, `resolve` checks that nothing the decision depends on has changed. The memory under the new key, and every candidate the decision supersedes, deletes, confirms or names in `superseded_by`, must still have the version and `updated_at` that `find_conflicts` read. A memory written, deleted or expired since counts as changed, and so does a live memory under the new key where there was none. A 0.5 store that writes the same table sets `updated_at` without bumping the version, so its writes count too. `resolve` then raises `ConflictsChanged`, whose `keys` name the memories that changed. It writes nothing and does not retry: run `find_conflicts` again and decide again. Candidates the decision keeps are not checked. A confirmation in between never counts, since confirming changes neither the version nor `updated_at`.

`resolve` raises `ValueError` before any I/O for:

- a `Conflicts` that did not come from `find_conflicts` on a store with the same table and index, such as one built by hand;
- a `Conflicts` whose proposal changed after `find_conflicts`, in its namespace, key, value, slot, slot value, provenance or valid time. `resolve` writes the memory planned then, so it refuses rather than drop the edit. A person who edits the memory in an approval step must call `find_conflicts` again with the edited memory;
- a `Conflicts` whose candidates were edited after `find_conflicts`, and a decision that does not fit them.

If a resolve of the same `Conflicts` already committed, as when an error left its outcome unknown and you call `resolve` again, the new call writes nothing. It returns the first result, rebuilt from the stored rows, with `already_applied=True`, whatever decision it was given. [Retries and Rebuilt Results](#retries-and-rebuilt-results) says what the rows can prove.

### Waiting for a Person

A `Conflicts` and a `Decision` can wait in graph state while a person decides, for example across a LangGraph `interrupt()`. The checkpointer's serializer must then rebuild the store's types. From langgraph-checkpoint 4.0.1, the serializer logs a warning for each type that is not on its msgpack allowlist, and refuses it when `LANGGRAPH_STRICT_MSGPACK=true` is set. A refused type comes back as a plain dict, so a `Conflicts` loses the plan that `resolve` needs, and `resolve` refuses it. `RECONCILE_SERDE_TYPES`, from `langchain_cockroachdb.store`, lists every class a reconcile value holds, in the form `allowed_msgpack_modules` takes. Give the checkpointer a serializer that allows them:

```python
from typing import TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from langchain_cockroachdb import CockroachDBSaver
from langchain_cockroachdb.store import (
    RECONCILE_SERDE_TYPES,
    Conflicts,
    Decision,
    ReconcileResult,
)


class Review(TypedDict, total=False):
    text: str
    conflicts: Conflicts
    result: ReconcileResult


def find(state: Review) -> Review:
    value = {"slot": "home_city", "text": state["text"]}
    return {"conflicts": store.find_conflicts(memories, None, value, slot="slot")}


def decide(state: Review) -> Review:
    decision = interrupt(state["conflicts"])  # resumes with the person's Decision
    return {"result": store.resolve(state["conflicts"], decision)}


builder = StateGraph(Review)
builder.add_node("find", find)
builder.add_node("decide", decide)
builder.add_edge(START, "find")
builder.add_edge("find", "decide")

serde = JsonPlusSerializer(allowed_msgpack_modules=RECONCILE_SERDE_TYPES)
with CockroachDBSaver.from_conn_string(DB_URI, serde=serde) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "review-1"}}
    graph.invoke({"text": "Lives in Lisbon now"}, config)  # stops at the interrupt

    # Later, perhaps in another process: a person looks at the candidates and decides
    conflicts = graph.get_state(config).values["conflicts"]
    decision = Decision(actions={c.key: "supersede" for c in conflicts.candidates})
    print(graph.invoke(Command(resume=decision), config)["result"])
```

The constant covers `Conflicts` with what it keeps for `resolve`, `Decision`, `ReconcileResult`, `ConditionalWrite`, `MemoryRecord` and `MemoryVersion`. With a list of types, the serializer refuses every other type that is not built in, so add your own state types to it, as in `[*RECONCILE_SERDE_TYPES, ("my_app.state", "Ticket")]`. Before langgraph-checkpoint 4.0.1, `JsonPlusSerializer` has no `allowed_msgpack_modules` argument, and passing it raises TypeError, so leave it out there. Those releases, back to 2.1.2, the oldest this package supports, rebuild every type.

### History

With history on, every write that ends a version keeps it, in the same transaction: a `put` that overwrites a memory (reason `"overwritten"`), a `delete` (`"deleted"`) and a supersede (`"superseded"`, with `superseded_by`). That includes plain `put` and `delete`, so LangMem-style tools, which write through them and keep no history of their own, get version history from the store.

`history(namespace, key, limit=20, offset=0)` returns the old versions of a memory, newest first, as `MemoryVersion`s. The live version is not one of them: `get` or `record` returns it.

```python
for version in store.history(memories, "seat-1"):
    print(version.version, version.reason, version.superseded_by, version.value["text"])
    print("  true from", version.valid_at, "to", version.invalid_at, "per", version.provenance)
    print("  stored from", version.written_at, "to", version.ended_at)
```

Each version has its `key`, `value` and `provenance`; `valid_at` and `invalid_at`, when its fact was true; `written_at` and `ended_at`, when the store held it; its `reason`; `superseded_by`; and `confirmations` and `confirmed_at`, the confirmations it had when it ended. A key's versions are numbered across its live row and its history, so a key written again after a delete carries on from its last version. `history` raises `ValueError` on a store without history.

Without a key, `history(namespace)` lists the old versions of every memory in the namespace, not in its descendants. They come newest first by `ended_at`, then by key, then highest version first, so pages are stable, and `limit` and `offset` page through them. This is how to find the old versions of memories that `reconcile` saved under keys the store generated. Each version names its `key` and `superseded_by`, so you can group versions by a field of their value, or follow `superseded_by` to the memory that replaced each one. Here is a timeline of what the store held for each slot:

```python
from collections import defaultdict

timeline = defaultdict(list)
for version in store.history(memories, limit=100):
    timeline[version.value.get("slot")].append(version)
for slot, versions in timeline.items():
    print(slot)
    for version in versions:
        print(" ", version.ended_at, version.reason, version.value["text"], version.superseded_by)
```

`history(namespace)` reads the namespace's whole history to sort it, which suits per-user namespaces like this one. For a namespace with a very large history, page with `limit` and `offset`: each page comes back small, though each call still reads the whole history.

Turn history on for every store that writes the same table, and only once every writer runs 0.6.0. A store with history off numbers a key it creates from 1, without looking at the key's history, and so does a 0.5 writer. If history already holds a version 1 of that key, the next archive of the new version 1 is dropped, since history keeps one row per version. A 0.5 writer also never archives, so the versions it overwrites or deletes are not kept.

`ttl_days` dates each version when it is archived. Versions archived without it never expire, and changing it does not re-date versions already archived. `history` leaves out versions past their TTL, even before CockroachDB's TTL job deletes them.

TTL expiry is not recorded in history: CockroachDB's TTL job deletes expired rows directly. For an audit trail that includes expiry, a [changefeed](https://www.cockroachlabs.com/docs/stable/change-data-capture-overview) on the items table captures those deletes.

### Forgetting

On a store with history, `delete` is a soft delete: the memory leaves the items table, and the version it ends stays in history. `purge` removes a memory for good. One statement deletes the item, its vectors and its history, and `purge` returns whether anything was deleted. `delete_namespace` deletes the history under the prefix too, with the items and vectors.

```python
store.put(memories, "passport", {"text": "Passport number X1234567"})
store.delete(memories, "passport")
print([v.reason for v in store.history(memories, "passport")])  # ['deleted']
store.purge(memories, "passport")  # True: the item, its vectors and its history
store.delete_namespace(("user", "bob"))  # bob's items, vectors and history
```

`purge` forgets one key. A memory that `reconcile` or `resolve` superseded keeps its own key in history, so forgetting a fact together with the versions it replaced means purging each of those keys. `history(namespace)` lists them, and a superseded version's `superseded_by` names the memory that replaced it. To forget everything a namespace holds, such as all of one user's memories, use `delete_namespace`.

`delete_namespace` also deletes history on a store created without history, when the history table exists, so a store that kept history before leaves nothing of the namespace behind. A role that did not run `setup()` needs SELECT and DELETE on that table for this, even with history off. `purge` raises `ValueError` on a store without history, where `delete` already removes a memory and its vectors. To purge one key's leftover history after turning history off, call `purge` on a store created with `history=True` for the same table.

### Provenance and Valid Time

Provenance is any JSON object that says where a memory came from, such as a source, a confidence or a thread. `reconcile`, `find_conflicts` and `put_if` take it as `provenance`, and the store keeps it in its own column. It does not affect embeddings, filters or `Item.value`. Policies see it on candidates, `history` returns it for each version, and `record` returns it with the live memory:

```python
record = store.record(memories, aisle.key)
print(record.version, record.provenance, record.valid_at, record.confirmations)
# 1 {'source': 'chat', 'thread': 'thread-7'} 2026-10-29 09:12:31.123456+00:00 0
```

A plain `put` writes no provenance and no valid time, so a put over a memory clears both. With history on, the version it ends keeps them.

A memory's valid time is when its fact became true, as opposed to when the store learned it. An agent that extracts memories from an old conversation passes the conversation's time as `valid_at`, a timezone-aware datetime. Without one, the valid time comes from the database's clock, never the client's: `now()`, read with the candidates, or the latest candidate's `updated_at` if that is later, so a correction never looks older than the fact it corrects. The memory then stores no valid time and is valid from its `updated_at`, like a memory from a plain `put`.

`supersede_same_slot` and `supersede_similar` compare valid times, so a fact that arrives late cannot replace a newer one. Here an agent reads a conversation from 2024, after the aisle preference was stored:

```python
from datetime import datetime, timezone

late = store.reconcile(
    memories,
    None,
    {"slot": "seat", "text": "Prefers window seats"},
    slot="slot",
    policy=supersede_same_slot(),
    valid_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
)
print(late.new)  # superseded
```

With history on, the late fact goes straight into history, superseded by the aisle preference. Its version there is true from May 2024 until the aisle preference became true. Without history it is skipped, and `new` is `"skipped"`.

Only these two policies compare valid times. A late fact under a key whose live memory is newer is skipped, not recorded, since a memory cannot be superseded by the memory under its own key.

### Confirmation-Aware Expiry

`confirm(namespace, key)` marks a live memory as confirmed: `confirmed_at` becomes now, `confirmations` goes up by one, and the memory's expiry, with its vectors', is renewed from its own TTL. It returns False when no live memory has the key. Confirming changes neither the version nor `updated_at`, so it never invalidates a pending `find_conflicts`, and it adds no history.

Confirmations belong to the fact, not the key. A write that changes a memory's value, through `put`, `reconcile`, `resolve` or `put_if`, starts `confirmations` and `confirmed_at` again from none, while saving the same value again keeps them. With history on, the version it ends keeps its count. A memory that expired counts as gone, so a write over it starts from none too.

`ttl` sets the expiry instead: a number of minutes, None for no expiry, or a function of the `MemoryRecord` that returns either. The function gets the memory as the confirm's transaction read it, before this confirmation counts:

```python
DAY = 60 * 24  # minutes


def longer_each_time(record):
    """30 days, doubled for each earlier confirmation, at most a year."""
    return min(30 * DAY * 2**record.confirmations, 365 * DAY)


store.confirm(memories, aisle.key, ttl=longer_each_time)  # True
```

A TTL function runs inside the transaction, like a policy, so the rules for policies apply to it.

Retrieval is not confirmation. `record` and `find_conflicts` never push back an expiry, while `get` and `search` do when `refresh_on_read` is on, as it is by default. For expiry that follows confirmations rather than reads, build the store with `refresh_on_read` set to `False`, as in [Turning It On](#turning-it-on), or pass `refresh_ttl=False` to reads. `skip_duplicates` confirms the memory that a new one repeats, so a fact the user keeps stating stays.

### Concurrent Reconciles

The candidate read and the writes of a reconcile share one serializable transaction. If another writer changes what a policy saw, one of the two transactions retries, and its policy sees the other's write. So two agents that reconcile contradictory memories in one namespace, with a policy that supersedes, end with exactly one live fact. For similar candidates this holds up to the scan limit, as [The Scan Limit](#the-scan-limit) explains next.

### The Scan Limit

`reconcile_scan_limit` is a store setting, 2000 by default, passed to the constructor or to `from_conn_string`. A namespace with at most that many vectors has every one of them ranked exactly, read by primary key. Serializable isolation covers that whole read, so a concurrent insert into the namespace makes one of the two transactions retry.

The slot read is not bounded by the scan limit. With a `slot`, the store finds the slot matches by scanning the namespace's live items, since no index covers slot values, so that read grows with the namespace whatever the limit says.

A namespace with more vectors takes its similar candidates from a window of the C-SPANN index, as a search does. A concurrent insert into a partition that the index search did not read is then seen only by the next reconcile. The window makes room for the vectors of the key and slot matches, which are candidates already. A selective `filter`, or expired memories, can still fill it and hide a match beyond it. When the window shows that this may have happened, the store ranks every vector of the namespace in one more round trip.

### Conditional Writes

Two threads that read the same memory and both write it back with `put` lose one of the updates without a word. `put_if` writes only if the live memory under the key still has the version and `updated_at` of `expected`, the `MemoryRecord` you read with `record`. On a mismatch it writes nothing and returns `applied=False` with `current`, the live memory as its transaction read it, or None if there is none. Merge with it, and write again:

```python
record = store.record(memories, "airlines")
while True:
    names = record.item.value["airlines"] if record else []
    airlines = sorted({*names, "KLM"})
    result = store.put_if(
        memories,
        "airlines",
        {"text": "Flies with " + ", ".join(airlines), "airlines": airlines},
        expected=record,
    )
    if result.applied:
        break
    record = result.current  # another writer got there first: merge with its memory
```

A conflict is an expected outcome here, so it is returned, not raised. `expected=None` writes only if no live memory has the key, so a memory is created once, as in the first pass of this loop. An expired memory counts as absent. On a match `put_if` writes as `put` does, in the transaction that checked, with `provenance` and `valid_at` if given, and with history on it keeps the version it replaces. The result is a `ConditionalWrite` with `applied`, `key`, `version` (the version written, or None), `current` and `already_applied`.

`delete_if(namespace, key, expected=record)` deletes the same way. Its `applied=True` means the memory matched `expected` and this call deleted it, or, after an attempt of the call that may have committed, that the memory is gone and was this call's to delete.

Both are safe to retry, and a retry never reports a false conflict. If an attempt commits and its result is lost, the next attempt reports `applied=True`. `put_if` finds its own write id on the memory, which a confirmation in between leaves in place, and sets `already_applied`. `delete_if` finds the memory gone, and sets `already_applied` when its own history row proves the delete, which needs history on. If another writer changed the memory between the lost commit and the retry, the retry reports a conflict, with that writer's memory, or None, as `current`: that is the current state. Both are single-key calls, outside `batch`, since LangGraph's batch operations cannot carry a precondition.

### Retries and Rebuilt Results

Every method here retries transient errors under the store's `RetryPolicy`, like every other store call. Each `reconcile` and `resolve` picks a write id before its first attempt and marks the rows it writes with it, and the memories it confirms with it as their confirmation id. `confirm` marks a memory with a confirmation id of its own and leaves its write id alone, so a confirmation that lands between a lost commit and the retry of a write does not disturb the retry. An attempt can commit and still lose its result on the way back, as when the connection drops while the COMMIT's answer travels. The retry then finds the rows with its write id, writes nothing, and returns the result rebuilt from those rows, with `already_applied=True`. A rebuilt result lists only what the stored rows prove:

- `kept` is empty, and on a store without history so are `superseded` and `deleted`, since such a store keeps no trace of what it removed.
- A written memory that another writer replaced before the retry gives `new="written"` with `version=None`, when a history row proves the write: the version it overwrote, or a memory it superseded.
- A write to a new key that superseded, deleted and confirmed nothing, and that another writer replaced between the lost commit and the retry, leaves no trace, so the retry runs it again. Otherwise the rebuilt result says `new="skipped"` with the keys it confirmed or deleted, although the memory was written and then replaced.
- Without history, a decision that skipped the new memory or recorded it as superseded, and otherwise only deleted or superseded candidates, leaves no row either, and runs again after a lost commit.
- A `confirm` call, or a confirm-only `reconcile` or `resolve`, whose commit was lost can count twice if another confirmation of the same memory lands before the retry.
- A create-only `put_if(expected=None)` whose commit was lost can create the memory again if another writer deleted it before the retry.
- On a store without history, calling `resolve` again after an error escaped it, as when its retries ran out, can raise `ConflictsChanged` for the keys its own earlier call removed. `find_conflicts` then shows the current state.

### Tools for an Agent

With `create_agent`, a `remember` tool can reconcile what the model wants to save, under a topic the model names. Recall stays a plain search, since superseded memories are no longer live. `remember` lets the store generate the keys, so a `past_memories` tool reads the old versions with `history(namespace)` and keeps those about the topic:

```python
from langchain.tools import ToolRuntime, tool

POLICY = first_of(skip_duplicates(), supersede_same_slot())


@tool
def remember(text: str, topic: str, runtime: ToolRuntime[Context, dict]) -> str:
    """Save a fact about the user. topic names what it is about, such as "seat" or "home_city"."""
    assert isinstance(runtime.store, CockroachDBStore)
    namespace = ("user", runtime.context.user_id, "memories")
    result = runtime.store.reconcile(
        namespace,
        None,
        {"topic": topic, "text": text},
        slot="topic",
        policy=POLICY,
        provenance={"source": "chat", "thread": runtime.config["configurable"]["thread_id"]},
    )
    if result.new != "written":
        return f"Already known: {text}"
    return f"Saved: {text}" + (f", replacing {len(result.superseded)}" if result.superseded else "")


@tool
def recall(query: str, runtime: ToolRuntime[Context, dict]) -> str:
    """Search the user's memories."""
    assert runtime.store is not None
    namespace = ("user", runtime.context.user_id, "memories")
    items = runtime.store.search(namespace, query=query, limit=3)
    return "\n".join(item.value["text"] for item in items) or "No memories found."


@tool
def past_memories(topic: str, runtime: ToolRuntime[Context, dict]) -> str:
    """List what the user used to say about a topic, newest first."""
    assert isinstance(runtime.store, CockroachDBStore)
    namespace = ("user", runtime.context.user_id, "memories")
    lines = [
        f"{version.ended_at:%Y-%m-%d}: {version.value['text']} ({version.reason})"
        for version in runtime.store.history(namespace, limit=100)
        if version.value.get("topic") == topic
    ]
    return "\n".join(lines) or f"No earlier memories about {topic}."
```

`Context` is the one from [With create_agent](#with-create_agent). `past_memories` needs a store with history on. With `AsyncCockroachDBStore`, write the tools as async functions that call `areconcile`, `asearch` and `ahistory`. Tools that know only `BaseStore` write through plain `put` and `delete`, and still get history on a store with history on.

### Upgrading From 0.5

`setup()` applies migration 2, which adds the columns these methods use to the items table in one statement: `version`, `provenance`, `confirmed_at`, `confirmations`, `write_id`, `valid_at` and `confirm_id`. Until it runs, the store works as 0.5 did, and the new methods raise `ValueError` asking for `setup()`. With history off and these methods unused, the store behaves as in 0.5 apart from the new columns. During a rolling upgrade, 0.5 writers do not bump the version but do set `updated_at`, which is why `resolve`, `put_if` and `delete_if` check both.

On a large table, run `setup()` once from a single process before rolling out 0.6.0. The backfill of the new columns takes longer the more items the table holds, and the processes that start meanwhile then find migration 2 in place.

Several processes can still run `setup()` at once, as when pods restart into 0.6.0 together. One of them backfills the new columns. CockroachDB answers the same statement from the others at once, before the columns exist, so each of them waits until the columns are public and SHOW JOBS shows no schema change job on the table, for up to `SCHEMA_CHANGE_SETTLE_SECONDS` (10 minutes), before it records migration 2. A store that finds migration 2 recorded can then write the new columns right away. Two cases leave the columns alone to decide: SHOW JOBS shows a role without admin only its own jobs, and it can cut a long qualified table name from a job's description. If no job applies the migration, as when a peer's migration failed, the waiting `setup()` applies it itself and logs that at INFO. The wait logs as described in [Watching retries](resilience.md#watching-retries). If it runs out, `setup()` raises `TimeoutError` saying what remained, and running it again waits again. This wait comes on top of the TTL steps, each of which can wait out another schema change on its table for up to ten minutes.

0.5 writers know nothing of history or confirmations. They never archive what they overwrite or delete. One that re-creates a deleted key starts it again at version 1, which history then cannot keep, since it already holds a version 1 of that key. So turn history on only once every writer runs 0.6.0. A 0.5 write that changes a value also leaves its confirmations as they were.

A 0.5 writer leaves a row's write id alone too. If a plain `put` commits and its result is lost, and a 0.5 writer writes the same key before the put's retry, the retry finds its own write id on the row and leaves the 0.5 value, but writes its own vectors over the 0.5 writer's. That needs both the rolling-upgrade window and a lost commit. The next write of the key replaces both.

An application role that did not run `setup()` needs SELECT, INSERT, UPDATE and DELETE on the items and vectors tables, and on the history table when history is on:

```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE store, store_vectors TO app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE store_history TO app;  -- with history
GRANT SELECT ON TABLE store_migrations TO app;
```

With history off, the role still needs SELECT and DELETE on a history table that exists, left from when history was on: `delete_namespace` deletes the namespace's rows there, and without those grants it fails as a whole rather than forget only part of the namespace.

The store reads the migrations table to learn whether migration 2 is in place. Without SELECT on it, or without a migrations table, as when the tables were created without `setup()`, the store looks for migration 2's columns on the items table instead. Those can show up a moment before every node uses them, so on a multi-node cluster one write can fail in the seconds after `setup()` applies migration 2. Run `setup()` before rolling out 0.6.0, or grant SELECT on the migrations table, to avoid that. Without that grant, the first store call that runs inside a transaction the caller opened also gets the permission error, once per store. Only the grant avoids it.

## Differences from PostgresStore

1. **Expiry on the server.** CockroachDB's row-level TTL job deletes expired items, so no sweeper thread is needed. Expired items are left out of reads, searches and listings by default (`omit_expired` is on).
2. **Atomic batches, retries and concurrency.** All puts and deletes of one batch commit in one transaction, every call retries transient errors under the store's `RetryPolicy`, and a `CockroachDBStore` on a pool runs batches concurrently.
3. **No stale vectors.** Putting an item again replaces all its vectors, and deleting an item deletes them.
4. **Filters.** A comparison with a number matches numeric values only, where PostgresStore compares text and `InMemoryStore` converts values with `float()`. Nested dictionaries match nested keys, and `$ne` also matches items without the key.
5. **Deterministic paging.** Results with the same score, or the same `updated_at`, are ordered by namespace and key, so paging with `offset` is stable.
6. **Configurable tables.** `table_name` and `schema` set the table names, so one database can hold several stores.
7. **`delete_namespace`.** `delete_namespace(namespace_prefix)` and `adelete_namespace` delete a namespace and its descendants in one transaction and return the number of items deleted.
8. **Warnings and checks.** A `query` on a store without `index` logs a WARNING. Changing the index configuration after `setup()` raises an error or logs a warning, as described in [Changing the Index Configuration](#changing-the-index-configuration).
9. **`disable_ttl` and `drop_vector_index`.** `disable_ttl()` and `adisable_ttl()` remove the table TTL setting, and `drop_vector_index()` and `adrop_vector_index()` drop the vector index so that `setup()` can build it again, for example with another distance.
10. **No `deserializer`.** psycopg decodes JSONB values before the store sees them, so the store does not take PostgresStore's `deserializer=` argument. To transform values as they are read, wrap `get` and `search`.
11. **Keeping memories true.** `reconcile`, `find_conflicts`, `resolve`, `record`, `confirm`, `put_if`, `delete_if`, `history` and `purge`, and their async forms, as described in [Keeping Memories True](#keeping-memories-true). They are additional methods: `BaseStore` and `Item` are unchanged.

## Next Steps

- [Long-Term Memory Example](../examples/long-term-memory.md) - The runnable example, step by step
- [Memory Reconciliation Example](../examples/memory-reconciliation.md) - Reconcile, history, confirm and conditional writes, step by step
- [API Reference](../api/store.md) - Every class, method and setting
- [Checkpointer Guide](checkpointer.md) - Short-term memory for LangGraph threads
- [Retries and Resilience](resilience.md) - Retry settings and hang guards
