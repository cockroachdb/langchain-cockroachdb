# LangGraph Long-Term Memory

Keep memories that outlive a conversation thread in CockroachDB: user preferences, facts an agent learned, profiles.

## Overview

A LangGraph checkpointer keeps the state of one thread, which is short-term memory. A store keeps what every thread can read, which is long-term memory. `CockroachDBStore` and `AsyncCockroachDBStore` implement LangGraph's `BaseStore`, so they work wherever LangGraph takes a store: `create_agent(store=...)`, `StateGraph.compile(store=...)` and Deep Agents.

The API follows `PostgresStore` from [langgraph-checkpoint-postgres](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres): `from_conn_string`, `setup()`, the `index` and `ttl` arguments and the TTL sweeper methods work the same way. What this store adds or does differently is listed in [Differences from PostgresStore](#differences-from-postgresstore).

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
| `retry_policy` | `RetryPolicy()` | How operations retry |
| `connect_timeout` | `10.0` | Seconds to wait when connecting, or `None` |
| `prepare_threshold` | `5` | psycopg prepared statement threshold, or `None` to disable |
| `connect_kwargs` | `None` | Extra psycopg connection options, for example `{"application_name": "my-service"}` |
| `table_name` | `"store"` | Name of the items table |
| `schema` | `"public"` | Schema of the store's tables |

### Your Own Pool or Connection

The constructor takes a psycopg connection or a `psycopg_pool` pool, plus `pipe`, `index`, `ttl`, `retry_policy`, `table_name` and `schema`. The connection or pool must use autocommit and `dict_row`. One pool can serve the store and the checkpointer:

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

Every call retries transient CockroachDB errors under the store's `RetryPolicy`. Pass `retry_policy=` to change it; see [Retries and Resilience](resilience.md) for the fields. Every write is an UPSERT, a DELETE, or an UPDATE that pushes back an expiry. Each is safe to run twice, so a retry after an ambiguous result is safe.

A `batch()` call validates every op and serializes every value before any statement runs. It then runs its reads, one round trip each for its gets, its searches and its listings, and commits all its puts and deletes together in one transaction. Semantic searches under a namespace add a round trip before the searches, and sometimes one after them, as [How Searches Use the Index](#how-searches-use-the-index) explains. Embeddings are computed once, before the first attempt, so a retry does not call the embedding model again. When one batch puts the same key twice, the last put wins.

On CockroachDB v25.4, a TTL change can meet a schema change that another process runs on the same table. The TTL step of `setup()`, `disable_ttl()` and `adisable_ttl()` retry that error, as described in [Configuring retries](resilience.md#configuring-retries).

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
| `beam_size` | `None` | C-SPANN beam width. A larger beam finds more of the true nearest neighbors and reads more of the index |

A put can override `fields`: `index=False` stores the item without vectors, and `index=["text", "summary"]` embeds those paths instead.

```python
store.put(("user", "alice", "profile"), "name", {"name": "Alice"}, index=False)
```

Each embedded text is one vector row, so an item can have several. Putting an item again replaces all its vectors, and deleting it deletes them. A search returns each item once, with the score of its best match. Higher scores are better: the score is `1 - distance` for cosine, and the negated distance for `l2` and `inner_product`. Results are ordered by score, then by namespace and key.

### How Searches Use the Index

The vector index has the namespace as a prefix column, because most memory searches name one user's namespace. A search with a `query` under a namespace prefix runs in two steps. First a short read lists the namespaces at or under the prefix that have vectors. Then the ranked search runs on the index, limited to those namespaces. For one user's namespace that list has one entry, so the search reads only that namespace's part of the index.

The index finds approximate nearest neighbors, which `beam_size` tunes. It gives the search a window of candidate vectors, twice as many as the page needs, and the filter and the expiry check then apply to those candidates. A selective filter, or many expired items that the TTL job has not deleted yet, can leave fewer matches in the window than the page asks for. When that happens and the window was full, or nothing matched, the search runs a second, exact query over the namespaces the first step found: it applies the filter and the expiry check first, ranks every match and returns that page instead. A window with fewer candidates than its size held every vector of those namespaces, so its result is already exact and no second query runs. For one user's memories the second query reads little, but it costs a round trip.

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

CockroachDB deletes expired rows itself, with [row-level TTL](https://www.cockroachlabs.com/docs/stable/row-level-ttl). With a `ttl` configuration, `setup()` sets `ttl_expiration_expression` to the `expires_at` column on the items table, and on the vectors table when there is an index. The job runs on the cron that `sweep_interval_minutes` gives, or `ttl_job_cron`, and hourly (`@hourly`) by default. A changed `sweep_interval_minutes` or `ttl_job_cron` takes effect at the next `setup()`. Rows without an expiry are never deleted. Vector rows carry their item's expiry, so they expire with it.

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

`delete_namespace` deletes every item in a namespace and its descendants, with their vectors, and returns the number of items deleted. `("user", "alice")` covers `("user", "alice", "memories")` but not `("user", "alice2")`. It runs as one statement, so it commits completely or not at all. If a retry follows a commit whose result was lost, the count is that of the retry, usually 0. An empty prefix raises `InvalidNamespaceError`. `BaseStore` has no such method; it is specific to this store.

## Tables

| Table | Holds |
|---|---|
| `store` | Items: `prefix` (the namespace), `key`, `value` (JSONB), `created_at`, `updated_at`, `expires_at` and `ttl_minutes` |
| `store_vectors` | One row per embedded text, with the item's expiry. Created only with `index` |
| `store_migrations` | The migration versions applied |

`table_name="memories"` names the tables `memories`, `memories_vectors` and `memories_migrations`, so several stores can share a database. `schema=` puts them in another schema, which must exist. The default schema is `public`.

The store creates plain tables, with no multi-region locality such as `REGIONAL BY ROW` or `GLOBAL`. You can set one yourself with `ALTER TABLE`; `setup()` leaves it as you set it.

## Use with LangGraph

### With create_agent

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
def save_memory(text: str, runtime: ToolRuntime[Context]) -> str:
    """Save a fact about the user for later conversations."""
    assert runtime.store is not None
    namespace = ("user", runtime.context.user_id, "memories")
    runtime.store.put(namespace, str(uuid.uuid4()), {"text": text})
    return f"Saved: {text}"


@tool
def recall(query: str, runtime: ToolRuntime[Context]) -> str:
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

## Next Steps

- [Long-Term Memory Example](../examples/long-term-memory.md) - The runnable example, step by step
- [API Reference](../api/store.md) - Every class, method and setting
- [Checkpointer Guide](checkpointer.md) - Short-term memory for LangGraph threads
- [Retries and Resilience](resilience.md) - Retry settings and hang guards
