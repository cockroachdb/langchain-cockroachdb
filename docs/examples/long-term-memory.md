# Long-Term Memory Example

`examples/long_term_memory.py` gives a LangGraph agent long-term memory in CockroachDB. The agent saves a memory in one conversation thread and finds it again, by semantic search, in another thread.

## Running the Example

```bash
# Install the package, and langchain for create_agent (it also installs langgraph)
pip install langchain-cockroachdb langchain

# Start CockroachDB
docker-compose up -d

# Run the long-term memory example
python examples/long_term_memory.py
```

It needs no API key. The chat model is scripted, and the embeddings come from `DeterministicFakeEmbedding`, which gives each text a fixed random vector. That does not rank by meaning, so the example keeps one memory per user at a time. A real embedding model ranks by meaning with the same code.

The store uses its own tables (`memory_example`, `memory_example_vectors` and `memory_example_migrations`) and drops them at the end, even after a failure. The checkpointer uses the default checkpoint tables, and the example deletes its threads from them.

## The Store and the Checkpointer on One Pool

The store gets semantic search (`index`) and expiry (`ttl`). The checkpointer keeps each thread's messages, and shares the pool:

```python
INDEX = {"dims": 64, "embed": DeterministicFakeEmbedding(size=64), "fields": ["text"]}
# default_ttl is in minutes: items expire 30 days after they were written or last read
TTL = {"default_ttl": 60 * 24 * 30, "refresh_on_read": True}

with ConnectionPool(
    LIBPQ_URL,
    min_size=1,
    max_size=4,
    kwargs={"autocommit": True, "row_factory": dict_row},
    open=True,
) as pool:
    store = CockroachDBStore(pool, index=INDEX, ttl=TTL, table_name=TABLE)
    store.setup()
    checkpointer = CockroachDBSaver(pool)
    checkpointer.setup()
```

The application can also write items itself. The profile is never embedded and never expires. Bob's memory shows that searches and deletes stay inside one user's namespace:

```python
store.put(("user", "alice", "profile"), "name", {"name": "Alice"}, index=False, ttl=None)
store.put(("user", "bob", "memories"), "seat", {"text": "I prefer aisle seats."})
```

## Tools That Use the Store

Tools reach the store through `ToolRuntime`, and the user id comes from the run's context:

```python
@dataclass
class Context:
    """Who the agent is talking to, passed with each call."""

    user_id: str


def memories(runtime: ToolRuntime) -> tuple[str, ...]:
    """The namespace of the current user's memories."""
    return ("user", runtime.context.user_id, "memories")


@tool
def save_memory(text: str, runtime: ToolRuntime) -> str:
    """Save a fact about the user for later conversations."""
    assert runtime.store is not None
    runtime.store.put(memories(runtime), str(uuid.uuid4()), {"text": text})
    return f"Saved to {memories(runtime)}: {text}"


@tool
def recall(query: str, runtime: ToolRuntime) -> str:
    """Search the user's memories."""
    assert runtime.store is not None
    found = runtime.store.search(memories(runtime), query=query, limit=1)
    return f"From memory: {found[0].value['text']}" if found else "I have no memories of you."
```

## Two Threads, One Memory

Each message starts a new thread, so the second thread has none of the first one's messages. The answer comes from the store:

```python
agent = create_agent(
    ScriptedModel(),
    tools=[save_memory, recall],
    context_schema=Context,
    checkpointer=checkpointer,
    store=store,
)
alice = Context(user_id="alice")
for thread, text in [
    ("thread-1", "Remember that I prefer window seats."),
    ("thread-2", "Where do I like to sit on a plane?"),
]:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread}},
        context=alice,
    )
    print(result["messages"][-1].content)
```

## Namespaces and Deleting a User

```python
store.list_namespaces(prefix=("user", "alice"))
# [('user', 'alice', 'memories'), ('user', 'alice', 'profile')]

store.delete_namespace(("user", "alice"))  # 2, the memory and the profile
store.list_namespaces(prefix=("user",))
# [('user', 'bob', 'memories')]
```

## The Async Version

The second half builds `AsyncCockroachDBStore` with `from_conn_string` and `pool_config`, and its tools await the store's `aput` and `asearch`:

```python
async with (
    AsyncCockroachDBStore.from_conn_string(
        CONNECTION_STRING,
        pool_config={"min_size": 1, "max_size": 4},
        index=INDEX,
        ttl=TTL,
        table_name=TABLE,
    ) as store,
    AsyncCockroachDBSaver.from_conn_string(CONNECTION_STRING) as checkpointer,
):
    await store.setup()
    await checkpointer.setup()
    ...
    removed = await store.adelete_namespace(("user",))  # every user's items
```

## Output

```text
1. CockroachDBStore and CockroachDBSaver on one connection pool

   setup() created the store tables, a C-SPANN vector index and row-level TTL
   alice's profile: {'name': 'Alice'}
   thread 1, user:  Remember that I prefer window seats.
   thread 1, agent: Saved to ('user', 'alice', 'memories'): I prefer window seats.
   thread 2, user:  Where do I like to sit on a plane?
   thread 2, agent: From memory: I prefer window seats.
   alice's namespaces: [('user', 'alice', 'memories'), ('user', 'alice', 'profile')]
   delete_namespace(('user', 'alice')) removed 2 items
   namespaces left: [('user', 'bob', 'memories')]
   Done

2. AsyncCockroachDBStore from a connection string, with pool_config

   thread 1, user:  Remember that my favorite city is Lisbon.
   thread 1, agent: Saved to ('user', 'alice', 'memories'): my favorite city is Lisbon.
   thread 2, user:  Which city do I like best?
   thread 2, agent: From memory: my favorite city is Lisbon.
   namespaces: [('user', 'alice', 'memories'), ('user', 'bob', 'memories')]
   adelete_namespace(('user',)) removed 2 items, every user's
   Done
```

## Next Steps

- [Long-Term Memory Guide](../guides/long-term-memory.md) - Semantic search, TTL, pools and the differences from PostgresStore
- [API Reference](../api/store.md) - Full API documentation
- [Checkpointer Example](checkpointer.md) - Short-term memory for LangGraph threads
