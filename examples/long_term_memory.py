"""Long-term memory for a LangGraph agent, kept in CockroachDB.

It uses create_agent from the langchain package, which also installs langgraph:

    pip install langchain-cockroachdb langchain

Run it against a local cluster:

    COCKROACHDB_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable" \\
        python examples/long_term_memory.py

The agent saves a memory in one conversation thread and finds it again by
semantic search in another. The store holds memories across threads; the
checkpointer holds each thread's messages. The chat model is scripted and
the embeddings come from DeterministicFakeEmbedding, so no API key or
network is needed. That embedding gives each text a fixed random vector, so
it does not rank by meaning; a real embedding model does, with the same code.

The store uses its own tables (memory_example, memory_example_vectors and
memory_example_migrations) and drops them at the end. The checkpointer uses
the default checkpoint tables, as examples/checkpointer.py does, and the
example deletes its threads from them.
"""

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb import (
    AsyncCockroachDBSaver,
    AsyncCockroachDBStore,
    CockroachDBSaver,
    CockroachDBStore,
)

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)
# Raw psycopg needs a postgresql:// URL; cockroachdb:// is the same server
LIBPQ_URL = "postgresql://" + CONNECTION_STRING.split("://", 1)[1]
TABLE = "memory_example"
RUN = uuid.uuid4().hex[:8]

# Embed each memory's "text" field. Use a real model, such as OpenAIEmbeddings,
# with its own dims, for search by meaning.
INDEX = {"dims": 64, "embed": DeterministicFakeEmbedding(size=64), "fields": ["text"]}
# default_ttl is in minutes: items expire 30 days after they were written or last read
TTL = {"default_ttl": 60 * 24 * 30, "refresh_on_read": True}

REMEMBER = "remember that "


@dataclass
class Context:
    """Who the agent is talking to, passed with each call."""

    user_id: str


class ScriptedModel(BaseChatModel):
    """Stands in for an LLM: saves "Remember that ..." and searches for anything else."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            text = str(last.content)
            if text.lower().startswith(REMEMBER):
                name, args = "save_memory", {"text": text[len(REMEMBER) :]}
            else:
                name, args = "recall", {"query": text}
            call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}
            message = AIMessage(content="", tool_calls=[call])
        else:
            # Answer with what the tool returned
            message = AIMessage(content=str(last.content))
        return ChatResult(generations=[ChatGeneration(message=message)])


def memories(runtime: ToolRuntime[Context]) -> tuple[str, ...]:
    """The namespace of the current user's memories."""
    return ("user", runtime.context.user_id, "memories")


@tool
def save_memory(text: str, runtime: ToolRuntime[Context]) -> str:
    """Save a fact about the user for later conversations."""
    assert runtime.store is not None
    runtime.store.put(memories(runtime), str(uuid.uuid4()), {"text": text})
    return f"Saved to {memories(runtime)}: {text}"


@tool
def recall(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search the user's memories."""
    assert runtime.store is not None
    found = runtime.store.search(memories(runtime), query=query, limit=1)
    return f"From memory: {found[0].value['text']}" if found else "I have no memories of you."


# The async agent awaits async tools, which call the store's a-prefixed methods
@tool("save_memory")
async def asave_memory(text: str, runtime: ToolRuntime[Context]) -> str:
    """Save a fact about the user for later conversations."""
    assert runtime.store is not None
    await runtime.store.aput(memories(runtime), str(uuid.uuid4()), {"text": text})
    return f"Saved to {memories(runtime)}: {text}"


@tool("recall")
async def arecall(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search the user's memories."""
    assert runtime.store is not None
    found = await runtime.store.asearch(memories(runtime), query=query, limit=1)
    return f"From memory: {found[0].value['text']}" if found else "I have no memories of you."


def show(thread: int, text: str, result: dict[str, Any]) -> None:
    """Print one turn of a conversation."""
    print(f"   thread {thread}, user:  {text}")
    print(f"   thread {thread}, agent: {result['messages'][-1].content}")


def sync_example() -> None:
    """The store and the checkpointer on one connection pool."""
    print("1. CockroachDBStore and CockroachDBSaver on one connection pool\n")

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
        print("   setup() created the store tables, a C-SPANN vector index and row-level TTL")

        # The application writes some items itself. This one is never embedded
        # (index=False) and never expires (ttl=None).
        store.put(("user", "alice", "profile"), "name", {"name": "Alice"}, index=False, ttl=None)
        store.put(("user", "bob", "memories"), "seat", {"text": "I prefer aisle seats."})
        print(f"   alice's profile: {store.get(('user', 'alice', 'profile'), 'name').value}")

        agent = create_agent(
            ScriptedModel(),
            tools=[save_memory, recall],
            context_schema=Context,
            checkpointer=checkpointer,
            store=store,
        )
        alice = Context(user_id="alice")
        # Each message starts a new conversation thread
        turns = {
            f"memory-example-{RUN}-1": "Remember that I prefer window seats.",
            f"memory-example-{RUN}-2": "Where do I like to sit on a plane?",
        }
        try:
            for number, (thread, text) in enumerate(turns.items(), start=1):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": text}]},
                    {"configurable": {"thread_id": thread}},
                    context=alice,
                )
                show(number, text, result)
        finally:
            for thread in turns:
                checkpointer.delete_thread(thread)

        print(f"   alice's namespaces: {store.list_namespaces(prefix=('user', 'alice'))}")
        removed = store.delete_namespace(("user", "alice"))
        print(f"   delete_namespace(('user', 'alice')) removed {removed} items")
        print(f"   namespaces left: {store.list_namespaces(prefix=('user',))}")

    print("   Done\n")


async def async_example() -> None:
    """The async store from a connection string, with its own pool."""
    print("2. AsyncCockroachDBStore from a connection string, with pool_config\n")

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
        # Safe to run again: setup() and every migration are idempotent
        await store.setup()
        await checkpointer.setup()

        agent = create_agent(
            ScriptedModel(),
            tools=[asave_memory, arecall],
            context_schema=Context,
            checkpointer=checkpointer,
            store=store,
        )
        alice = Context(user_id="alice")
        turns = {
            f"memory-example-{RUN}-3": "Remember that my favorite city is Lisbon.",
            f"memory-example-{RUN}-4": "Which city do I like best?",
        }
        try:
            for number, (thread, text) in enumerate(turns.items(), start=1):
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": text}]},
                    {"configurable": {"thread_id": thread}},
                    context=alice,
                )
                show(number, text, result)
        finally:
            for thread in turns:
                await checkpointer.adelete_thread(thread)

        print(f"   namespaces: {await store.alist_namespaces(prefix=('user',))}")
        removed = await store.adelete_namespace(("user",))
        print(f"   adelete_namespace(('user',)) removed {removed} items, every user's")

    print("   Done\n")


def drop_tables() -> None:
    """Drop the store's tables, so the next run starts clean."""
    with psycopg.connect(LIBPQ_URL, autocommit=True) as conn:
        for suffix in ("_vectors", "", "_migrations"):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(f"{TABLE}{suffix}"))
            )


def main() -> None:
    """Run both halves, then drop the tables, even after a failure."""
    print("Long-term memory for LangGraph agents with CockroachDB\n")
    print("=" * 50 + "\n")

    try:
        sync_example()
        asyncio.run(async_example())
    finally:
        drop_tables()
        print(f"Dropped {TABLE}, {TABLE}_vectors and {TABLE}_migrations\n")

    print("=" * 50)
    print("\nAll examples complete!")


if __name__ == "__main__":
    main()
