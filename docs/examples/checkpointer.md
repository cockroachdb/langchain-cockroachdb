# Checkpointer Examples

Working examples for the LangGraph checkpointer with CockroachDB.

## Running the Example

```bash
# Start CockroachDB
docker-compose up -d

# Run the checkpointer example
python examples/checkpointer.py
```

## Sync Checkpointer

Basic synchronous usage with `CockroachDBSaver`:

```python
from langchain_cockroachdb import CockroachDBSaver

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

with CockroachDBSaver.from_conn_string(DB_URI) as saver:
    saver.setup()

    # Use with LangGraph
    graph = workflow.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "user-123"}}
    result = graph.invoke({"messages": [("user", "Hello!")]}, config)
```

## Async Checkpointer

High-concurrency usage with `AsyncCockroachDBSaver`:

```python
from langchain_cockroachdb import AsyncCockroachDBSaver

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

async with AsyncCockroachDBSaver.from_conn_string(DB_URI) as saver:
    await saver.setup()

    graph = workflow.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "user-456"}}
    result = await graph.ainvoke({"messages": [("user", "Hello!")]}, config)
```

## Connection Pool Mode

Production-ready setup with connection pooling:

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

## Chatbot with Memory

Complete example of a LangGraph chatbot with persistent memory:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_cockroachdb import CockroachDBSaver

llm = ChatOpenAI(model="gpt-4o-mini")


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

with CockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "session-1"}}

    # Multi-turn conversation with memory
    app.invoke({"messages": [("user", "My name is Alice")]}, config)
    result = app.invoke({"messages": [("user", "What is my name?")]}, config)
    print(result["messages"][-1].content)  # Should remember "Alice"
```

## Human-in-the-Loop

Interrupt and resume workflow execution:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_cockroachdb import CockroachDBSaver


def propose(state: MessagesState):
    return {"messages": [("assistant", "Proposed action: send email")]}


def execute(state: MessagesState):
    return {"messages": [("assistant", "Email sent successfully")]}


graph = StateGraph(MessagesState)
graph.add_node("propose", propose)
graph.add_node("execute", execute)
graph.add_edge(START, "propose")
graph.add_edge("propose", "execute")
graph.add_edge("execute", END)

DB_URI = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

with CockroachDBSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"],
    )

    config = {"configurable": {"thread_id": "approval-1"}}

    # Step 1: Runs "propose", pauses before "execute"
    result = app.invoke({"messages": [("user", "Send the report")]}, config)
    print("Waiting for approval...")

    # Step 2: Human approves, resume execution
    result = app.invoke(None, config)
    print(result["messages"][-1])
```

## Next Steps

- [Checkpointer Guide](../guides/checkpointer.md) - Detailed guide with all connection modes
- [API Reference](../api/checkpointer.md) - Full API documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/) - LangGraph framework
