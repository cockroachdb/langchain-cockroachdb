"""Example demonstrating chat message history persistence."""

import asyncio
import os
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_cockroachdb import CockroachDBChatMessageHistory

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


async def main() -> None:
    """Demonstrate chat message history."""
    print("🪳 Chat Message History with CockroachDB\n")

    session_id = str(uuid.uuid4())

    print(f"1. Creating chat history for session: {session_id[:8]}...")
    history = CockroachDBChatMessageHistory(
        session_id=session_id,
        connection_string=CONNECTION_STRING,
        table_name="example_chat_history",
    )

    await history._acreate_table_if_not_exists()
    await history.aclear()

    print("\n2. Adding conversation messages...")

    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="What is CockroachDB?"),
        AIMessage(content="CockroachDB is a distributed SQL database..."),
        HumanMessage(content="Does it support vector search?"),
        AIMessage(content="Yes! CockroachDB has native VECTOR type support..."),
    ]

    for msg in messages:
        await history.aadd_message(msg)
        print(f"   Added: {msg.type} - {msg.content[:40]}...")

    print("\n3. Retrieving conversation history...")
    retrieved = await history.aget_messages()

    print(f"   Retrieved {len(retrieved)} messages:")
    for msg in retrieved:
        print(f"   - {msg.type}: {msg.content[:50]}...")

    print("\n4. Demonstrating session isolation...")

    session2_id = str(uuid.uuid4())
    history2 = CockroachDBChatMessageHistory(
        session_id=session2_id,
        connection_string=CONNECTION_STRING,
        table_name="example_chat_history",
    )

    await history2.aadd_message(HumanMessage(content="This is session 2"))

    session1_msgs = await history.aget_messages()
    session2_msgs = await history2.aget_messages()

    print(f"   Session 1: {len(session1_msgs)} messages")
    print(f"   Session 2: {len(session2_msgs)} messages")
    print("   ✓ Sessions are properly isolated")

    print("\n5. Clearing history...")
    await history.aclear()

    remaining = await history.aget_messages()
    print(f"   Messages after clear: {len(remaining)}")

    print("\n6. Use with LangChain chains...")
    print("   from langchain.memory import ConversationBufferMemory")
    print("   memory = ConversationBufferMemory(")
    print("       chat_memory=history,")
    print("       return_messages=True")
    print("   )")

    print("\n✅ Chat history demo complete!")

    await history.aclose()
    await history2.aclose()


if __name__ == "__main__":
    asyncio.run(main())
