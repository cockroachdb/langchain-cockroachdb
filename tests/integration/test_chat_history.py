"""Integration tests for chat message history."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_cockroachdb.chat_message_histories import CockroachDBChatMessageHistory


@pytest.mark.asyncio
class TestCockroachDBChatMessageHistory:
    """Test chat message history with real database."""

    @pytest.fixture
    async def history(self, connection_string: str) -> CockroachDBChatMessageHistory:
        """Create chat message history for testing."""
        history = CockroachDBChatMessageHistory(
            session_id="test_session",
            connection_string=connection_string,
            table_name="test_messages",
        )

        await history._acreate_table_if_not_exists()
        await history.aclear()

        return history

    async def test_add_and_get_messages(self, history: CockroachDBChatMessageHistory) -> None:
        """Test adding and retrieving messages."""
        msg1 = HumanMessage(content="Hello!")
        msg2 = AIMessage(content="Hi there!")

        await history.aadd_message(msg1)
        await history.aadd_message(msg2)

        messages = await history.aget_messages()

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello!"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there!"

    async def test_add_multiple_messages(self, history: CockroachDBChatMessageHistory) -> None:
        """Test adding multiple messages at once."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
        ]

        await history.aadd_messages(messages)

        retrieved = await history.aget_messages()
        assert len(retrieved) == 3

    async def test_message_ordering(self, history: CockroachDBChatMessageHistory) -> None:
        """Test that messages are retrieved in order."""
        for i in range(5):
            await history.aadd_message(HumanMessage(content=f"Message {i}"))

        messages = await history.aget_messages()

        assert len(messages) == 5
        for i, msg in enumerate(messages):
            assert msg.content == f"Message {i}"

    async def test_clear_messages(self, history: CockroachDBChatMessageHistory) -> None:
        """Test clearing messages."""
        await history.aadd_message(HumanMessage(content="Test"))
        await history.aadd_message(AIMessage(content="Response"))

        messages = await history.aget_messages()
        assert len(messages) == 2

        await history.aclear()

        messages = await history.aget_messages()
        assert len(messages) == 0

    async def test_session_isolation(self, connection_string: str) -> None:
        """Test that different sessions are isolated."""
        history1 = CockroachDBChatMessageHistory(
            session_id="session_1",
            connection_string=connection_string,
            table_name="test_isolation",
        )

        history2 = CockroachDBChatMessageHistory(
            session_id="session_2",
            connection_string=connection_string,
            table_name="test_isolation",
        )

        await history1._acreate_table_if_not_exists()
        await history1.aclear()
        await history2.aclear()

        await history1.aadd_message(HumanMessage(content="Session 1"))
        await history2.aadd_message(HumanMessage(content="Session 2"))

        messages1 = await history1.aget_messages()
        messages2 = await history2.aget_messages()

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0].content == "Session 1"
        assert messages2[0].content == "Session 2"

        await history1.aclose()
        await history2.aclose()

    async def test_system_message(self, history: CockroachDBChatMessageHistory) -> None:
        """Test storing system messages."""
        msg = SystemMessage(content="System prompt")
        await history.aadd_message(msg)

        messages = await history.aget_messages()

        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "System prompt"

    async def test_message_with_metadata(self, history: CockroachDBChatMessageHistory) -> None:
        """Test storing messages with metadata."""
        msg = HumanMessage(
            content="Test",
            additional_kwargs={"user_id": "123", "timestamp": "2024-01-01"},
        )
        await history.aadd_message(msg)

        messages = await history.aget_messages()

        assert len(messages) == 1
        assert messages[0].additional_kwargs.get("user_id") == "123"

    async def test_empty_session(self, history: CockroachDBChatMessageHistory) -> None:
        """Test retrieving from empty session."""
        messages = await history.aget_messages()
        assert len(messages) == 0
