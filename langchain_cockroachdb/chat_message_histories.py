"""Chat message history implementation for CockroachDB."""

import json
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class CockroachDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in CockroachDB."""

    def __init__(
        self,
        session_id: str,
        connection_string: Optional[str] = None,
        engine: Optional[AsyncEngine] = None,
        table_name: str = "message_store",
        schema: str = "public",
    ):
        """Initialize chat message history.

        Args:
            session_id: Unique session/thread identifier
            connection_string: Database connection string
            engine: Existing async engine
            table_name: Table name for messages
            schema: Database schema
        """
        if engine is None and connection_string is None:
            raise ValueError("Either engine or connection_string must be provided")

        self.session_id = session_id
        self.table_name = table_name
        self.schema = schema
        self._fqn = f"{schema}.{table_name}"

        if engine is None:
            if connection_string is None:
                raise ValueError("connection_string is required when engine is None")

            # Ensure we use the async driver (psycopg, not psycopg2)
            if connection_string.startswith("cockroachdb://"):
                connection_string = connection_string.replace(
                    "cockroachdb://", "cockroachdb+psycopg://", 1
                )

            self.engine = create_async_engine(connection_string)
            self._owns_engine = True
        else:
            self.engine = engine
            self._owns_engine = False

    async def _acreate_table_if_not_exists(self) -> None:
        """Create message store table if it doesn't exist."""
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self._fqn} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id TEXT NOT NULL,
                message JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """

        index_sql = f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_session_idx 
            ON {self._fqn} (session_id, created_at)
        """

        async with self.engine.begin() as conn:
            await conn.execute(text(create_sql))
            await conn.execute(text(index_sql))

    def create_table_if_not_exists(self) -> None:
        """Create table (sync wrapper)."""
        import asyncio

        asyncio.run(self._acreate_table_if_not_exists())

    @property
    def messages(self) -> list[BaseMessage]:
        """Get all messages for this session."""
        import asyncio

        return asyncio.run(self.aget_messages())

    async def aget_messages(self) -> list[BaseMessage]:
        """Get all messages for this session (async)."""
        query = f"""
            SELECT message 
            FROM {self._fqn} 
            WHERE session_id = :session_id 
            ORDER BY created_at ASC
        """

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), {"session_id": self.session_id})
            rows = result.fetchall()

        if not rows:
            return []

        items = [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
        messages: list[BaseMessage] = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session."""
        import asyncio

        asyncio.run(self.aadd_message(message))

    async def aadd_message(self, message: BaseMessage) -> None:
        """Add a message (async)."""
        message_dict = messages_to_dict([message])[0]
        message_json = json.dumps(message_dict)

        insert_sql = f"""
            INSERT INTO {self._fqn} (session_id, message)
            VALUES (:session_id, CAST(:message AS jsonb))
        """

        async with self.engine.begin() as conn:
            await conn.execute(
                text(insert_sql), {"session_id": self.session_id, "message": message_json}
            )

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages."""
        import asyncio

        asyncio.run(self.aadd_messages(messages))

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages (async)."""
        for message in messages:
            await self.aadd_message(message)

    def clear(self) -> None:
        """Clear all messages for this session."""
        import asyncio

        asyncio.run(self.aclear())

    async def aclear(self) -> None:
        """Clear messages (async)."""
        delete_sql = f"""
            DELETE FROM {self._fqn} 
            WHERE session_id = :session_id
        """

        async with self.engine.begin() as conn:
            await conn.execute(text(delete_sql), {"session_id": self.session_id})

    async def aclose(self) -> None:
        """Close engine if we own it."""
        if self._owns_engine:
            await self.engine.dispose()

    def close(self) -> None:
        """Close engine (sync)."""
        import asyncio

        asyncio.run(self.aclose())

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._owns_engine and hasattr(self, "engine"):
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.engine.dispose())
                else:
                    asyncio.run(self.engine.dispose())
            except Exception:
                pass
