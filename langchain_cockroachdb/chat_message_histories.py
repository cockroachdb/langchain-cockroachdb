"""Chat message history implementation for CockroachDB."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from langchain_cockroachdb._loop import get_background_loop
from langchain_cockroachdb._sql import validate_identifier
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.retry import RetryPolicy


class CockroachDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in CockroachDB.

    Pass a shared ``CockroachDBEngine`` to reuse one connection pool and one
    retry policy across sessions. With only a connection string the history
    opens and owns its own engine, which ``close()`` or ``aclose()`` releases.

    Messages get ids generated here and are written with ON CONFLICT DO
    NOTHING, so retrying after an ambiguous result never duplicates them.
    """

    def __init__(
        self,
        session_id: str,
        connection_string: str | None = None,
        engine: CockroachDBEngine | AsyncEngine | None = None,
        table_name: str = "message_store",
        schema: str = "public",
        *,
        retry_policy: RetryPolicy | None = None,
    ):
        """Initialize chat message history.

        Args:
            session_id: Unique session/thread identifier
            connection_string: Database connection string, used when no engine is given
            engine: Shared CockroachDBEngine (recommended) or a SQLAlchemy AsyncEngine
            table_name: Table name for messages
            schema: Database schema
            retry_policy: How operations retry (default: the engine's policy)
        """
        if engine is None and connection_string is None:
            raise ValueError("Either engine or connection_string must be provided")
        validate_identifier(table_name, what="table name")
        validate_identifier(schema, what="schema")
        self.session_id = session_id
        self.table_name = table_name
        self.schema = schema
        self._fqn = f"{schema}.{table_name}"
        self._closed = False
        if isinstance(engine, CockroachDBEngine):
            self._crdb, self._owns_engine = engine, False
        elif engine is not None:
            self._crdb, self._owns_engine = CockroachDBEngine.from_engine(engine), False
        else:
            assert connection_string is not None
            self._crdb = CockroachDBEngine.from_connection_string(connection_string)
            self._owns_engine = True
        self.engine: AsyncEngine = self._crdb.engine
        self.retry_policy = retry_policy or self._crdb.retry_policy

    async def _run(
        self,
        statements: list[str],
        params: dict[str, Any] | None,
        *,
        label: str,
        begin: bool,
        fetch: bool = False,
    ) -> list[Any]:
        """Run statements in one attempt under the retry policy; return the last result's rows."""
        timeout = self.retry_policy.attempt_timeout

        async def attempt() -> list[Any]:
            rows: list[Any] = []
            async with self._crdb._connect(begin=begin, timeout=timeout, label=label) as conn:
                for statement in statements:
                    result = await conn.execute(text(statement), params or {})
                    if fetch:
                        rows = list(result.fetchall())
            return rows

        return await self.retry_policy.arun(attempt, label=label)

    async def _create_table(self) -> None:
        await self._crdb._run_ddl(
            [
                f"""
                CREATE TABLE IF NOT EXISTS {self._fqn} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id TEXT NOT NULL,
                    message JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """,
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_session_idx "
                f"ON {self._fqn} (session_id, created_at)",
            ],
            label=f"create table {self._fqn}",
            policy=self.retry_policy,
        )

    async def acreate_table_if_not_exists(self) -> None:
        """Create the message table and its session index if they do not exist."""
        await self._crdb._run_as_async(self._create_table())

    async def _acreate_table_if_not_exists(self) -> None:
        """Older name for acreate_table_if_not_exists, kept for existing callers."""
        await self.acreate_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        """Create table (sync)."""
        self._crdb._run_as_sync(self._create_table())

    async def _get_messages(self) -> list[BaseMessage]:
        rows = await self._run(
            [
                f"SELECT message FROM {self._fqn} "
                "WHERE session_id = :session_id ORDER BY created_at ASC"
            ],
            {"session_id": self.session_id},
            label=f"read messages from {self._fqn}",
            begin=False,
            fetch=True,
        )
        items = [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
        messages: list[BaseMessage] = messages_from_dict(items)
        return messages

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Get all messages for this session.

        Note: BaseChatMessageHistory declares messages as a writable attribute,
        but we expose it as a read-only property (matching langchain-postgres).
        """
        return self._crdb._run_as_sync(self._get_messages())

    async def aget_messages(self) -> list[BaseMessage]:
        """Get all messages for this session (async)."""
        return await self._crdb._run_as_async(self._get_messages())

    async def _insert_messages(self, messages: Sequence[BaseMessage], ids: Sequence[str]) -> None:
        """Insert messages in one statement; replaying the same ids changes nothing.

        Rows get now() plus their position in microseconds, so messages added
        together keep their order even though they share a transaction.
        """
        if not messages:
            return
        params: dict[str, Any] = {"session_id": self.session_id}
        rows = []
        for n, (message_id, item) in enumerate(
            zip(ids, messages_to_dict(list(messages)), strict=True)
        ):
            rows.append(
                f"(:id{n}, :session_id, CAST(:message{n} AS JSONB), "
                f"now() + CAST(:offset{n} AS INT) * INTERVAL '1 microsecond')"
            )
            params.update({f"id{n}": message_id, f"message{n}": json.dumps(item), f"offset{n}": n})
        await self._run(
            [
                f"INSERT INTO {self._fqn} (id, session_id, message, created_at) "
                f"VALUES {', '.join(rows)} ON CONFLICT (id) DO NOTHING"
            ],
            params,
            label=f"add messages to {self._fqn}",
            begin=True,
        )

    async def _add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Ids are chosen once, outside the retry loop, so every attempt writes the same rows
        await self._insert_messages(messages, [str(uuid.uuid4()) for _ in messages])

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session."""
        self.add_messages([message])

    async def aadd_message(self, message: BaseMessage) -> None:
        """Add a message (async)."""
        await self.aadd_messages([message])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages atomically, in order."""
        self._crdb._run_as_sync(self._add_messages(messages))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages atomically, in order (async)."""
        await self._crdb._run_as_async(self._add_messages(messages))

    async def _clear(self) -> None:
        await self._run(
            [f"DELETE FROM {self._fqn} WHERE session_id = :session_id"],
            {"session_id": self.session_id},
            label=f"clear messages in {self._fqn}",
            begin=True,
        )

    def clear(self) -> None:
        """Clear all messages for this session."""
        self._crdb._run_as_sync(self._clear())

    async def aclear(self) -> None:
        """Clear messages (async)."""
        await self._crdb._run_as_async(self._clear())

    async def aclose(self) -> None:
        """Close the engine if this history created it."""
        if self._owns_engine and not self._closed:
            self._closed = True
            await self._crdb.aclose()

    def close(self) -> None:
        """Close the engine if this history created it (sync)."""
        if self._owns_engine and not self._closed:
            self._closed = True
            self._crdb.close()

    def __del__(self) -> None:
        """Release an owned engine that was never closed, without blocking."""
        if getattr(self, "_owns_engine", False) and not getattr(self, "_closed", True):
            try:  # noqa: SIM105
                asyncio.run_coroutine_threadsafe(self._crdb.engine.dispose(), get_background_loop())
            except Exception:
                pass
