"""Retries in action: real CockroachDB errors, recovered automatically.

Run it against a local development cluster; injecting faults needs admin rights:

    COCKROACHDB_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable" \\
        python examples/resilience.py

CockroachDB itself raises every error here: crdb_internal.force_error for the
SQLSTATEs the retry policy handles, and CANCEL SESSIONS for dropped
connections. Each WARNING line is a retry the application never sees.
"""

import asyncio
import logging
import os
import uuid

import psycopg
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import event

from langchain_cockroachdb import (
    RETRYABLE_SQLSTATES,
    AsyncCockroachDBSaver,
    AsyncCockroachDBVectorStore,
    CockroachDBChatMessageHistory,
    CockroachDBEngine,
    RetryPolicy,
)

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)
LIBPQ_URL = "postgresql://" + CONNECTION_STRING.split("://", 1)[1]
APP = f"resilience_demo_{uuid.uuid4().hex[:6]}"

logging.basicConfig(level=logging.WARNING, format="   %(levelname)s %(name)s: %(message)s")


def unsafe_internals_setting_exists() -> bool:
    """crdb_internal.force_error needs allow_unsafe_internals on CockroachDB v26 and later."""
    with psycopg.connect(LIBPQ_URL, autocommit=True) as conn:
        try:
            conn.execute("SET allow_unsafe_internals = true")
        except psycopg.Error:
            return False
    return True


NEEDS_SETTING = unsafe_internals_setting_exists()


def fail_next_statement(engine: CockroachDBEngine, sqlstate: str, match: str) -> None:
    """Make the next statement containing ``match`` fail on the server with ``sqlstate``."""
    sync_engine = engine.engine.sync_engine
    injected = [False]

    def before(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if injected[0] or match not in statement.lower():
            return statement, parameters
        injected[0] = True
        if NEEDS_SETTING:
            cursor.execute("SET allow_unsafe_internals = true")
        print(f"   injecting {sqlstate} into: {statement.split()[0]} ...")
        return (
            "SELECT crdb_internal.force_error(%(code)s, %(message)s)",
            {"code": sqlstate, "message": f"demo fault {sqlstate}"},
        )

    event.listen(sync_engine, "before_cursor_execute", before, retval=True)


def kill_demo_connections() -> None:
    """Cancel every session this demo opened, like a load balancer reset."""
    with psycopg.connect(LIBPQ_URL, autocommit=True) as admin:
        admin.execute(
            sql.SQL(
                "CANCEL SESSIONS (SELECT session_id FROM [SHOW CLUSTER SESSIONS] "
                "WHERE application_name = {})"
            ).format(sql.Literal(APP))
        )


async def main() -> None:
    print("\nRetries in action\n")
    policy = RetryPolicy(
        max_attempts=6,
        retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"},
        attempt_timeout=10,
    )
    # pool_pre_ping is off here only so dropped connections reach the retry
    # logic; keep it on in production, where it quietly replaces them.
    engine = CockroachDBEngine.from_connection_string(
        CONNECTION_STRING,
        retry_policy=policy,
        pool_pre_ping=False,
        connect_args={"application_name": APP},
    )
    table = f"resilience_{uuid.uuid4().hex[:6]}"
    await engine.ainit_vectorstore_table(table, 8)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=8), table)

    print("1. Vector store insert hits a serialization failure (40001)")
    fail_next_statement(engine, "40001", "insert into")
    ids = await store.aadd_texts(["CockroachDB retries for you", "Retries are idempotent here"])
    print(f"   stored {len(ids)} documents\n")

    print("2. Vector search gets an ambiguous result (40003)")
    fail_next_statement(engine, "40003", "as distance")
    results = await store.asimilarity_search("retries", k=2)
    print(f"   found {len(results)} documents\n")

    print("3. Chat history write lands on a draining node (57P01)")
    history = CockroachDBChatMessageHistory(
        str(uuid.uuid4()), engine=engine, table_name=f"{table}_chat"
    )
    await history.acreate_table_if_not_exists()
    fail_next_statement(engine, "57P01", "insert into")
    await history.aadd_messages(
        [HumanMessage("Is anything lost?"), AIMessage("No, and nothing is doubled.")]
    )
    print(f"   history holds {len(await history.aget_messages())} messages\n")

    print("4. Every pooled connection is killed")
    kill_demo_connections()
    await asyncio.sleep(0.3)
    print(f"   history still reads {len(await history.aget_messages())} messages\n")

    print("5. LangGraph checkpointer on a pool loses its connections between writes")
    async with AsyncConnectionPool(
        LIBPQ_URL,
        min_size=1,
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": APP},
        open=False,
    ) as pool:
        saver = AsyncCockroachDBSaver(pool, retry_policy=policy)
        await saver.setup()
        thread = {"configurable": {"thread_id": str(uuid.uuid4()), "checkpoint_ns": ""}}
        saved = await saver.aput(thread, empty_checkpoint(), {"step": 1}, {})
        kill_demo_connections()
        await asyncio.sleep(0.3)
        print(
            f"   checkpoint readable after the reset: {await saver.aget_tuple(saved) is not None}"
        )
        await saver.adelete_thread(thread["configurable"]["thread_id"])

    await engine.adrop_table(table)
    await engine.adrop_table(f"{table}_chat")
    await engine.aclose()
    print("\nEvery operation succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
