"""Example demonstrating LangGraph checkpointer with CockroachDB.

Shows how to use CockroachDBSaver and AsyncCockroachDBSaver to persist
LangGraph workflow state for short-term memory, human-in-the-loop, and
fault-tolerant agent execution.
"""

import asyncio
import os

from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from langgraph.checkpoint.serde.types import TASKS

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


def sync_example() -> None:
    """Demonstrate synchronous checkpointer usage."""
    print("1. Synchronous CockroachDBSaver\n")

    with CockroachDBSaver.from_conn_string(CONNECTION_STRING) as saver:
        saver.setup()
        print("   Tables created successfully")

        # Save a checkpoint
        config = {"configurable": {"thread_id": "demo-thread", "checkpoint_ns": ""}}
        chkpnt = empty_checkpoint()
        metadata = {"source": "input", "step": 1, "description": "Initial state"}

        saved_config = saver.put(config, chkpnt, metadata, {})
        print(f"   Saved checkpoint: {saved_config['configurable']['checkpoint_id'][:12]}...")

        # Retrieve it
        result = saver.get_tuple(saved_config)
        print(f"   Retrieved: source={result.metadata['source']}, step={result.metadata['step']}")

        # Save a second checkpoint (creates parent chain)
        chkpnt_2 = create_checkpoint(chkpnt, {}, 1)
        saved_config_2 = saver.put(saved_config, chkpnt_2, {"source": "loop", "step": 2}, {})
        print(f"   Saved checkpoint 2: {saved_config_2['configurable']['checkpoint_id'][:12]}...")

        # Get latest by thread_id
        latest_config = {"configurable": {"thread_id": "demo-thread", "checkpoint_ns": ""}}
        latest = saver.get_tuple(latest_config)
        print(f"   Latest checkpoint step: {latest.metadata['step']}")
        print(f"   Has parent: {latest.parent_config is not None}")

        # List checkpoints
        all_checkpoints = list(saver.list(latest_config))
        print(f"   Total checkpoints for thread: {len(all_checkpoints)}")

        # Filter by metadata
        loop_checkpoints = list(saver.list(None, filter={"source": "loop"}))
        print(f"   Loop checkpoints: {len(loop_checkpoints)}")

        # Store intermediate writes
        saver.put_writes(saved_config_2, [("output", "hello world")], task_id="task-1")
        result_with_writes = saver.get_tuple(saved_config_2)
        print(f"   Pending writes: {len(result_with_writes.pending_writes)}")

        # Delete thread
        saver.delete_thread("demo-thread")
        deleted = saver.get_tuple(latest_config)
        print(f"   After delete: {'None' if deleted is None else 'still exists'}")

    print("   Done\n")


async def async_example() -> None:
    """Demonstrate asynchronous checkpointer usage."""
    print("2. Asynchronous AsyncCockroachDBSaver\n")

    async with AsyncCockroachDBSaver.from_conn_string(CONNECTION_STRING) as saver:
        await saver.setup()
        print("   Tables created successfully")

        # Save checkpoints for multiple threads
        for i in range(3):
            config = {
                "configurable": {
                    "thread_id": f"async-thread-{i}",
                    "checkpoint_ns": "",
                }
            }
            chkpnt = empty_checkpoint()
            await saver.aput(config, chkpnt, {"source": "input", "thread_num": i}, {})

        print("   Saved 3 thread checkpoints")

        # List all
        all_results = [c async for c in saver.alist(None)]
        print(f"   Total checkpoints: {len(all_results)}")

        # Filter
        filtered = [c async for c in saver.alist(None, filter={"thread_num": 1})]
        print(f"   Thread 1 checkpoints: {len(filtered)}")

        # Get by thread
        thread_config = {"configurable": {"thread_id": "async-thread-0"}}
        thread_results = [c async for c in saver.alist(thread_config)]
        print(f"   async-thread-0 checkpoints: {len(thread_results)}")

        # Cleanup
        for i in range(3):
            await saver.adelete_thread(f"async-thread-{i}")
        print("   Cleaned up all threads")

    print("   Done\n")


async def pending_sends_example() -> None:
    """Demonstrate pending sends migration (used by LangGraph internally)."""
    print("3. Pending Sends Migration\n")

    async with AsyncCockroachDBSaver.from_conn_string(CONNECTION_STRING) as saver:
        await saver.setup()

        config = {"configurable": {"thread_id": "sends-demo", "checkpoint_ns": ""}}

        # Save initial checkpoint
        chkpnt_0 = empty_checkpoint()
        config = await saver.aput(config, chkpnt_0, {}, {})

        # Write task sends (simulates LangGraph sending messages between nodes)
        await saver.aput_writes(
            config,
            [(TASKS, "message-1"), (TASKS, "message-2")],
            task_id="task-1",
        )
        await saver.aput_writes(
            config,
            [(TASKS, "message-3")],
            task_id="task-2",
        )
        print("   Wrote 3 pending sends")

        # Current checkpoint doesn't have sends in channel_values
        current = await saver.aget_tuple(config)
        print(f"   Current channel_values: {current.checkpoint['channel_values']}")

        # Create next checkpoint -- sends migrate to channel_values
        chkpnt_1 = create_checkpoint(chkpnt_0, {}, 1)
        config = await saver.aput(config, chkpnt_1, {}, {})

        migrated = await saver.aget_tuple(config)
        tasks = migrated.checkpoint["channel_values"].get(TASKS, [])
        print(f"   Migrated sends: {tasks}")
        print(f"   TASKS in channel_versions: {TASKS in migrated.checkpoint['channel_versions']}")

        await saver.adelete_thread("sends-demo")

    print("   Done\n")


async def connection_pool_example() -> None:
    """Demonstrate connection pool mode."""
    print("4. Connection Pool Mode\n")

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from langchain_cockroachdb.checkpointer.async_saver import _sanitize_conn_string

    # Use the built-in helper to normalize cockroachdb:// URLs for raw psycopg.
    # Raw psycopg only understands postgresql://, so the helper converts
    # cockroachdb:// -> postgresql:// internally.  Users should always pass
    # cockroachdb:// URLs; the conversion is an implementation detail.
    pg_uri = _sanitize_conn_string(CONNECTION_STRING)

    async with AsyncConnectionPool(
        pg_uri,
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},
    ) as pool:
        saver = AsyncCockroachDBSaver(pool)
        await saver.setup()

        config = {"configurable": {"thread_id": "pool-demo", "checkpoint_ns": ""}}
        chkpnt = empty_checkpoint()
        config = await saver.aput(config, chkpnt, {"mode": "pool"}, {})

        result = await saver.aget_tuple(config)
        print(f"   Pool mode works: {result is not None}")
        print(f"   Metadata: {result.metadata}")

        await saver.adelete_thread("pool-demo")

    print("   Done\n")


async def main() -> None:
    """Run all checkpointer examples."""
    print("LangGraph Checkpointer with CockroachDB\n")
    print("=" * 50 + "\n")

    sync_example()
    await async_example()
    await pending_sends_example()
    await connection_pool_example()

    print("=" * 50)
    print("\nAll examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
