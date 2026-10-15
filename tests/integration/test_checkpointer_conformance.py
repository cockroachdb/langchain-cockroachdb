"""LangGraph's official checkpointer conformance suite, against both savers."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

pytest.importorskip("langgraph.checkpoint.conformance")

from langgraph.checkpoint.conformance import checkpointer_test, validate  # noqa: E402
from langgraph.checkpoint.conformance.report import ProgressCallbacks  # noqa: E402

from langchain_cockroachdb import AsyncCockroachDBSaver, CockroachDBSaver  # noqa: E402

POOL_OPTIONS = {"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0}


class ThreadedSyncSaver(CockroachDBSaver):
    """Runs the sync saver's methods in threads so the async suite can drive them."""

    async def aget_tuple(self, config):  # type: ignore[no-untyped-def]
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):  # type: ignore[no-untyped-def]
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):  # type: ignore[no-untyped-def]
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):  # type: ignore[no-untyped-def]
        await asyncio.to_thread(self.delete_thread, thread_id)


async def test_async_saver_conformance(libpq_url: str) -> None:
    @checkpointer_test(name="AsyncCockroachDBSaver")
    async def factory() -> AsyncIterator[AsyncCockroachDBSaver]:
        async with AsyncConnectionPool(
            libpq_url, min_size=1, max_size=4, kwargs=POOL_OPTIONS, open=False
        ) as pool:
            saver = AsyncCockroachDBSaver(pool)
            await saver.setup()
            yield saver

    report = await validate(factory, progress=ProgressCallbacks.verbose())
    report.print_report()
    assert report.passed_all_base()


async def test_sync_saver_conformance(libpq_url: str) -> None:
    @checkpointer_test(name="CockroachDBSaver")
    async def factory() -> AsyncIterator[ThreadedSyncSaver]:
        pool = ConnectionPool(libpq_url, min_size=1, max_size=4, kwargs=POOL_OPTIONS, open=True)
        try:
            saver = ThreadedSyncSaver(pool)
            await asyncio.to_thread(saver.setup)
            yield saver
        finally:
            pool.close()

    report = await validate(factory, progress=ProgressCallbacks.verbose())
    report.print_report()
    assert report.passed_all_base()
