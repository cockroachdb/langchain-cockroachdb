"""Integration tests for CockroachDBEngine."""

import gc
import os
import threading
import uuid

import pytest
from sqlalchemy import text

from langchain_cockroachdb.engine import CockroachDBEngine


@pytest.mark.asyncio
class TestCockroachDBEngine:
    """Test CockroachDBEngine with real database."""

    async def test_from_connection_string(self, connection_string: str) -> None:
        """Test engine creation from connection string."""
        engine = CockroachDBEngine.from_connection_string(connection_string)

        assert engine is not None
        assert engine.engine is not None

        await engine.aclose()

    async def test_from_engine(self, async_engine) -> None:
        """Test engine creation from existing engine."""
        engine = CockroachDBEngine.from_engine(async_engine)

        assert engine is not None
        assert engine.engine == async_engine

    async def test_ainit_vectorstore_table(self, cockroachdb_engine: CockroachDBEngine) -> None:
        """Test vector store table creation."""
        table_name = "test_vectors"
        vector_dim = 3

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=vector_dim,
            drop_if_exists=True,
        )

        async with cockroachdb_engine.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """),
                {"table_name": table_name},
            )
            columns = {row[0]: row[1] for row in result.fetchall()}

        assert "id" in columns
        assert "content" in columns
        assert "embedding" in columns
        assert "metadata" in columns
        assert columns["metadata"] == "jsonb"

    async def test_ainit_vectorstore_table_with_tsvector(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """Test table creation with full-text search support."""
        table_name = "test_fts_vectors"
        vector_dim = 3

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_dimension=vector_dim,
            create_tsvector=True,
            drop_if_exists=True,
        )

        async with cockroachdb_engine.engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    AND column_name LIKE '%tsvector%'
                """),
                {"table_name": table_name},
            )
            columns = result.fetchall()

        assert len(columns) > 0
        assert columns[0][1] == "tsvector"

    async def test_context_manager(self, connection_string: str) -> None:
        """Test async context manager."""
        async with CockroachDBEngine.from_connection_string(connection_string) as engine:
            assert engine is not None

            async with engine.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

    async def test_connection_pooling(self, connection_string: str) -> None:
        """Test connection pool configuration."""
        engine = CockroachDBEngine.from_connection_string(
            connection_string,
            pool_size=5,
            max_overflow=10,
        )

        assert engine.engine.pool.size() == 5

        await engine.aclose()


def test_sync_calls_from_many_threads(connection_string: str) -> None:
    engine = CockroachDBEngine.from_connection_string(connection_string)
    errors: list[BaseException] = []

    def create(i: int) -> None:
        try:
            engine.init_vectorstore_table(f"thread_table_{i}", 3, drop_if_exists=True)
            engine.drop_table(f"thread_table_{i}")
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not any(thread.is_alive() for thread in threads), "a worker thread did not finish"
    engine.close()
    assert errors == []


async def test_postgresql_spelling_works(connection_string: str) -> None:
    url = "postgresql://" + connection_string.split("://", 1)[1]
    async with CockroachDBEngine.from_connection_string(url) as engine:
        await engine.ainit_vectorstore_table("spelling_test", 3, drop_if_exists=True)


async def test_invalid_identifiers_are_rejected_before_sql(connection_string: str) -> None:
    async with CockroachDBEngine.from_connection_string(connection_string) as engine:
        with pytest.raises(ValueError, match="table name"):
            await engine.ainit_vectorstore_table("bad-name", 3)
        with pytest.raises(ValueError, match="vector_dimension"):
            await engine.ainit_vectorstore_table("ok_name", 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_sync_api_works_in_forked_child(connection_string: str) -> None:
    import time

    engine = CockroachDBEngine.from_connection_string(
        connection_string, pool_pre_ping=False, pool_size=1, max_overflow=0
    )
    engine.init_vectorstore_table("fork_test", 3, drop_if_exists=True)

    async def get_session_id() -> str:
        async with engine._connect(begin=False, timeout=None, label="session id check") as conn:
            result = await conn.execute(text("SHOW session_id"))
            return result.scalar_one()

    parent_session_id = engine._run_as_sync(get_session_id())

    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            child_session_id = engine._run_as_sync(get_session_id())
            if child_session_id == parent_session_id:
                os._exit(99)  # child reused parent session, hook failed
            engine.init_vectorstore_table("fork_child", 3, drop_if_exists=True)
            engine.drop_table("fork_child")
            gc.collect()
            code = 0
        finally:
            os._exit(code)

    deadline = time.time() + 30
    while time.time() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        raise AssertionError("child hung")

    exit_code = os.waitstatus_to_exitcode(status)
    assert exit_code == 0, f"child failed with code {exit_code}"

    session_after_fork = engine._run_as_sync(get_session_id())
    assert session_after_fork == parent_session_id, "parent session was killed by child"

    engine.drop_table("fork_test")
    engine.close()


DDL_OFF = {"options": "-c autocommit_before_ddl=off -c create_table_with_schema_locked=on"}


async def test_repeat_setup_works_with_autocommit_before_ddl_off(engine_factory) -> None:
    # Forcing schema_locked on makes this fail the same way on v25.4, where it defaults off
    engine = engine_factory(connect_args=DDL_OFF)
    name = f"ddl_off_{uuid.uuid4().hex[:8]}"
    for _ in range(2):
        await engine.ainit_vectorstore_table(
            name, 8, namespace_column="tenant_id", create_tsvector=True
        )
    async with engine.engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'public' AND tablename = :table"
            ),
            {"table": name},
        )
        indexes = {row[0] for row in result}
    assert {f"{name}_tenant_id_idx", f"{name}_content_tsvector_idx"} <= indexes
    await engine.adrop_table(name)
    await engine.adrop_table(name)
