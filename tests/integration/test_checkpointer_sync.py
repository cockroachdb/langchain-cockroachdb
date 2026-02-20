"""Sync checkpointer integration tests for CockroachDB.

Modeled on langgraph-checkpoint-postgres test_sync.py.
Tests CockroachDBSaver with connection, pool, and pipeline modes.
"""

from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_cockroachdb.checkpointer.saver import CockroachDBSaver, _sanitize_conn_string


def _pg_uri(connection_string: str) -> str:
    """Convert cockroachdb:// to postgresql:// for raw psycopg."""
    return _sanitize_conn_string(connection_string)


@contextmanager
def _base_saver(connection_string: str):
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    pg_uri = _pg_uri(connection_string)
    base_uri = pg_uri.rsplit("/", 1)[0]

    with Connection.connect(pg_uri, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")

    try:
        with Connection.connect(
            f"{base_uri}/{database}?sslmode=disable"
            if "sslmode=disable" in pg_uri
            else f"{base_uri}/{database}?sslmode=verify-full",
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = CockroachDBSaver(conn)
            checkpointer.setup()
            yield checkpointer
    finally:
        with Connection.connect(pg_uri, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


@contextmanager
def _pool_saver(connection_string: str):
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    pg_uri = _pg_uri(connection_string)
    base_uri = pg_uri.rsplit("/", 1)[0]

    with Connection.connect(pg_uri, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")

    db_uri = (
        f"{base_uri}/{database}?sslmode=disable"
        if "sslmode=disable" in pg_uri
        else f"{base_uri}/{database}?sslmode=verify-full"
    )
    try:
        with ConnectionPool(
            db_uri,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
        ) as pool:
            checkpointer = CockroachDBSaver(pool)
            checkpointer.setup()
            yield checkpointer
    finally:
        with Connection.connect(pg_uri, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


@contextmanager
def _saver(name: str, connection_string: str):
    if name == "base":
        with _base_saver(connection_string) as saver:
            yield saver
    elif name == "pool":
        with _pool_saver(connection_string) as saver:
            yield saver


@contextmanager
def _from_conn_string_saver(connection_string: str):
    """Test the from_conn_string factory method."""
    database = f"test_{uuid4().hex[:16]}"
    pg_uri = _pg_uri(connection_string)
    base_uri = pg_uri.rsplit("/", 1)[0]

    with Connection.connect(pg_uri, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")

    db_uri = (
        f"{base_uri}/{database}?sslmode=disable"
        if "sslmode=disable" in pg_uri
        else f"{base_uri}/{database}?sslmode=verify-full"
    )
    try:
        # Re-add cockroachdb:// to test URL sanitization
        crdb_uri = db_uri.replace("postgresql://", "cockroachdb://", 1)
        with CockroachDBSaver.from_conn_string(crdb_uri) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        with Connection.connect(pg_uri, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {database} CASCADE")


class TestCockroachDBSaverSync:
    """Test sync CockroachDBSaver."""

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_put_get(self, connection_string: str, saver_name: str) -> None:
        """Test basic put and get_tuple operations."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
            }
            chkpnt = empty_checkpoint()
            config = saver.put(config, chkpnt, {"source": "input", "step": 2}, {})

            result = saver.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == chkpnt["id"]
            assert result.metadata["source"] == "input"
            assert result.metadata["step"] == 2

            # Also test get latest by thread_id (no checkpoint_id)
            latest_config: RunnableConfig = {
                "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
            }
            result_latest = saver.get_tuple(latest_config)
            assert result_latest is not None
            assert result_latest.checkpoint["id"] == chkpnt["id"]

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_search(self, connection_string: str, saver_name: str) -> None:
        """Test list with metadata filtering."""
        with _saver(saver_name, connection_string) as saver:
            c1: RunnableConfig = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            c2: RunnableConfig = {"configurable": {"thread_id": "thread-2", "checkpoint_ns": ""}}
            c3: RunnableConfig = {
                "configurable": {"thread_id": "thread-2", "checkpoint_ns": "inner"}
            }
            saver.put(c1, empty_checkpoint(), {"source": "input", "step": 2}, {})
            saver.put(c2, empty_checkpoint(), {"source": "loop", "step": 1}, {})
            saver.put(c3, empty_checkpoint(), {}, {})

            results = list(saver.list(None, filter={"source": "input"}))
            assert len(results) == 1

            results = list(saver.list(None, filter={"step": 1}))
            assert len(results) == 1

            results = list(saver.list(None, filter={}))
            assert len(results) == 3

            results = list(saver.list(None, filter={"source": "update", "step": 1}))
            assert len(results) == 0

            results = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
            assert len(results) == 2
            namespaces = {r.config["configurable"]["checkpoint_ns"] for r in results}
            assert namespaces == {"", "inner"}

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_list_limit(self, connection_string: str, saver_name: str) -> None:
        """Test list with limit parameter."""
        with _saver(saver_name, connection_string) as saver:
            c1: RunnableConfig = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
            c2: RunnableConfig = {"configurable": {"thread_id": "thread-2", "checkpoint_ns": ""}}
            saver.put(c1, empty_checkpoint(), {}, {})
            saver.put(c2, empty_checkpoint(), {}, {})

            results = list(saver.list(None, limit=1))
            assert len(results) == 1

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_put_writes(self, connection_string: str, saver_name: str) -> None:
        """Test put_writes stores intermediate writes."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-writes", "checkpoint_ns": ""}
            }
            config = saver.put(config, empty_checkpoint(), {}, {})
            saver.put_writes(config, [("channel1", "value1")], task_id="task-1")

            result = saver.get_tuple(config)
            assert result is not None
            assert len(result.pending_writes) == 1
            assert result.pending_writes[0][1] == "channel1"
            assert result.pending_writes[0][2] == "value1"

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_delete_thread(self, connection_string: str, saver_name: str) -> None:
        """Test deleting all checkpoints for a thread."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-delete", "checkpoint_ns": ""}
            }
            config = saver.put(config, empty_checkpoint(), {"source": "input"}, {})

            result = saver.get_tuple(config)
            assert result is not None

            saver.delete_thread("thread-delete")

            lookup: RunnableConfig = {
                "configurable": {"thread_id": "thread-delete", "checkpoint_ns": ""}
            }
            result = saver.get_tuple(lookup)
            assert result is None

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_get_nonexistent(self, connection_string: str, saver_name: str) -> None:
        """Test get_tuple returns None for non-existent checkpoint."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "nonexistent",
                    "checkpoint_ns": "",
                }
            }
            result = saver.get_tuple(config)
            assert result is None

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_parent_checkpoint(self, connection_string: str, saver_name: str) -> None:
        """Test parent checkpoint linkage."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-parent",
                    "checkpoint_ns": "",
                }
            }
            chkpnt_1 = empty_checkpoint()
            config = saver.put(config, chkpnt_1, {"step": 1}, {})

            chkpnt_2 = create_checkpoint(chkpnt_1, {}, 1)
            config = saver.put(config, chkpnt_2, {"step": 2}, {})

            result = saver.get_tuple(config)
            assert result is not None
            assert result.parent_config is not None
            assert result.parent_config["configurable"]["checkpoint_id"] == chkpnt_1["id"]

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_null_chars(self, connection_string: str, saver_name: str) -> None:
        """Test that null characters in metadata are handled."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-null", "checkpoint_ns": ""}
            }
            config = saver.put(config, empty_checkpoint(), {"my_key": "\x00abc"}, {})
            assert saver.get_tuple(config).metadata["my_key"] == "abc"

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_combined_metadata(self, connection_string: str, saver_name: str) -> None:
        """Test that config metadata is merged with checkpoint metadata."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-meta",
                    "checkpoint_ns": "",
                },
                "metadata": {"run_id": "my_run_id"},
            }
            chkpnt = create_checkpoint(empty_checkpoint(), {}, 1)
            saver.put(config, chkpnt, {"source": "loop", "step": 1}, {})

            lookup: RunnableConfig = {
                "configurable": {"thread_id": "thread-meta", "checkpoint_ns": ""}
            }
            result = saver.get_tuple(lookup)
            assert result.metadata["source"] == "loop"
            assert result.metadata["run_id"] == "my_run_id"

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_pending_sends_migration(self, connection_string: str, saver_name: str) -> None:
        """Test pending sends are migrated to the next checkpoint."""
        with _saver(saver_name, connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-sends",
                    "checkpoint_ns": "",
                }
            }
            checkpoint_0 = empty_checkpoint()
            config = saver.put(config, checkpoint_0, {}, {})
            saver.put_writes(config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1")
            saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")

            # checkpoint_0 should not have pending sends
            tuple_0 = saver.get_tuple(config)
            assert tuple_0.checkpoint["channel_values"] == {}

            # create next checkpoint
            checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
            config = saver.put(config, checkpoint_1, {}, {})

            # checkpoint_1 should have pending sends from checkpoint_0
            tuple_1 = saver.get_tuple(config)
            assert tuple_1.checkpoint["channel_values"] == {TASKS: ["send-1", "send-2", "send-3"]}
            assert TASKS in tuple_1.checkpoint["channel_versions"]

    def test_from_conn_string(self, connection_string: str) -> None:
        """Test the from_conn_string factory with cockroachdb:// URL."""
        with _from_conn_string_saver(connection_string) as saver:
            config: RunnableConfig = {
                "configurable": {"thread_id": "thread-factory", "checkpoint_ns": ""}
            }
            chkpnt = empty_checkpoint()
            config = saver.put(config, chkpnt, {"source": "input"}, {})
            result = saver.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == chkpnt["id"]

    @pytest.mark.parametrize("saver_name", ["base", "pool"])
    def test_idempotent_setup(self, connection_string: str, saver_name: str) -> None:
        """Test that setup() can be called multiple times safely."""
        with _saver(saver_name, connection_string) as saver:
            # setup() was already called in the fixture; call it again
            saver.setup()
            # Should still work
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-idempotent",
                    "checkpoint_ns": "",
                }
            }
            chkpnt = empty_checkpoint()
            config = saver.put(config, chkpnt, {}, {})
            result = saver.get_tuple(config)
            assert result is not None
