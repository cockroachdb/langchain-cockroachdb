"""Unit test for _run_ddl per-statement retry behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_cockroachdb import CockroachDBEngine, RetryPolicy


@pytest.mark.asyncio
async def test_run_ddl_retries_individual_statements() -> None:
    """Transient failure on statement 2 retries statement 2 only; statement 1 runs once."""
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://root@localhost:1/db",
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff=0.01, jitter=0),
    )

    # Track which statements were executed
    executed = []
    attempt_counts = {"stmt1": 0, "stmt2": 0, "stmt3": 0}

    async def mock_execute(statement):
        # Track execution
        stmt_text = str(statement)
        if "CREATE TABLE t1" in stmt_text:
            stmt_id = "stmt1"
        elif "CREATE TABLE t2" in stmt_text:
            stmt_id = "stmt2"
        elif "CREATE TABLE t3" in stmt_text:
            stmt_id = "stmt3"
        else:
            stmt_id = "unknown"

        executed.append(stmt_id)
        attempt_counts[stmt_id] += 1

        # Fail statement 2 on first attempt with a retryable error
        if stmt_id == "stmt2" and attempt_counts[stmt_id] == 1:
            from psycopg.errors import SerializationFailure

            raise SerializationFailure("40001: restart transaction")

        return MagicMock()

    # Mock the connection
    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute

    # execution_options should return an awaitable that resolves to mock_conn
    async def mock_execution_options(**kwargs):
        return mock_conn

    mock_conn.execution_options = mock_execution_options

    # Mock the context manager for _connect
    mock_connect_cm = AsyncMock()
    mock_connect_cm.__aenter__.return_value = mock_conn
    mock_connect_cm.__aexit__.return_value = None

    with patch.object(engine, "_connect", return_value=mock_connect_cm):
        await engine._run_ddl(
            [
                "CREATE TABLE t1 (id INT PRIMARY KEY)",
                "CREATE TABLE t2 (id INT PRIMARY KEY)",
                "CREATE TABLE t3 (id INT PRIMARY KEY)",
            ],
            label="test setup",
        )

    # Verify statement 1 ran once, statement 2 ran twice (1 failure + 1 success), statement 3 ran once
    assert attempt_counts["stmt1"] == 1, "Statement 1 should run once"
    assert attempt_counts["stmt2"] == 2, "Statement 2 should run twice (retry after failure)"
    assert attempt_counts["stmt3"] == 1, "Statement 3 should run once"

    # Verify the execution order: stmt1, stmt2 (fail), stmt2 (success), stmt3
    assert executed == ["stmt1", "stmt2", "stmt2", "stmt3"]

    await engine.aclose()
