"""Unit test for _run_ddl per-statement retry behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.exc import OperationalError

from langchain_cockroachdb import CockroachDBEngine, RetryPolicy


async def test_run_ddl_retries_individual_statements() -> None:
    """Transient failure on statement 2 retries statement 2 only; statement 1 runs once."""
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://root@localhost:1/db",
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff=0.01, jitter=0),
    )

    executed = []
    attempt_counts = {"stmt1": 0, "stmt2": 0, "stmt3": 0}
    connect_count = 0

    async def mock_execute(statement):
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

        if stmt_id == "stmt2" and attempt_counts[stmt_id] == 1:
            from psycopg.errors import SerializationFailure

            orig_error = SerializationFailure("40001: restart transaction")
            raise OperationalError("statement", {}, orig_error)

        return MagicMock()

    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute

    async def mock_execution_options(**kwargs):
        return mock_conn

    mock_conn.execution_options = mock_execution_options

    def mock_connect_factory(*args, **kwargs):
        nonlocal connect_count
        connect_count += 1
        mock_connect_cm = AsyncMock()
        mock_connect_cm.__aenter__.return_value = mock_conn
        mock_connect_cm.__aexit__.return_value = None
        return mock_connect_cm

    try:
        with patch.object(engine, "_connect", side_effect=mock_connect_factory):
            await engine._run_ddl(
                [
                    "CREATE TABLE t1 (id INT PRIMARY KEY)",
                    "CREATE TABLE t2 (id INT PRIMARY KEY)",
                    "CREATE TABLE t3 (id INT PRIMARY KEY)",
                ],
                label="test setup",
            )

        assert attempt_counts["stmt1"] == 1
        assert attempt_counts["stmt2"] == 2
        assert attempt_counts["stmt3"] == 1
        assert executed == ["stmt1", "stmt2", "stmt2", "stmt3"]
        assert connect_count == 4, "Should check out a fresh connection per attempt"
    finally:
        await engine.aclose()
