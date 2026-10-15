"""Test that sync methods calling async methods don't trigger false mixed-use warnings."""

import asyncio
import logging

from langchain_cockroachdb import CockroachDBEngine

URL = "cockroachdb://root@localhost:1/defaultdb?sslmode=disable"


def test_sync_method_calling_async_method_does_not_warn(caplog) -> None:
    """Sync method internally calling public async method should not warn about mixed use."""
    engine = CockroachDBEngine.from_connection_string(URL)

    # Define a custom async method that calls another async method
    async def async_operation_that_calls_another() -> str:
        # This simulates a scenario where an async method might call
        # another public async method through _run_as_async
        await engine._run_as_async(asyncio.sleep(0, result="inner"))
        return "ok"

    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.engine"):
        # Call through sync interface
        result = engine._run_as_sync(async_operation_that_calls_another())

    assert result == "ok"
    # Should not warn about mixed use because both operations are on the background loop
    assert "one style per engine" not in caplog.text
