"""Unit tests for retry utilities."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from psycopg import OperationalError
from sqlalchemy.exc import DBAPIError, ResourceClosedError

from langchain_cockroachdb.retry import (
    async_retry_with_backoff,
    is_retryable_error,
    sync_retry_with_backoff,
)


class TestIsRetryableError:
    """Test error classification."""

    def test_40001_serialization_error(self) -> None:
        """Test 40001 SERIALIZABLE error detection."""
        error = Exception(
            "restart transaction: TransactionRetryWithProtoRefreshError: TransactionRetryError: retry txn (RETRY_SERIALIZABLE)"
        )
        assert is_retryable_error(error) is True

    def test_40001_in_message(self) -> None:
        """Test 40001 error code detection."""
        error = Exception("ERROR: restart transaction: error code 40001")
        assert is_retryable_error(error) is True

    def test_connection_closed(self) -> None:
        """Test connection closed error."""
        error = ResourceClosedError("This Connection is closed")
        assert is_retryable_error(error) is True

    def test_operational_error(self) -> None:
        """Test psycopg OperationalError."""
        error = OperationalError("connection timeout")
        assert is_retryable_error(error) is True

    def test_dbapi_operational_error(self) -> None:
        """Test SQLAlchemy wrapped OperationalError."""
        orig_error = OperationalError("server closed the connection")
        error = DBAPIError("statement", {}, orig_error, False)
        assert is_retryable_error(error) is True

    def test_non_retryable_error(self) -> None:
        """Test non-retryable errors."""
        error = ValueError("invalid input")
        assert is_retryable_error(error) is False

    def test_syntax_error(self) -> None:
        """Test non-retryable SQL error."""
        error = Exception("syntax error at or near")
        assert is_retryable_error(error) is False


class TestAsyncRetryWithBackoff:
    """Test async retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Test successful execution without retries."""
        mock_func = AsyncMock(return_value="success")

        @async_retry_with_backoff(max_retries=3)
        async def test_func() -> str:
            return await mock_func()

        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_40001(self) -> None:
        """Test retry on serialization error."""
        call_count = 0

        @async_retry_with_backoff(max_retries=3, initial_backoff=0.01)
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("restart transaction: error code 40001")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test max retries exceeded."""

        @async_retry_with_backoff(max_retries=2, initial_backoff=0.01)
        async def test_func() -> str:
            raise Exception("restart transaction: error code 40001")

        with pytest.raises(Exception, match="40001"):
            await test_func()

    @pytest.mark.asyncio
    async def test_non_retryable_error_immediate_fail(self) -> None:
        """Test non-retryable error fails immediately."""
        call_count = 0

        @async_retry_with_backoff(max_retries=5, initial_backoff=0.01)
        async def test_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        with pytest.raises(ValueError, match="invalid input"):
            await test_func()

        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_backoff_timing(self) -> None:
        """Test exponential backoff timing."""
        call_times = []

        @async_retry_with_backoff(
            max_retries=3,
            initial_backoff=0.1,
            backoff_multiplier=2.0,
            jitter=False,  # Disable jitter for predictable timing
        )
        async def test_func() -> str:
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise Exception("restart transaction: error code 40001")
            return "success"

        await test_func()

        # Verify exponential backoff: 0.1s, 0.2s
        assert len(call_times) == 3
        # First retry after ~0.1s
        assert call_times[1] - call_times[0] >= 0.09
        # Second retry after ~0.2s
        assert call_times[2] - call_times[1] >= 0.19

    @pytest.mark.asyncio
    async def test_max_backoff_limit(self) -> None:
        """Test max backoff limit."""
        call_times = []

        @async_retry_with_backoff(
            max_retries=5,
            initial_backoff=1.0,
            max_backoff=1.5,
            backoff_multiplier=2.0,
            jitter=False,
        )
        async def test_func() -> str:
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 4:
                raise Exception("restart transaction: error code 40001")
            return "success"

        await test_func()

        # Backoff should be capped at max_backoff
        # Expected: 1.0, 1.5 (capped), 1.5 (capped)
        assert call_times[2] - call_times[1] >= 1.4
        assert call_times[2] - call_times[1] < 1.7
        assert call_times[3] - call_times[2] >= 1.4
        assert call_times[3] - call_times[2] < 1.7


class TestSyncRetryWithBackoff:
    """Test sync retry decorator."""

    def test_success_no_retry(self) -> None:
        """Test successful execution without retries."""
        mock_func = Mock(return_value="success")

        @sync_retry_with_backoff(max_retries=3)
        def test_func() -> str:
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_40001(self) -> None:
        """Test retry on serialization error."""
        call_count = 0

        @sync_retry_with_backoff(max_retries=3, initial_backoff=0.01)
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("restart transaction: error code 40001")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self) -> None:
        """Test max retries exceeded."""

        @sync_retry_with_backoff(max_retries=2, initial_backoff=0.01)
        def test_func() -> str:
            raise Exception("restart transaction: error code 40001")

        with pytest.raises(Exception, match="40001"):
            test_func()

    def test_non_retryable_error_immediate_fail(self) -> None:
        """Test non-retryable error fails immediately."""
        call_count = 0

        @sync_retry_with_backoff(max_retries=5, initial_backoff=0.01)
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        with pytest.raises(ValueError, match="invalid input"):
            test_func()

        assert call_count == 1  # Should not retry
