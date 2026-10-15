"""Test that invalid retry arguments fail at engine construction, not at first use."""

import pytest

from langchain_cockroachdb import CockroachDBEngine

URL = "cockroachdb://root@localhost:1/defaultdb?sslmode=disable"


def test_invalid_retry_max_attempts_fails_at_construction() -> None:
    """retry_max_attempts=0 should fail when from_connection_string is called."""
    with pytest.raises(ValueError, match="max_attempts must be an integer of at least 1"):
        CockroachDBEngine.from_connection_string(URL, retry_max_attempts=0)


def test_invalid_retry_jitter_fails_at_construction() -> None:
    """retry_jitter=0.3 (not a bool) should fail when from_connection_string is called."""
    with pytest.raises(
        TypeError, match="jitter must be True, False, or None; for a ratio, use RetryPolicy"
    ):
        CockroachDBEngine.from_connection_string(URL, retry_jitter=0.3)  # type: ignore[arg-type]


def test_invalid_retry_initial_backoff_fails_at_construction() -> None:
    """Negative initial_backoff should fail at construction."""
    with pytest.raises(ValueError, match="initial_backoff must not be negative"):
        CockroachDBEngine.from_connection_string(URL, retry_initial_backoff=-1.0)
