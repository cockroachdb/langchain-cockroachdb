"""Transaction retry logic for CockroachDB.

Implements exponential backoff with jitter for handling:
- 40001 serialization failures (SERIALIZABLE isolation)
- Transient connection errors
- Network timeouts

Based on CockroachDB best practices and patterns from crdb-dump project.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_retryable_error(error: Exception) -> bool:
    """Check if error is transient and should be retried.

    Args:
        error: Exception to check

    Returns:
        True if error should be retried
    """
    error_str = str(error).lower()
    error_code = getattr(error, "pgcode", None)

    # CockroachDB serialization failure (40001)
    if error_code == "40001":
        return True

    # PostgreSQL/CockroachDB transient error patterns
    transient_patterns = [
        "restart transaction",
        "serialization failure",
        "connection",
        "timeout",
        "closed",
        "broken pipe",
        "connection reset",
        "too many clients",
        "server closed",
        "query_wait",
    ]

    return any(pattern in error_str for pattern in transient_patterns)


def async_retry_with_backoff(
    max_retries: int = 5,
    initial_backoff: float = 0.1,
    max_backoff: float = 10.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """Retry async function with exponential backoff on transient errors.

    Implements retry logic following CockroachDB best practices:
    - Exponential backoff to reduce contention
    - Jitter to prevent thundering herd
    - Automatic detection of retryable errors (40001, connection issues)

    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff delay in seconds
        max_backoff: Maximum backoff delay in seconds
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add randomization to backoff to prevent thundering herd

    Returns:
        Decorated async function with retry logic

    Example:
        ```python
        @async_retry_with_backoff(max_retries=3)
        async def insert_data(conn):
            await conn.execute(text("INSERT ..."))
        ```
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            backoff = initial_backoff
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    last_exception = e

                    # Check if error is retryable
                    if not is_retryable_error(e):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    # If we've exhausted retries, raise
                    if attempt >= max_retries - 1:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded in {func.__name__}: {e}"
                        )
                        raise

                    # Calculate backoff with optional jitter (add 0-50% randomization)
                    actual_backoff = backoff * (0.5 + random.random() * 0.5) if jitter else backoff

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {actual_backoff:.2f}s delay. Error: {e}"
                    )

                    # Sleep before retry
                    await asyncio.sleep(actual_backoff)

                    # Exponential backoff for next iteration
                    backoff = min(backoff * backoff_multiplier, max_backoff)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry loop completed without success in {func.__name__}")

        return wrapper

    return decorator


def sync_retry_with_backoff(
    max_retries: int = 5,
    initial_backoff: float = 0.1,
    max_backoff: float = 10.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """Retry sync function with exponential backoff on transient errors.

    Synchronous version of async_retry_with_backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff delay in seconds
        max_backoff: Maximum backoff delay in seconds
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add randomization to backoff

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import time

            backoff = initial_backoff
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not is_retryable_error(e):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    if attempt >= max_retries - 1:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded in {func.__name__}: {e}"
                        )
                        raise

                    actual_backoff = backoff * (0.5 + random.random() * 0.5) if jitter else backoff

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {actual_backoff:.2f}s delay. Error: {e}"
                    )

                    time.sleep(actual_backoff)
                    backoff = min(backoff * backoff_multiplier, max_backoff)

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry loop completed without success in {func.__name__}")

        return wrapper

    return decorator
