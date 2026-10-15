"""Retry policy for CockroachDB operations.

CockroachDB runs transactions at SERIALIZABLE isolation by default, so clients
are expected to retry some errors. This module decides which errors are worth
retrying and runs operations with exponential backoff and jitter.

A 40001 error means the transaction did not commit, so retrying it is always
safe. After 40003 (result is ambiguous) or a lost connection the transaction
may have committed, so only idempotent work should be retried. Every write
this package issues is idempotent (UPSERT, ON CONFLICT DO NOTHING, DELETE,
IF [NOT] EXISTS).
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import importlib
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import psycopg
from psycopg_pool import PoolTimeout
from sqlalchemy.exc import DBAPIError, StatementError
from sqlalchemy.exc import TimeoutError as PoolCheckoutTimeout

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_SQLSTATES: frozenset[str] = frozenset(
    {
        "40001",  # serialization_failure: the transaction did not commit
        "40003",  # statement_completion_unknown: the result is ambiguous
        "57P01",  # admin_shutdown: the node is draining or shutting down
        "58C00",  # range_unavailable: replicas unreachable (CockroachDB)
        "58C01",  # internal_connection_failure between nodes (CockroachDB)
        "53300",  # too_many_connections
    }
)

# 08 is connection_exception, including CockroachDB sqlproxy's 08C00
RETRYABLE_SQLSTATE_CLASSES: frozenset[str] = frozenset({"08"})

# Codes too generic to decide on; message phrases decide for these
UNCATEGORIZED_SQLSTATES: frozenset[str] = frozenset({"XXUUU", "XX000"})

TRANSIENT_MESSAGES: tuple[str, ...] = (
    # Lost or refused connections, as worded by libpq, psycopg and CockroachDB
    "server closed the connection unexpectedly",
    "connection already closed",
    "the connection is closed",
    "the connection is lost",
    "connection socket closed",
    "could not receive data from server",
    "could not send data to server",
    "ssl connection has been closed unexpectedly",
    "ssl syscall error",
    "eof detected",
    "connection refused",
    "connection reset by peer",
    "broken pipe",
    "connection timed out",
    "operation timed out",
    "timeout expired",
    "i/o timeout",
    "no route to host",
    "network is unreachable",
    # Name resolution during failover
    "could not translate host name",
    "failed to resolve host",
    "temporary failure in name resolution",
    "no such host",
    # Node drains, restarts and connection limits (connect-time errors carry no code)
    "server is not accepting clients",
    "server is shutting down",
    "terminating connection",
    "too many clients",
    # CockroachDB retry and failover errors that can arrive without a specific code
    "restart transaction",
    "transactionretryerror",
    "result is ambiguous",
    "failed to connect",
    "initial connection heartbeat failed",
    "sending to all replicas failed",
)

# Connect-time failures that retrying cannot fix. psycopg reports these
# without a SQLSTATE, and retrying a bad password can trip account lockouts.
NON_TRANSIENT_MESSAGES: tuple[str, ...] = (
    "password authentication failed",
    "authentication failed",
    "no pg_hba.conf entry",
    "certificate",
    "does not support ssl",
    "ssl was required",
    "invalid connection option",
    "invalid dsn",
    "does not exist",
    "permission denied",
)

_MAX_CHAIN = 16


def _error_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an error, the driver error SQLAlchemy wrapped, and explicit causes.

    ``__context__`` is not followed: an unrelated error raised while handling a
    retryable one must not inherit its classification.
    """
    seen: set[int] = set()
    pending = [error]
    while pending and len(seen) < _MAX_CHAIN:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        orig = getattr(current, "orig", None)
        if isinstance(orig, BaseException):
            pending.append(orig)
        if current.__cause__ is not None:
            pending.append(current.__cause__)


def get_sqlstate(error: BaseException) -> str | None:
    """Return the SQLSTATE carried by an error or by anything that caused it.

    Reads ``sqlstate`` (psycopg 3, asyncpg) and ``pgcode`` (psycopg2), looking
    through SQLAlchemy's wrapped ``orig`` and ``raise ... from`` causes.

    Args:
        error: The exception to inspect

    Returns:
        The five-character SQLSTATE, or None if nothing in the chain has one
    """
    for current in _error_chain(error):
        for attr in ("sqlstate", "pgcode"):
            code = getattr(current, attr, None)
            if isinstance(code, str) and len(code) == 5:
                return code.upper()
    return None


@functools.cache
def _driver_error_types() -> tuple[type[BaseException], ...]:
    """Database driver error types, including optional drivers when installed."""
    types: list[type[BaseException]] = [psycopg.Error, DBAPIError]
    for module_name, attr in (("psycopg2", "Error"), ("asyncpg", "PostgresError")):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        types.append(getattr(module, attr))
    return tuple(types)


def _describe(error: BaseException) -> str:
    """One-line description for logs: driver error type, SQLSTATE, first message line.

    When an application wraps a driver error, this walks __cause__ and __context__
    to find the innermost driver error and describes that instead of the wrapper.
    """
    driver_error = error
    # The chain runs from the outermost error inward, so the last match is the innermost
    for candidate in _error_chain(error):
        if isinstance(candidate, _driver_error_types()):
            driver_error = candidate
    code = get_sqlstate(error)
    lines = str(driver_error).strip().splitlines()
    summary = lines[0] if lines else ""
    where = f" (SQLSTATE {code})" if code else ""
    return f"{type(driver_error).__name__}{where}: {summary}"


def _as_codes(values: Iterable[str], field: str) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError(f"{field} must be a collection of codes, not a single string")
    return frozenset(str(value).upper() for value in values)


def _as_phrases(values: Iterable[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field} must be a collection of phrases, not a single string")
    return tuple(str(value).lower() for value in values)


@dataclass(frozen=True)
class RetryPolicy:
    """Which CockroachDB errors to retry, and how.

    Every field can be changed. ``RetryPolicy(**mapping)`` accepts plain lists
    for the code and phrase fields, so policies can come from YAML or
    environment variables. Use ``dataclasses.replace`` to derive a policy.

    Attributes:
        max_attempts: Total attempts, including the first (default: 5)
        initial_backoff: Delay before the first retry, in seconds (default: 0.1)
        max_backoff: Upper bound for any delay, in seconds (default: 10.0)
        backoff_multiplier: Growth factor between delays (default: 2.0)
        jitter: Fraction of each delay that is randomized, 0 to 1 (default: 0.5)
        retryable_sqlstates: SQLSTATEs to retry (default: RETRYABLE_SQLSTATES)
        retryable_sqlstate_classes: Two-character SQLSTATE classes to retry
            (default: RETRYABLE_SQLSTATE_CLASSES)
        uncategorized_sqlstates: Codes too generic to decide on; message
            phrases decide instead (default: UNCATEGORIZED_SQLSTATES)
        transient_messages: Phrases that mark an error without a usable code
            as transient (default: TRANSIENT_MESSAGES)
        non_transient_messages: Phrases that rule out a retry for errors
            without a usable code (default: NON_TRANSIENT_MESSAGES)
        retry_on: Custom predicate that replaces the rules above
        attempt_timeout: Seconds before a hung query or write is abandoned and
            retried; None disables the guard (default: None)
        ddl_attempt_timeout: Same guard, bounds each schema change (default: None)

    Example:
        ```python
        # Ride through multi-region failovers: also retry statement timeouts
        # and cut off attempts that hang for more than 10 seconds.
        policy = RetryPolicy(
            retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"},
            attempt_timeout=10,
        )
        ```
    """

    max_attempts: int = 5
    initial_backoff: float = 0.1
    max_backoff: float = 10.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.5
    retryable_sqlstates: frozenset[str] = RETRYABLE_SQLSTATES
    retryable_sqlstate_classes: frozenset[str] = RETRYABLE_SQLSTATE_CLASSES
    uncategorized_sqlstates: frozenset[str] = UNCATEGORIZED_SQLSTATES
    transient_messages: tuple[str, ...] = TRANSIENT_MESSAGES
    non_transient_messages: tuple[str, ...] = NON_TRANSIENT_MESSAGES
    retry_on: Callable[[BaseException], bool] | None = None
    attempt_timeout: float | None = None
    ddl_attempt_timeout: float | None = None

    def __post_init__(self) -> None:
        self._validate_max_attempts()
        self._validate_numeric_fields()
        self._validate_timeouts()
        self._normalize_collections()

    def _validate_max_attempts(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be an integer of at least 1")

    def _validate_numeric_fields(self) -> None:
        for name in ("initial_backoff", "max_backoff", "backoff_multiplier", "jitter"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, not NaN or infinity")
        if self.initial_backoff < 0:
            raise ValueError("initial_backoff must not be negative")
        if self.max_backoff < self.initial_backoff:
            raise ValueError("max_backoff must be at least initial_backoff")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if isinstance(self.jitter, bool):
            raise TypeError("jitter is a ratio from 0 to 1, for example 0.5 or 0")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")

    def _validate_timeouts(self) -> None:
        for name in ("attempt_timeout", "ddl_attempt_timeout"):
            value = getattr(self, name)
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite, not NaN or infinity")
                if value <= 0:
                    raise ValueError(f"{name} must be positive or None")

    def _normalize_collections(self) -> None:
        for name in (
            "retryable_sqlstates",
            "retryable_sqlstate_classes",
            "uncategorized_sqlstates",
        ):
            object.__setattr__(self, name, _as_codes(getattr(self, name), name))
        for name in ("transient_messages", "non_transient_messages"):
            object.__setattr__(self, name, _as_phrases(getattr(self, name), name))

    def is_retryable(self, error: BaseException) -> bool:
        """Return True if ``error`` is worth retrying under this policy."""
        if self.retry_on is not None:
            return bool(self.retry_on(error))
        code = get_sqlstate(error)
        if code is not None:
            if code in self.retryable_sqlstates or code[:2] in self.retryable_sqlstate_classes:
                return True
            if code not in self.uncategorized_sqlstates:
                return False
        chain = list(_error_chain(error))
        if not any(isinstance(e, _driver_error_types()) for e in chain):
            return False
        if any(isinstance(e, (PoolTimeout, PoolCheckoutTimeout)) for e in chain):
            return False
        # SQLAlchemy embeds the statement and its parameters in str(StatementError)
        # and str(DBAPIError), which can hold user data, so only driver and
        # application messages count for phrase matching.
        messages = [str(e).lower() for e in chain if not isinstance(e, StatementError)]
        if any(phrase in m for m in messages for phrase in self.non_transient_messages):
            return False
        if any(getattr(e, "connection_invalidated", False) for e in chain):
            return True
        return any(phrase in m for m in messages for phrase in self.transient_messages)

    def delay(self, retry_number: int, *, rand: Callable[[], float] = random.random) -> float:
        """Seconds to wait before retry ``retry_number`` (0 for the first retry)."""
        exponent = min(max(retry_number, 0), 64)
        try:
            base = min(self.initial_backoff * self.backoff_multiplier**exponent, self.max_backoff)
        except OverflowError:
            base = self.max_backoff
        return base * (1 - self.jitter * rand())

    def run(
        self,
        fn: Callable[[], T],
        *,
        label: str = "operation",
        can_retry: Callable[[BaseException], bool] | None = None,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> T:
        """Call ``fn`` until it succeeds or fails with an error not worth retrying.

        Args:
            fn: The attempt; called again from scratch on every retry
            label: What the attempt does, used in log messages
            can_retry: Extra check; when it returns False the error is raised
            sleep: Sleep function (tests pass a fake)

        Returns:
            Whatever ``fn`` returns
        """
        attempt = 1
        while True:
            try:
                return fn()
            except Exception as error:
                if not self._should_retry(error, attempt, label, can_retry):
                    raise
                pause = self.delay(attempt - 1)
                self._log_retry(error, attempt, pause, label)
                sleep(pause)
                attempt += 1

    async def arun(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        label: str = "operation",
        can_retry: Callable[[BaseException], bool] | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> T:
        """Async version of :meth:`run`; ``fn`` returns a new awaitable per attempt."""
        attempt = 1
        while True:
            try:
                return await fn()
            except Exception as error:
                if not self._should_retry(error, attempt, label, can_retry):
                    raise
                pause = self.delay(attempt - 1)
                self._log_retry(error, attempt, pause, label)
                await sleep(pause)
                attempt += 1

    def _should_retry(
        self,
        error: BaseException,
        attempt: int,
        label: str,
        can_retry: Callable[[BaseException], bool] | None,
    ) -> bool:
        if not self.is_retryable(error):
            return False
        if can_retry is not None and not can_retry(error):
            return False
        if attempt >= self.max_attempts:
            logger.warning(
                "Giving up on %s after %d attempts: %s", label, attempt, _describe(error)
            )
            return False
        return True

    def _log_retry(self, error: BaseException, attempt: int, pause: float, label: str) -> None:
        logger.warning(
            "Retrying %s after %s (attempt %d of %d failed, next try in %.2fs)",
            label,
            _describe(error),
            attempt,
            self.max_attempts,
            pause,
        )


DEFAULT_RETRY_POLICY = RetryPolicy()


def is_retryable_error(error: BaseException) -> bool:
    """Return True if ``error`` is worth retrying under the default policy."""
    return DEFAULT_RETRY_POLICY.is_retryable(error)


def resolve_retry_policy(
    policy: RetryPolicy | None,
    *,
    base: RetryPolicy,
    max_attempts: int | None = None,
    initial_backoff: float | None = None,
    max_backoff: float | None = None,
    backoff_multiplier: float | None = None,
    jitter: bool | None = None,
) -> RetryPolicy:
    """Combine an explicit policy, or flat ``retry_*`` arguments, with a base policy.

    Raises:
        ValueError: If both a policy and flat arguments are given
        TypeError: If jitter is not bool or None
    """
    if jitter is not None and not isinstance(jitter, bool):
        raise TypeError(
            "jitter must be True, False, or None; for a ratio, use RetryPolicy(jitter=...)"
        )
    overrides: dict[str, Any] = {
        "max_attempts": max_attempts,
        "initial_backoff": initial_backoff,
        "max_backoff": max_backoff,
        "backoff_multiplier": backoff_multiplier,
        "jitter": None if jitter is None else (0.5 if jitter else 0.0),
    }
    overrides = {name: value for name, value in overrides.items() if value is not None}
    if policy is not None:
        if overrides:
            raise ValueError("Pass either retry_policy or retry_* arguments, not both")
        return policy
    return dataclasses.replace(base, **overrides) if overrides else base


def _decorator_policy(
    max_retries: int | None,
    initial_backoff: float | None,
    max_backoff: float | None,
    backoff_multiplier: float | None,
    jitter: bool | None,
    retry_on: Callable[[BaseException], bool] | None,
    policy: RetryPolicy | None,
) -> RetryPolicy:
    if policy is not None and retry_on is not None:
        raise ValueError("Pass either policy or retry_on, not both")
    if max_retries is not None and max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    flat_args = [max_retries, initial_backoff, max_backoff, backoff_multiplier, jitter]
    if policy is not None and any(arg is not None for arg in flat_args):
        raise ValueError("Pass either policy or max_retries / backoff parameters, not both")
    resolved = resolve_retry_policy(
        policy,
        base=DEFAULT_RETRY_POLICY,
        max_attempts=max_retries,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
    )
    return dataclasses.replace(resolved, retry_on=retry_on) if retry_on is not None else resolved


def async_retry_with_backoff(
    max_retries: int | None = None,
    initial_backoff: float | None = None,
    max_backoff: float | None = None,
    backoff_multiplier: float | None = None,
    jitter: bool | None = None,
    *,
    retry_on: Callable[[BaseException], bool] | None = None,
    policy: RetryPolicy | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry an async function on transient CockroachDB errors.

    Unset arguments fall back to ``RetryPolicy`` defaults (5 attempts, 0.1s
    initial backoff, 10s cap, doubling, jitter on). The whole function runs
    again on each attempt, so it should open its own transaction.

    After 40003 or a lost connection the work may already have committed:
    only decorate idempotent work, or pass
    ``retry_on=lambda e: get_sqlstate(e) == "40001"``.

    Args:
        max_retries: Total attempts, including the first
        initial_backoff: Delay before the first retry, in seconds
        max_backoff: Upper bound for any delay, in seconds
        backoff_multiplier: Growth factor between delays
        jitter: Randomize delays (True maps to a 0.5 ratio)
        retry_on: Custom predicate deciding which errors to retry
        policy: A complete RetryPolicy, instead of the arguments above

    Example:
        ```python
        @async_retry_with_backoff(max_retries=3)
        async def upsert(conn):
            await conn.execute(text("UPSERT INTO t VALUES (1, 'a')"))
        ```
    """
    resolved = _decorator_policy(
        max_retries, initial_backoff, max_backoff, backoff_multiplier, jitter, retry_on, policy
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        label = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await resolved.arun(lambda: func(*args, **kwargs), label=label)

        return wrapper

    return decorator


def sync_retry_with_backoff(
    max_retries: int | None = None,
    initial_backoff: float | None = None,
    max_backoff: float | None = None,
    backoff_multiplier: float | None = None,
    jitter: bool | None = None,
    *,
    retry_on: Callable[[BaseException], bool] | None = None,
    policy: RetryPolicy | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function on transient CockroachDB errors.

    Synchronous version of :func:`async_retry_with_backoff`, with the same
    arguments and the same idempotency caveat.
    """
    resolved = _decorator_policy(
        max_retries, initial_backoff, max_backoff, backoff_multiplier, jitter, retry_on, policy
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        label = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return resolved.run(lambda: func(*args, **kwargs), label=label)

        return wrapper

    return decorator
