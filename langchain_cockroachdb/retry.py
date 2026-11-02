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

# Errors that roll a transaction back, so it did not commit:
# serialization_failure and deadlock_detected. After any other error, such
# as 40003 or a lost connection, a transaction that sent its COMMIT may have
# committed (may_have_committed).
DEFINITE_ABORT_SQLSTATES: frozenset[str] = frozenset({"40001", "40P01"})

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
    "unexpected eof while reading",
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

# CockroachDB v25.4 refuses to change a table's TTL settings while another
# schema change on that table runs. Several processes setting up the same
# tables at once then fail each other's TTL step. The other change finishes
# soon, so schema_change_policy retries an error with this SQLSTATE and one of
# these phrases. 0A000 alone means "feature not supported", so the phrase is
# required too.
SCHEMA_CHANGE_BUSY_SQLSTATE: str = "0A000"
SCHEMA_CHANGE_BUSY_MESSAGES: tuple[str, ...] = (
    "another schema change on the table is being processed",
)

# The busy error means waiting for the other schema change to finish, which
# takes far longer than a transient error takes to clear: a column backfill
# on a table with data, say, or any schema change on a multi-region cluster.
# So schema_change_policy gives it a budget of its own: this many attempts,
# or the policy's max_attempts if larger, with the policy's backoff growing
# up to this many seconds. Under the default policy that waits about a
# minute.
SCHEMA_CHANGE_BUSY_ATTEMPTS: int = 30
SCHEMA_CHANGE_BUSY_MAX_BACKOFF: float = 3.0

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


def may_have_committed(error: BaseException) -> bool:
    """Whether a transaction that sent its COMMIT, then failed with ``error``, may have committed.

    Only a definite abort, an error whose SQLSTATE is in
    DEFINITE_ABORT_SQLSTATES, proves that it did not. A 40003, a lost
    connection, an attempt the hang guard abandoned and every other error
    leave the outcome unknown, so they count as may have committed.

    Args:
        error: The exception that ended the transaction's attempt

    Returns:
        False only for a definite abort
    """
    return get_sqlstate(error) not in DEFINITE_ABORT_SQLSTATES


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

    When an application wraps a driver error, this follows the chain that
    ``_error_chain`` walks (SQLAlchemy's ``orig`` and explicit ``__cause__``,
    not ``__context__``) to the innermost driver error and describes that
    instead of the wrapper.

    A SQLAlchemy ``StatementError`` with no driver error inside is described
    by its class and SQLSTATE only, because its message can carry the SQL
    and its parameters.
    """
    driver_error = error
    # The chain runs from the outermost error inward, so the last match is the innermost
    for candidate in _error_chain(error):
        if isinstance(candidate, _driver_error_types()):
            driver_error = candidate
    code = get_sqlstate(error)
    where = f" (SQLSTATE {code})" if code else ""
    if isinstance(driver_error, StatementError):
        return f"{type(driver_error).__name__}{where}"
    lines = str(driver_error).strip().splitlines()
    summary = lines[0] if lines else ""
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
        # SQLAlchemy sets this after it discards a connection that broke. The
        # checkpointers set it when psycopg reports a pooled connection broken.
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
        failures: dict[bool, int] = {}
        while True:
            try:
                return fn()
            except Exception as error:
                pause = self._pause_before_retry(error, failures, label, can_retry)
                if pause is None:
                    raise
                sleep(pause)

    async def arun(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        label: str = "operation",
        can_retry: Callable[[BaseException], bool] | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> T:
        """Async version of :meth:`run`; ``fn`` returns a new awaitable per attempt."""
        failures: dict[bool, int] = {}
        while True:
            try:
                return await fn()
            except Exception as error:
                pause = self._pause_before_retry(error, failures, label, can_retry)
                if pause is None:
                    raise
                await sleep(pause)

    def _budget(self, error: BaseException) -> RetryPolicy:
        """The policy whose attempts and backoff apply to retrying ``error``.

        A plain policy applies its own to every error. schema_change_policy
        returns one that gives a busy schema change a budget of its own.
        """
        return self

    def _pause_before_retry(
        self,
        error: BaseException,
        failures: dict[bool, int],
        label: str,
        can_retry: Callable[[BaseException], bool] | None,
    ) -> float | None:
        """Seconds to wait before the next attempt after ``error``, or None to raise it.

        ``failures`` counts a run's failed attempts by budget: under the
        policy's own (True) or under the one _budget gives instead (False).
        The errors of one budget never use up the attempts of the other.
        """
        budget = self._budget(error)
        own = budget is self
        attempt = failures[own] = failures.get(own, 0) + 1
        if not self._should_retry(error, attempt, budget.max_attempts, label, can_retry):
            return None
        pause = budget.delay(attempt - 1)
        self._log_retry(error, attempt, budget.max_attempts, pause, label)
        return pause

    def _should_retry(
        self,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        label: str,
        can_retry: Callable[[BaseException], bool] | None,
    ) -> bool:
        if not self.is_retryable(error):
            return False
        if can_retry is not None and not can_retry(error):
            return False
        if attempt >= max_attempts:
            logger.warning(
                "Giving up on %s after %d attempts: %s", label, attempt, _describe(error)
            )
            return False
        return True

    def _log_retry(
        self, error: BaseException, attempt: int, max_attempts: int, pause: float, label: str
    ) -> None:
        logger.warning(
            "Retrying %s after %s (attempt %d of %d failed, next try in %.2fs)",
            label,
            _describe(error),
            attempt,
            max_attempts,
            pause,
        )


DEFAULT_RETRY_POLICY = RetryPolicy()


def is_schema_change_busy(
    error: BaseException, messages: Iterable[str] = SCHEMA_CHANGE_BUSY_MESSAGES
) -> bool:
    """True if ``error`` says another schema change on the table is still running.

    The error must carry SCHEMA_CHANGE_BUSY_SQLSTATE and one of ``messages``
    (default: SCHEMA_CHANGE_BUSY_MESSAGES), matched without regard to case.
    """
    if get_sqlstate(error) != SCHEMA_CHANGE_BUSY_SQLSTATE:
        return False
    phrases = _as_phrases(messages, "messages")
    return any(
        phrase in str(current).lower() for current in _error_chain(error) for phrase in phrases
    )


@dataclass(frozen=True)
class _SchemaChangePolicy(RetryPolicy):
    """A policy that retries a busy schema change under a budget of its own.

    schema_change_policy builds it from a policy, whose fields it keeps.
    ``busy_attempts`` and ``busy_max_backoff`` are that function's
    ``attempts`` and ``max_backoff``.
    """

    busy_messages: tuple[str, ...] = SCHEMA_CHANGE_BUSY_MESSAGES
    busy_attempts: int = SCHEMA_CHANGE_BUSY_ATTEMPTS
    busy_max_backoff: float = SCHEMA_CHANGE_BUSY_MAX_BACKOFF
    # The busy error's budget: these fields with its attempts and backoff cap
    _busy_budget: RetryPolicy = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "busy_messages", _as_phrases(self.busy_messages, "messages"))
        if (
            isinstance(self.busy_attempts, bool)
            or not isinstance(self.busy_attempts, int)
            or self.busy_attempts < 1
        ):
            raise ValueError("attempts must be an integer of at least 1")
        if not math.isfinite(self.busy_max_backoff) or self.busy_max_backoff < 0:
            raise ValueError("max_backoff must be finite and not negative")
        own = {field.name: getattr(self, field.name) for field in dataclasses.fields(RetryPolicy)}
        # max_attempts=1 turns every retry off, the busy error's included
        attempts = 1 if self.max_attempts == 1 else max(self.max_attempts, self.busy_attempts)
        budget = RetryPolicy(
            **{
                **own,
                "max_attempts": attempts,
                "max_backoff": max(self.initial_backoff, self.busy_max_backoff),
            }
        )
        object.__setattr__(self, "_busy_budget", budget)

    def _budget(self, error: BaseException) -> RetryPolicy:
        if is_schema_change_busy(error, self.busy_messages):
            return self._busy_budget
        return self


def schema_change_policy(
    policy: RetryPolicy,
    messages: Iterable[str] = SCHEMA_CHANGE_BUSY_MESSAGES,
    *,
    attempts: int = SCHEMA_CHANGE_BUSY_ATTEMPTS,
    max_backoff: float = SCHEMA_CHANGE_BUSY_MAX_BACKOFF,
) -> RetryPolicy:
    """``policy``, also retrying a schema change that met another running one.

    For idempotent schema changes, such as TTL settings, that several
    processes may run at once. The busy error (see is_schema_change_busy)
    is retried even when the policy's own ``retry_on`` would return False
    for it. It means waiting for the other schema change to finish, such as
    a column backfill, which takes longer than a transient error takes to
    clear. So it has a budget of its own: ``attempts`` attempts (default
    SCHEMA_CHANGE_BUSY_ATTEMPTS), or the policy's ``max_attempts`` if that
    is larger, with the policy's backoff growing up to ``max_backoff``
    seconds (default SCHEMA_CHANGE_BUSY_MAX_BACKOFF), or up to its
    ``initial_backoff`` if that is larger. Under the default policy that
    waits about a minute.

    Every other error is decided by ``policy`` as it is, including its
    ``retry_on``, and keeps its attempts and backoff. Those attempts are
    counted apart from the busy error's. Timeouts are the policy's, and
    ``max_attempts=1`` turns every retry off.

    Raises:
        ValueError: If ``attempts`` is not an integer of at least 1, or
            ``max_backoff`` is negative or not finite
    """
    phrases = _as_phrases(messages, "messages")

    def retry_on(error: BaseException) -> bool:
        return is_schema_change_busy(error, phrases) or policy.is_retryable(error)

    fields = {field.name: getattr(policy, field.name) for field in dataclasses.fields(RetryPolicy)}
    return _SchemaChangePolicy(
        **{**fields, "retry_on": retry_on},
        busy_messages=phrases,
        busy_attempts=attempts,
        busy_max_backoff=max_backoff,
    )


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
    jitter_name: str = "retry_jitter",
) -> RetryPolicy:
    """Combine an explicit policy, or flat ``retry_*`` arguments, with a base policy.

    ``jitter`` is an on or off switch: True or 1 means a ratio of 0.5, False
    or 0 means none. ``jitter_name`` is the argument's name in error messages.

    Raises:
        ValueError: If both a policy and flat arguments are given
        TypeError: If jitter is not None, True, False, 0 or 1
    """
    # The ints 0 and 1 are accepted so callers who passed them keep working
    if jitter is not None and not (
        isinstance(jitter, bool) or (type(jitter) is int and jitter in (0, 1))
    ):
        raise TypeError(
            f"{jitter_name} must be True, False, 0 or 1, not {jitter!r}; "
            "for a ratio, use RetryPolicy(jitter=...)"
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
        jitter_name="jitter",
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
