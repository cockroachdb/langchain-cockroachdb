"""Unit tests for retry classification and RetryPolicy."""

import asyncio
import logging

import psycopg
import pytest
from psycopg import errors as pg_errors
from psycopg_pool import PoolTimeout
from sqlalchemy.exc import OperationalError as SAOperationalError
from sqlalchemy.exc import TimeoutError as SAPoolTimeout

from langchain_cockroachdb.retry import (
    DEFAULT_RETRY_POLICY,
    RETRYABLE_SQLSTATES,
    RetryPolicy,
    async_retry_with_backoff,
    get_sqlstate,
    is_retryable_error,
    resolve_retry_policy,
    sync_retry_with_backoff,
)

RETRYABLE = [
    "40001",
    "40003",
    "57P01",
    "58C00",
    "58C01",
    "53300",
    "08000",
    "08001",
    "08003",
    "08004",
    "08006",
    "08007",
    "08P01",
    "08C00",
]
NOT_RETRYABLE = [
    "23505",
    "42P01",
    "22P02",
    "57014",
    "40002",
    "40P01",
    "XXA00",
    "28P01",
    "25P02",
    "42501",
]


def server_error(code: str, message: str = "boom") -> psycopg.Error:
    """A psycopg error carrying ``code``, as the driver raises it for server errors."""
    try:
        error = pg_errors.lookup(code)(message)
    except KeyError:
        error = psycopg.OperationalError(message)
    error.sqlstate = code  # psycopg sets this per instance for codes it has no class for
    return error


def wrapped(orig: BaseException, **kwargs) -> SAOperationalError:
    return SAOperationalError("SELECT 1", {}, orig, **kwargs)


def chained(orig: BaseException) -> BaseException:
    try:
        try:
            raise orig
        except Exception as inner:
            raise RuntimeError("application wrapper") from inner
    except RuntimeError as outer:
        return outer


SHAPES = {"raw": lambda e: e, "wrapped": wrapped, "chained": chained}


class TestClassification:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("code", RETRYABLE)
    def test_retryable_codes(self, code: str, shape: str) -> None:
        assert is_retryable_error(SHAPES[shape](server_error(code)))

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("code", NOT_RETRYABLE)
    def test_definite_codes_are_not_retried(self, code: str, shape: str) -> None:
        assert not is_retryable_error(SHAPES[shape](server_error(code)))

    @pytest.mark.parametrize(
        "message",
        [
            'relation "connection_logs" does not exist',
            "query execution canceled due to statement timeout",
        ],
    )
    def test_code_beats_transient_looking_text(self, message: str) -> None:
        code = "42P01" if "relation" in message else "57014"
        assert not is_retryable_error(wrapped(server_error(code, message)))

    @pytest.mark.parametrize(
        "message",
        [
            "consuming input failed: server closed the connection unexpectedly",
            'connection failed: connection to server at "10.0.0.1", port 26257 failed: Connection refused',
            "connection timeout expired",
            "the connection is lost",
            "connection socket closed",
            "SSL SYSCALL error: EOF detected",
            "failed to resolve host 'db': [Errno 8] nodename nor servname provided, or not known",
            "server is not accepting clients, try another node",
            "sorry, too many clients already",
        ],
    )
    def test_codeless_connection_loss_is_retried(self, message: str) -> None:
        assert is_retryable_error(psycopg.OperationalError(message))
        assert is_retryable_error(wrapped(psycopg.OperationalError(message)))

    @pytest.mark.parametrize(
        "message",
        [
            'connection failed: connection to server at "127.0.0.1", port 26257 failed: '
            "ERROR:  password authentication failed for user bob",
            "connection failed: server does not support SSL, but SSL was required",
            "SSL error: certificate verify failed",
            'invalid connection option "nope"',
            'database "missing" does not exist',
            "permission denied for database app",
        ],
    )
    def test_codeless_configuration_errors_are_not_retried(self, message: str) -> None:
        assert not is_retryable_error(psycopg.OperationalError(message))

    def test_invalidated_connection_is_retried(self) -> None:
        error = wrapped(psycopg.OperationalError("unusual wording"), connection_invalidated=True)
        assert is_retryable_error(error)

    def test_parameters_in_sqlalchemy_message_do_not_veto(self) -> None:
        orig = psycopg.OperationalError(
            "consuming input failed: server closed the connection unexpectedly"
        )
        error = SAOperationalError("INSERT ...", {"c0": "certificate does not exist"}, orig)
        assert is_retryable_error(error)

    def test_pool_timeouts_are_not_retried(self) -> None:
        assert not is_retryable_error(PoolTimeout("couldn't get a connection after 30.00 sec"))
        assert not is_retryable_error(
            SAPoolTimeout("QueuePool limit reached, connection timed out")
        )

    @pytest.mark.parametrize(
        "error",
        [ValueError("connection refused"), RuntimeError("restart transaction"), Exception("40001")],
    )
    def test_plain_exceptions_are_never_retried_on_text(self, error: Exception) -> None:
        assert not is_retryable_error(error)

    def test_uncategorized_code_falls_back_to_messages(self) -> None:
        assert is_retryable_error(
            server_error("XXUUU", "initial connection heartbeat failed: dial tcp")
        )
        assert not is_retryable_error(server_error("XXUUU", "something unrelated"))

    def test_canceled_session_i_o_timeout_is_retryable(self) -> None:
        """CockroachDB v25.4 sends XXUUU + 'i/o timeout' when a session is canceled."""
        error = psycopg.InternalError("read tcp 10.88.0.35:26257->10.88.0.35:42564: i/o timeout")
        error.sqlstate = "XXUUU"
        assert is_retryable_error(error)
        assert is_retryable_error(wrapped(error))

    def test_psycopg3_connection_is_closed_is_retryable(self) -> None:
        """psycopg 3 reports 'the connection is closed' on a dead connection."""
        assert is_retryable_error(psycopg.OperationalError("the connection is closed"))

    def test_definite_code_beats_i_o_timeout_message(self) -> None:
        """A definite (non-uncategorized) code vetoes retry even with 'i/o timeout' text."""
        assert not is_retryable_error(
            server_error("42601", "read tcp 10.0.0.1:26257->10.0.0.2:5432: i/o timeout")
        )

    def test_definite_code_beats_transient_message(self) -> None:
        """Definite non-retryable code (not uncategorized) vetoes even with transient text."""
        error = wrapped(
            server_error(
                "XXA00",
                "transaction committed but schema change aborted with error: failed to connect to n3",
            )
        )
        assert not is_retryable_error(error)

    def test_non_transient_phrase_vetoes_before_invalidation(self) -> None:
        """Non-transient veto applies before checking connection_invalidated."""
        error = wrapped(
            psycopg.OperationalError("SSL error: certificate verify failed"),
            connection_invalidated=True,
        )
        assert not is_retryable_error(error)

    def test_non_transient_phrase_vetoes_multi_host_password_failure(self) -> None:
        """Password auth failure in multi-host message vetoes retry."""
        error = psycopg.OperationalError(
            'connection failed: connection to server at "10.0.0.1", port 26257 failed: '
            "Connection refused; "
            'connection to server at "10.0.0.2", port 26257 failed: '
            "ERROR: password authentication failed for user bob"
        )
        assert not is_retryable_error(error)

    def test_pool_timeout_with_transient_wrapper(self) -> None:
        """Pool timeout veto applies even under a wrapper with transient text."""
        try:
            raise PoolTimeout("couldn't get a connection after 30.00 sec")
        except PoolTimeout as pool_error:
            wrapper = RuntimeError("connection timed out while waiting for the pool")
            wrapper.__cause__ = pool_error
            assert not is_retryable_error(wrapper)


class TestGetSqlstate:
    def test_reads_psycopg2_style_pgcode(self) -> None:
        class Psycopg2Like(Exception):
            pgcode = "40001"

        assert get_sqlstate(Psycopg2Like()) == "40001"

    def test_follows_orig_and_cause(self) -> None:
        assert get_sqlstate(wrapped(server_error("40003"))) == "40003"
        assert get_sqlstate(chained(server_error("57P01"))) == "57P01"

    def test_ignores_implicit_context(self) -> None:
        try:
            try:
                raise server_error("40001")
            except psycopg.Error:
                raise ValueError("unrelated")  # noqa: B904
        except ValueError as error:
            assert get_sqlstate(error) is None

    def test_cycles_terminate(self) -> None:
        a, b = RuntimeError("a"), RuntimeError("b")
        a.__cause__, b.__cause__ = b, a
        assert get_sqlstate(a) is None


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert (policy.max_attempts, policy.initial_backoff, policy.max_backoff) == (5, 0.1, 10.0)
        assert (policy.backoff_multiplier, policy.jitter) == (2.0, 0.5)
        assert policy.retryable_sqlstates == RETRYABLE_SQLSTATES
        assert policy.attempt_timeout is None and policy.ddl_attempt_timeout is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_attempts": 0},
            {"max_attempts": True},
            {"initial_backoff": -1},
            {"initial_backoff": 1.0, "max_backoff": 0.5},
            {"backoff_multiplier": 0.5},
            {"jitter": 1.5},
            {"jitter": True},
            {"attempt_timeout": 0},
            {"ddl_attempt_timeout": -1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict) -> None:
        with pytest.raises((ValueError, TypeError)):
            RetryPolicy(**kwargs)

    def test_rejects_single_string_for_collections(self) -> None:
        with pytest.raises(TypeError):
            RetryPolicy(retryable_sqlstates="40001")

    def test_accepts_plain_data(self) -> None:
        policy = RetryPolicy(
            **{
                "max_attempts": 8,
                "retryable_sqlstates": ["40001", "57014"],
                "transient_messages": ["Custom Glitch"],
            }
        )
        assert policy.retryable_sqlstates == frozenset({"40001", "57014"})
        assert policy.transient_messages == ("custom glitch",)

    def test_statement_timeouts_are_opt_in(self) -> None:
        opted_in = RetryPolicy(retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"})
        assert opted_in.is_retryable(server_error("57014"))
        assert not RetryPolicy().is_retryable(server_error("57014"))

    def test_retry_on_replaces_the_rules(self) -> None:
        only_40001 = RetryPolicy(retry_on=lambda e: get_sqlstate(e) == "40001")
        assert only_40001.is_retryable(server_error("40001"))
        assert not only_40001.is_retryable(server_error("40003"))

    def test_custom_message_lists(self) -> None:
        policy = RetryPolicy(transient_messages=("custom glitch",))
        assert policy.is_retryable(psycopg.OperationalError("a Custom Glitch happened"))
        assert not policy.is_retryable(psycopg.OperationalError("connection refused"))

    def test_delay_grows_and_caps(self) -> None:
        policy = RetryPolicy(initial_backoff=0.1, max_backoff=0.5, backoff_multiplier=2, jitter=0)
        assert [policy.delay(n) for n in range(5)] == pytest.approx([0.1, 0.2, 0.4, 0.5, 0.5])

    def test_delay_jitter_bounds(self) -> None:
        policy = RetryPolicy(initial_backoff=1, max_backoff=1, jitter=0.5)
        assert policy.delay(0, rand=lambda: 0.0) == pytest.approx(1.0)
        assert policy.delay(0, rand=lambda: 1.0) == pytest.approx(0.5)

    def test_delay_does_not_overflow(self) -> None:
        assert RetryPolicy(max_backoff=10, jitter=0).delay(10_000) == pytest.approx(10.0)

    def test_run_retries_then_succeeds(self) -> None:
        calls, sleeps = [], []

        def fn() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise server_error("40001")
            return "ok"

        assert RetryPolicy(jitter=0).run(fn, label="t", sleep=sleeps.append) == "ok"
        assert len(calls) == 3
        assert sleeps == pytest.approx([0.1, 0.2])

    def test_run_gives_up_after_max_attempts(self, caplog: pytest.LogCaptureFixture) -> None:
        calls = []

        def fn() -> None:
            calls.append(1)
            raise server_error("40001")

        with (
            caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
            pytest.raises(psycopg.errors.SerializationFailure),
        ):
            RetryPolicy(max_attempts=3, jitter=0).run(
                fn, label="insert into t", sleep=lambda _: None
            )
        assert len(calls) == 3
        assert "Giving up on insert into t after 3 attempts" in caplog.text
        assert "40001" in caplog.text

    def test_run_does_not_retry_or_log_other_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        calls = []

        def fn() -> None:
            calls.append(1)
            raise ValueError("bad input")

        with (
            caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
            pytest.raises(ValueError),
        ):
            RetryPolicy().run(fn, sleep=lambda _: None)
        assert len(calls) == 1
        assert caplog.text == ""

    def test_can_retry_vetoes(self) -> None:
        calls = []

        def fn() -> None:
            calls.append(1)
            raise server_error("40001")

        with pytest.raises(psycopg.Error):
            RetryPolicy().run(fn, can_retry=lambda e: False, sleep=lambda _: None)
        assert len(calls) == 1

    def test_base_exceptions_pass_through(self) -> None:
        def fn() -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            RetryPolicy().run(fn, sleep=lambda _: None)

    async def test_arun_retries_then_succeeds(self) -> None:
        calls, sleeps = [], []

        async def fn() -> str:
            calls.append(1)
            if len(calls) < 2:
                raise wrapped(psycopg.OperationalError("x"), connection_invalidated=True)
            return "ok"

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        assert await RetryPolicy(jitter=0).arun(fn, sleep=fake_sleep) == "ok"
        assert sleeps == pytest.approx([0.1])

    async def test_arun_lets_cancellation_through(self) -> None:
        async def fn() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await RetryPolicy().arun(fn)


class TestResolveAndDecorators:
    def test_flat_arguments_override_the_base(self) -> None:
        policy = resolve_retry_policy(None, base=RetryPolicy(), max_attempts=7, jitter=False)
        assert (policy.max_attempts, policy.jitter) == (7, 0.0)
        assert resolve_retry_policy(None, base=RetryPolicy(), jitter=True).jitter == 0.5

    def test_policy_and_flat_arguments_conflict(self) -> None:
        with pytest.raises(ValueError):
            resolve_retry_policy(RetryPolicy(), base=RetryPolicy(), max_attempts=3)

    def test_no_overrides_returns_base(self) -> None:
        base = RetryPolicy(max_attempts=9)
        assert resolve_retry_policy(None, base=base) is base

    def test_default_policy_is_exported(self) -> None:
        assert RetryPolicy() == DEFAULT_RETRY_POLICY

    async def test_async_decorator_retries(self) -> None:
        calls = []

        @async_retry_with_backoff(max_retries=3, initial_backoff=0.001)
        async def fn() -> str:
            calls.append(1)
            if len(calls) < 2:
                raise server_error("40001")
            return "ok"

        assert await fn() == "ok"
        assert fn.__name__ == "fn"

    def test_sync_decorator_retry_on(self) -> None:
        calls = []

        @sync_retry_with_backoff(
            initial_backoff=0.001, retry_on=lambda e: get_sqlstate(e) == "40001"
        )
        def fn() -> None:
            calls.append(1)
            raise server_error("40003")

        with pytest.raises(psycopg.Error):
            fn()
        assert len(calls) == 1

    def test_decorator_rejects_zero_attempts(self) -> None:
        with pytest.raises(ValueError):
            sync_retry_with_backoff(max_retries=0)

    def test_decorator_rejects_policy_with_flat_arguments(self) -> None:
        with pytest.raises(ValueError):
            sync_retry_with_backoff(max_retries=3, policy=RetryPolicy())
        with pytest.raises(ValueError):
            sync_retry_with_backoff(policy=RetryPolicy(), retry_on=lambda e: True)

    def test_sync_decorator_on_functools_partial(self) -> None:
        """Decorating a functools.partial should not raise AttributeError."""
        import functools

        calls = []

        def base_fn(prefix: str) -> str:
            calls.append(1)
            if len(calls) < 2:
                raise server_error("40001")
            return f"{prefix}ok"

        partial_fn = functools.partial(base_fn, "result:")
        decorated = sync_retry_with_backoff(max_retries=3, initial_backoff=0.001)(partial_fn)

        assert decorated() == "result:ok"
        assert len(calls) == 2

    async def test_async_decorator_on_callable_instance(self) -> None:
        """Decorating a callable instance should not raise AttributeError."""

        class CallableTask:
            def __init__(self) -> None:
                self.calls = 0

            async def __call__(self) -> str:
                self.calls += 1
                if self.calls < 2:
                    raise server_error("40001")
                return "ok"

        task = CallableTask()
        decorated = async_retry_with_backoff(max_retries=3, initial_backoff=0.001)(task)

        assert await decorated() == "ok"
        assert task.calls == 2
