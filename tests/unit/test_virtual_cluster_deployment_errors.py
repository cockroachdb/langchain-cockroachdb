"""Offline checks for credential-safe live-deployment failures."""

import psycopg
import pytest

from tests.integration.test_virtual_cluster_deployment import (
    VirtualClusterTargets,
    hide_connection_errors,
)

SENSITIVE_URL = "postgresql://test:private-password@example.invalid/test"


def assert_redacted(error: pytest.fail.Exception, error_type: type[psycopg.Error]) -> None:
    expected_state = error_type.sqlstate or "unavailable"
    assert str(error) == (
        f"Virtual-cluster operation failed: {error_type.__name__}; "
        f"SQLSTATE={expected_state}. Connection details omitted."
    )
    assert SENSITIVE_URL not in str(error)
    assert error.__suppress_context__
    assert not error.pytrace


@pytest.mark.parametrize("error_type", [psycopg.OperationalError, psycopg.errors.InvalidPassword])
def test_sync_deployment_failure_omits_driver_arguments(error_type: type[psycopg.Error]) -> None:
    @hide_connection_errors
    def failing_operation() -> None:
        raise error_type(SENSITIVE_URL)

    with pytest.raises(pytest.fail.Exception) as caught:
        failing_operation()
    assert_redacted(caught.value, error_type)


@pytest.mark.parametrize("error_type", [psycopg.OperationalError, psycopg.errors.InvalidPassword])
async def test_async_deployment_failure_omits_driver_arguments(
    error_type: type[psycopg.Error],
) -> None:
    @hide_connection_errors
    async def failing_operation() -> None:
        raise error_type(SENSITIVE_URL)

    with pytest.raises(pytest.fail.Exception) as caught:
        await failing_operation()
    assert_redacted(caught.value, error_type)


def test_deployment_data_assertions_remain_visible() -> None:
    @hide_connection_errors
    def wrong_checkpoint() -> None:
        raise AssertionError("checkpoint belongs to peer")

    with pytest.raises(AssertionError, match="checkpoint belongs to peer"):
        wrong_checkpoint()


def test_deployment_target_repr_omits_urls() -> None:
    assert repr(VirtualClusterTargets(SENSITIVE_URL, SENSITIVE_URL)) == "VirtualClusterTargets()"
