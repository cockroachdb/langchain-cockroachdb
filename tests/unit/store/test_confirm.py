"""Unit tests for confirm planning and validation."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from langgraph.store.base import NOT_PROVIDED, Item

from langchain_cockroachdb import CockroachDBStore
from langchain_cockroachdb.store.reconcile import MemoryRecord


def test_validate_confirm_ttl_accepts_valid_values() -> None:
    """Valid ttl values are accepted without raising."""
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    # NOT_PROVIDED is valid
    store._validate_confirm_ttl(NOT_PROVIDED)
    # None is valid
    store._validate_confirm_ttl(None)
    # Positive numbers are valid
    store._validate_confirm_ttl(10.0)
    store._validate_confirm_ttl(1)
    # Callables are valid
    store._validate_confirm_ttl(lambda r: 60.0)


def test_validate_confirm_ttl_rejects_invalid_values() -> None:
    """Invalid ttl values raise ValueError before any I/O."""
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    # String is invalid
    with pytest.raises(ValueError, match="ttl must be"):
        store._validate_confirm_ttl("10")  # type: ignore[arg-type]
    # Zero is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._validate_confirm_ttl(0)
    # Negative is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._validate_confirm_ttl(-5.0)
    # Infinity is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._validate_confirm_ttl(math.inf)
    # NaN is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._validate_confirm_ttl(math.nan)


def test_resolve_confirm_ttl_returns_non_callable_as_is() -> None:
    """Non-callable ttl values are returned unchanged."""
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    record = MemoryRecord(
        item=Item(
            namespace=["user", "alice"],
            key="k",
            value={"text": "test"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        version=1,
        provenance=None,
        valid_at=datetime.now(timezone.utc),
        confirmed_at=None,
        confirmations=0,
    )
    assert store._resolve_confirm_ttl(NOT_PROVIDED, record) is NOT_PROVIDED
    assert store._resolve_confirm_ttl(None, record) is None
    assert store._resolve_confirm_ttl(10.0, record) == 10.0


def test_resolve_confirm_ttl_calls_callable_and_validates() -> None:
    """Callable ttl is called with the record and its result is validated."""
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    record = MemoryRecord(
        item=Item(
            namespace=["user", "alice"],
            key="k",
            value={"text": "test"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        version=1,
        provenance=None,
        valid_at=datetime.now(timezone.utc),
        confirmed_at=None,
        confirmations=2,
    )
    # Valid return: number
    result = store._resolve_confirm_ttl(lambda r: r.confirmations * 60.0, record)
    assert result == 120.0
    # Valid return: None
    result = store._resolve_confirm_ttl(lambda r: None, record)
    assert result is None


def test_resolve_confirm_ttl_rejects_invalid_callable_returns() -> None:
    """Callable returning invalid values raises ValueError."""
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    record = MemoryRecord(
        item=Item(
            namespace=["user", "alice"],
            key="k",
            value={"text": "test"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        version=1,
        provenance=None,
        valid_at=datetime.now(timezone.utc),
        confirmed_at=None,
        confirmations=0,
    )
    # String return is invalid
    with pytest.raises(ValueError, match="Callable ttl must return"):
        store._resolve_confirm_ttl(lambda r: "10", record)  # type: ignore[return-value]
    # Zero is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._resolve_confirm_ttl(lambda r: 0, record)
    # Negative is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._resolve_confirm_ttl(lambda r: -5.0, record)
    # Infinity is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._resolve_confirm_ttl(lambda r: math.inf, record)
    # NaN is invalid
    with pytest.raises(ValueError, match="positive finite"):
        store._resolve_confirm_ttl(lambda r: math.nan, record)


def test_confirm_write_id_check_builds_correct_statement() -> None:
    """The write_id check statement filters for live rows with the write_id."""
    store = CockroachDBStore(object())  # type: ignore[arg-type]
    sql, params = store._confirm_write_id_check("user.alice", "key-1", "write-id-123")
    # SQL should check for the prefix, key, live row, and write_id
    assert "prefix = %s" in sql
    assert "key = %s" in sql
    assert "expires_at IS NULL OR expires_at > now()" in sql
    assert "write_id = CAST(%s AS UUID)" in sql
    assert params == ["user.alice", "key-1", "write-id-123"]
