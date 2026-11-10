"""Checkpoint serializers for the store's unit tests."""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def strict_serializer(allowed: Iterable[tuple[str, ...]] = ()) -> JsonPlusSerializer:
    """A JsonPlusSerializer that rebuilds its built-in safe types and ``allowed``, and no other.

    LangGraph's strict mode (LANGGRAPH_STRICT_MSGPACK=true) runs this: the
    serializer allows nothing beyond its safe types
    (``allowed_msgpack_modules=None``), and a graph adds what its state
    schema names with ``with_msgpack_allowlist``. Any other type comes back
    raw. langgraph-checkpoint has the allowlist from 4.0.1. Older releases
    rebuild every type, so a test that needs a strict serializer skips there.
    """
    if "allowed_msgpack_modules" not in inspect.signature(JsonPlusSerializer).parameters:
        pytest.skip("langgraph-checkpoint before 4.0.1 has no msgpack allowlist")
    return JsonPlusSerializer(allowed_msgpack_modules=None).with_msgpack_allowlist(allowed)
