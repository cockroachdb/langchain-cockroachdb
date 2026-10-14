"""Small SQL helpers shared by the vector store, chat history and indexes."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, *, what: str) -> str:
    """Return ``name`` if it is a plain SQL identifier, otherwise raise ValueError.

    Table, schema, column and index names are interpolated into SQL, so they
    are limited to letters, digits and underscores.
    """
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid {what} {name!r}: use letters, digits and underscores, "
            "starting with a letter or underscore"
        )
    return name


def vector_literal(values: Sequence[float]) -> str:
    """Format an embedding as CockroachDB's VECTOR text form, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def parse_vector(value: Any) -> list[float]:
    """Turn a VECTOR value returned by the driver into a list of floats."""
    if isinstance(value, str):
        return [float(item) for item in json.loads(value)]
    return [float(item) for item in value]
