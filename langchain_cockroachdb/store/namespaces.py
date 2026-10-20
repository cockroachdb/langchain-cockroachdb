"""Namespace handling for the CockroachDB store.

Namespaces are stored as dot-separated text. Labels are non-empty strings without dots.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from langgraph.store.base import InvalidNamespaceError


def validate_labels(namespace: Sequence[str]) -> tuple[str, ...]:
    """Validate namespace labels.

    Args:
        namespace: Sequence of namespace labels.

    Returns:
        Validated namespace as a tuple.

    Raises:
        InvalidNamespaceError: If any label is invalid.
    """
    for label in namespace:
        if not isinstance(label, str):
            raise InvalidNamespaceError(
                f"Invalid namespace {namespace}: all labels must be strings"
            )
        if not label:
            raise InvalidNamespaceError(
                f"Invalid namespace {namespace}: labels cannot be empty strings"
            )
        if "." in label:
            raise InvalidNamespaceError(f"Invalid namespace {namespace}: labels cannot contain '.'")

    return tuple(namespace)


def to_text(namespace: Sequence[str]) -> str:
    """Convert namespace tuple to dot-separated text.

    Args:
        namespace: Namespace labels.

    Returns:
        Dot-separated namespace string.
    """
    return ".".join(namespace)


def from_text(prefix: str) -> tuple[str, ...]:
    """Convert dot-separated text to namespace tuple.

    Args:
        prefix: Dot-separated namespace string.

    Returns:
        Namespace labels as a tuple.
    """
    return tuple(prefix.split("."))


def escape_like(text: str) -> str:
    """Escape LIKE pattern metacharacters.

    Args:
        text: Text to escape.

    Returns:
        Text with backslash, percent, and underscore escaped.
    """
    # Escape backslash first, then percent and underscore
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def prefix_condition(column: str, namespace_prefix: Sequence[str]) -> tuple[str, list[str]]:
    """Build a SQL condition for namespace prefix matching.

    Args:
        column: Column name (validated identifier, already formatted).
        namespace_prefix: Namespace prefix to match.

    Returns:
        Tuple of (SQL condition, parameter values).
        Empty prefix returns ("TRUE", []).
        Non-empty returns a condition with exact match OR prefix match.
    """
    if not namespace_prefix:
        return ("TRUE", [])

    text = to_text(namespace_prefix)
    escaped = escape_like(text)

    return (f"({column} = %s OR {column} LIKE %s ESCAPE '\\')", [text, f"{escaped}.%"])


def match_pattern(path: Sequence[str], match_type: Literal["prefix", "suffix"]) -> str:
    """Build a regex pattern for namespace path matching.

    Matches PostgresStore behavior: "*" matches one label, other labels are escaped.

    Args:
        path: Path components ("*" is a wildcard matching one label).
        match_type: "prefix" or "suffix".

    Returns:
        Regex pattern string.
    """
    # Build pattern parts
    parts = []
    for component in path:
        if component == "*":
            parts.append(r"[^.]+")
        else:
            parts.append(re.escape(component))

    pattern = r"\.".join(parts)

    if match_type == "prefix":
        return rf"^{pattern}(\.|\z)"
    else:  # suffix
        return rf"(^|\.){pattern}\z"


def truncate_expression(column: str, max_depth: int) -> str:
    """Build a SQL expression to truncate namespace to max depth.

    Args:
        column: Column name (validated identifier, already formatted).
        max_depth: Maximum namespace depth (positive integer).

    Returns:
        SQL regexp_replace expression.

    Raises:
        ValueError: If max_depth is not a positive integer.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")

    # For max_depth N, match N labels: (label.){N-1}label
    # Quantifier is {N-1}, which is {0} for max_depth=1
    quantifier = max_depth - 1

    return f"regexp_replace({column}, '^((?:[^.]+\\.){{{quantifier}}}[^.]+).*$', '\\1')"
