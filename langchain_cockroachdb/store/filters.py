"""Filter translation from LangGraph filter format to SQL conditions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_cockroachdb._sql import validate_identifier

# Supported comparison operators
_COMPARISON_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}


def translate_filter(
    filter: Mapping[str, Any] | None, *, value_column: str = "value"
) -> tuple[list[str], list[Any]]:
    """Translate a LangGraph filter to SQL conditions and parameters.

    Args:
        filter: Filter dict (MongoDB-style operators), or None.
        value_column: Column name for the JSONB value (default "value"), or "alias.column".

    Returns:
        Tuple of (conditions list, parameters list). All conditions are ANDed.

    Raises:
        ValueError: If filter contains unknown operators or invalid comparison values,
            or if value_column is not a valid identifier or qualified identifier.
    """
    # Validate value_column
    parts = value_column.split(".")
    if len(parts) > 2:
        raise ValueError(
            f"value_column must be an identifier or alias.identifier, got {value_column!r}"
        )
    for part in parts:
        validate_identifier(part, what="value_column part")

    if not filter:
        return ([], [])

    conditions: list[str] = []
    params: list[Any] = []

    for key, value in filter.items():
        _translate_key(key, value, value_column, conditions, params)

    return (conditions, params)


def _translate_key(
    key: str,
    value: Any,
    value_column: str,
    conditions: list[str],
    params: list[Any],
) -> None:
    """Translate a single filter key-value pair."""
    # Check if value is an operator dict
    if isinstance(value, dict) and any(k.startswith("$") for k in value):
        # Operator dict
        for op, op_value in value.items():
            if op not in _COMPARISON_OPS:
                raise ValueError(f"Unknown filter operator: {op}")
            _translate_operator(key, op, op_value, value_column, conditions, params)
    elif isinstance(value, dict):
        # Nested dict without $ operators - recurse into leaf paths
        _translate_nested_dict(key, value, value_column, conditions, params)
    else:
        # Plain equality (scalar or list)
        _translate_equality(key, value, value_column, conditions, params)


def _translate_nested_dict(
    base_key: str,
    value: dict,
    value_column: str,
    conditions: list[str],
    params: list[Any],
) -> None:
    """Translate nested dict to leaf path conditions."""
    for nested_key, nested_value in value.items():
        path = [base_key, nested_key]
        if isinstance(nested_value, dict) and any(k.startswith("$") for k in nested_value):
            # Operator dict at nested path
            for op, op_value in nested_value.items():
                if op not in _COMPARISON_OPS:
                    raise ValueError(f"Unknown filter operator: {op}")
                _translate_path_operator(path, op, op_value, value_column, conditions, params)
        elif isinstance(nested_value, dict):
            # Further nesting - recurse
            _translate_nested_dict_path(path, nested_value, value_column, conditions, params)
        else:
            # Leaf - emit condition
            conditions.append(f"{value_column} #> %s = CAST(%s AS JSONB)")
            params.append(path)
            params.append(json.dumps(nested_value))


def _translate_nested_dict_path(
    path: list[str],
    value: dict,
    value_column: str,
    conditions: list[str],
    params: list[Any],
) -> None:
    """Translate nested dict with accumulated path."""
    for key, val in value.items():
        new_path = path + [key]
        if isinstance(val, dict) and any(k.startswith("$") for k in val):
            # Operator dict at nested path
            for op, op_value in val.items():
                if op not in _COMPARISON_OPS:
                    raise ValueError(f"Unknown filter operator: {op}")
                _translate_path_operator(new_path, op, op_value, value_column, conditions, params)
        elif isinstance(val, dict):
            # Further nesting
            _translate_nested_dict_path(new_path, val, value_column, conditions, params)
        else:
            # Leaf
            conditions.append(f"{value_column} #> %s = CAST(%s AS JSONB)")
            params.append(new_path)
            params.append(json.dumps(val))


def _translate_equality(
    key: str, value: Any, value_column: str, conditions: list[str], params: list[Any]
) -> None:
    """Translate plain equality (scalar or list)."""
    conditions.append(f"{value_column} -> %s = CAST(%s AS JSONB)")
    params.append(key)
    params.append(json.dumps(value))


def _translate_operator(
    key: str,
    op: str,
    value: Any,
    value_column: str,
    conditions: list[str],
    params: list[Any],
) -> None:
    """Translate a comparison operator."""
    if op == "$eq":
        # Same as plain equality
        _translate_equality(key, value, value_column, conditions, params)
    elif op == "$ne":
        # IS DISTINCT FROM to match missing keys
        conditions.append(f"({value_column} -> %s) IS DISTINCT FROM CAST(%s AS JSONB)")
        params.append(key)
        params.append(json.dumps(value))
    elif op in {"$gt", "$gte", "$lt", "$lte"}:
        _translate_comparison(key, op, value, value_column, conditions, params)
    else:
        raise ValueError(f"Unknown filter operator: {op}")


def _translate_comparison(
    key: str,
    op: str,
    value: Any,
    value_column: str,
    conditions: list[str],
    params: list[Any],
) -> None:
    """Translate a numeric or string comparison."""
    # Map operator to SQL
    sql_op = {
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
    }[op]

    # Check value type
    if isinstance(value, bool):
        raise ValueError(f"Cannot use comparison operator {op} with boolean value")
    elif isinstance(value, (int, float)):
        # Numeric comparison with CASE to prevent evaluation order issues
        conditions.append(
            f"CAST(CASE WHEN jsonb_typeof({value_column} -> %s) = 'number' "
            f"THEN {value_column} ->> %s END AS DECIMAL) {sql_op} %s"
        )
        params.append(key)
        params.append(key)
        params.append(value)
    elif isinstance(value, str):
        # Text comparison
        conditions.append(f"{value_column} ->> %s {sql_op} %s")
        params.append(key)
        params.append(value)
    else:
        raise ValueError(
            f"Cannot use comparison operator {op} with value of type {type(value).__name__}"
        )


def _translate_path_operator(
    path: list[str],
    op: str,
    value: Any,
    value_column: str,
    conditions: list[str],
    params: list[Any],
) -> None:
    """Translate a comparison operator at a nested path."""
    if op == "$eq":
        # Equality with #>
        conditions.append(f"{value_column} #> %s = CAST(%s AS JSONB)")
        params.append(path)
        params.append(json.dumps(value))
    elif op == "$ne":
        # IS DISTINCT FROM with #>
        conditions.append(f"({value_column} #> %s) IS DISTINCT FROM CAST(%s AS JSONB)")
        params.append(path)
        params.append(json.dumps(value))
    elif op in {"$gt", "$gte", "$lt", "$lte"}:
        # Comparison at nested path
        sql_op = {
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<=",
        }[op]

        if isinstance(value, bool):
            raise ValueError(f"Cannot use comparison operator {op} with boolean value")
        elif isinstance(value, (int, float)):
            # Numeric comparison with CASE to prevent evaluation order issues
            conditions.append(
                f"CAST(CASE WHEN jsonb_typeof({value_column} #> %s) = 'number' "
                f"THEN {value_column} #>> %s END AS DECIMAL) {sql_op} %s"
            )
            params.append(path)
            params.append(path)
            params.append(value)
        elif isinstance(value, str):
            # Text comparison with #>>
            conditions.append(f"{value_column} #>> %s {sql_op} %s")
            params.append(path)
            params.append(value)
        else:
            raise ValueError(
                f"Cannot use comparison operator {op} with value of type {type(value).__name__}"
            )
    else:
        raise ValueError(f"Unknown filter operator: {op}")
