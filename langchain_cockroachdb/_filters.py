"""Translate LangChain metadata filters into SQL with bound parameters.

Supports the operator grammar langchain-postgres uses: $eq, $ne, $lt, $lte,
$gt, $gte, $between, $in, $nin, $exists, $like, $ilike, $and, $or and $not.
Keys and values are always bound parameters, never part of the SQL text.
"""

from __future__ import annotations

import json
from typing import Any

_RANGE_OPERATORS = {"$lt": "<", "$lte": "<=", "$gt": ">", "$gte": ">="}


class FilterBuilder:
    """Builds the WHERE fragment for one query and collects its parameters.

    Example:
        ```python
        builder = FilterBuilder("metadata")
        where = builder.build({"category": "news", "year": {"$gte": 2020}})
        await conn.execute(text(f"SELECT ... WHERE {where}"), builder.params)
        ```
    """

    def __init__(self, metadata_column: str, *, prefix: str = "f") -> None:
        self._column = metadata_column
        self._prefix = prefix
        self.params: dict[str, Any] = {}

    def bind(self, value: Any) -> str:
        """Add a parameter and return its placeholder."""
        name = f"{self._prefix}{len(self.params)}"
        self.params[name] = value
        return f":{name}"

    def build(self, filter: dict[str, Any]) -> str:
        """Return a SQL boolean expression for ``filter``."""
        if not isinstance(filter, dict):
            raise ValueError(f"A filter must be a dict, got {type(filter).__name__}")
        if not filter:
            return "TRUE"
        clauses = [self._clause(key, value) for key, value in filter.items()]
        return clauses[0] if len(clauses) == 1 else "(" + " AND ".join(clauses) + ")"

    def _clause(self, key: str, value: Any) -> str:
        if key in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise ValueError(f"{key} expects a non-empty list of filters")
            joiner = " AND " if key == "$and" else " OR "
            return "(" + joiner.join(self.build(item) for item in value) + ")"
        if key == "$not":
            return f"NOT ({self.build(value)})"
        if not isinstance(key, str) or not key or key.startswith("$"):
            raise ValueError(f"Unsupported filter key {key!r}")
        if isinstance(value, dict):
            if not value:
                raise ValueError(f"Empty operator dict for {key!r}")
            parts = [self._operator(key, op, operand) for op, operand in value.items()]
            return parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"
        return self._operator(key, "$eq", value)

    def _operator(self, key: str, op: str, value: Any) -> str:
        if op not in {"$eq", "$ne", "$between", "$in", "$nin", "$exists", "$like", "$ilike"} | set(
            _RANGE_OPERATORS
        ):
            raise ValueError(f"Unsupported operator {op!r}")
        k = self.bind(key)
        field = f"{self._column}->CAST({k} AS STRING)"
        text_field = f"{self._column}->>CAST({k} AS STRING)"
        if op == "$eq":
            return f"{field} = {self._json(value)}"
        if op == "$ne":
            return f"{field} != {self._json(value)}"
        if op in _RANGE_OPERATORS:
            return self._range(field, text_field, _RANGE_OPERATORS[op], value, op)
        if op == "$between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("$between expects [low, high]")
            low = self._range(field, text_field, ">=", value[0], op)
            high = self._range(field, text_field, "<=", value[1], op)
            return f"({low} AND {high})"
        if op in ("$in", "$nin"):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{op} expects a list")
            if not value:
                return "FALSE" if op == "$in" else "TRUE"
            items = ", ".join(self._json(item) for item in value)
            keyword = "IN" if op == "$in" else "NOT IN"
            return f"{field} {keyword} ({items})"
        if op == "$exists":
            if not isinstance(value, bool):
                raise ValueError("$exists expects true or false")
            exists = f"{self._column} ? CAST({k} AS STRING)"
            return exists if value else f"NOT ({exists})"
        if not isinstance(value, str):
            raise ValueError(f"{op} expects a string pattern")
        keyword = "LIKE" if op == "$like" else "ILIKE"
        return f"{text_field} {keyword} {self.bind(value)}"

    def _json(self, value: Any) -> str:
        return f"CAST({self.bind(json.dumps(value))} AS JSONB)"

    def _range(self, field: str, text_field: str, sql_op: str, value: Any, op: str) -> str:
        # CASE keeps a row whose value has another JSON type from failing the cast
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{op} expects a number or a string")
        if isinstance(value, str):
            return (
                f"(CASE WHEN jsonb_typeof({field}) = 'string' THEN {text_field} END) "
                f"{sql_op} {self.bind(value)}"
            )
        return (
            f"(CASE WHEN jsonb_typeof({field}) = 'number' THEN CAST({text_field} AS DECIMAL) END) "
            f"{sql_op} CAST({self.bind(value)} AS DECIMAL)"
        )
