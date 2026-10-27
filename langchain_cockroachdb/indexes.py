"""Vector index abstractions for CockroachDB C-SPANN indexes."""

import re
from dataclasses import dataclass
from enum import Enum

from langchain_cockroachdb._sql import validate_identifier


def _check_positive_int(value: object, what: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{what} must be a positive integer")


class DistanceStrategy(str, Enum):
    """Distance strategies for vector similarity."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    INNER_PRODUCT = "ip"

    def get_operator(self) -> str:
        """Get SQL operator for distance calculation."""
        return {
            "l2": "<->",
            "cosine": "<=>",
            "ip": "<#>",
        }[self.value]

    def get_opclass(self) -> str:
        """Get index operator class."""
        return {
            "l2": "vector_l2_ops",
            "cosine": "vector_cosine_ops",
            "ip": "vector_ip_ops",
        }[self.value]


class CSPANNIndex:
    """CockroachDB C-SPANN vector index configuration.

    C-SPANN (CockroachDB SPANN) is CockroachDB's vector indexing algorithm
    for real-time indexing of billions of vectors in distributed systems.
    Based on Microsoft's SPANN (Space Partition with Approximate Nearest Neighbor).
    """

    def __init__(
        self,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        min_partition_size: int | None = None,
        max_partition_size: int | None = None,
        name: str | None = None,
    ):
        """Initialize C-SPANN index configuration.

        Args:
            distance_strategy: Distance metric to use
            min_partition_size: Minimum partition size for C-SPANN
            max_partition_size: Maximum partition size for C-SPANN
            name: Optional custom index name
        """
        if name is not None:
            validate_identifier(name, what="index name")
        _check_positive_int(min_partition_size, "min_partition_size")
        _check_positive_int(max_partition_size, "max_partition_size")
        self.distance_strategy = distance_strategy
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.name = name

    def get_index_name(self, table_name: str, column_name: str) -> str:
        """Name of the index on ``table_name.column_name``."""
        return self.name or f"{table_name}_{column_name}_vector_idx"

    def get_create_index_sql(
        self,
        table_name: str,
        column_name: str,
        *,
        schema: str = "public",
        prefix_columns: list[str] | None = None,
    ) -> str:
        """Generate CREATE VECTOR INDEX SQL.

        The vector column carries the operator class of the distance strategy;
        without it CockroachDB builds a vector_l2_ops index that cosine and
        inner product searches cannot use.

        Args:
            table_name: Target table name
            column_name: Vector column name
            schema: Database schema
            prefix_columns: Optional prefix columns for multi-tenant indexes

        Returns:
            SQL statement to create the index
        """
        vector_column = f"{column_name} {self.distance_strategy.get_opclass()}"
        columns = ", ".join([*(prefix_columns or []), vector_column])
        sql = (
            f"CREATE VECTOR INDEX IF NOT EXISTS {self.get_index_name(table_name, column_name)} "
            f"ON {schema}.{table_name} ({columns})"
        )
        with_clauses = []
        if self.min_partition_size is not None:
            with_clauses.append(f"min_partition_size = {self.min_partition_size}")
        if self.max_partition_size is not None:
            with_clauses.append(f"max_partition_size = {self.max_partition_size}")
        if with_clauses:
            sql += f" WITH ({', '.join(with_clauses)})"
        return sql

    def get_drop_index_sql(
        self, table_name: str, column_name: str, *, schema: str = "public"
    ) -> str:
        """Generate DROP INDEX SQL.

        Args:
            table_name: Target table name
            column_name: Vector column name
            schema: Database schema

        Returns:
            SQL statement to drop the index
        """
        return f"DROP INDEX IF EXISTS {schema}.{self.get_index_name(table_name, column_name)}"


class CSPANNQueryOptions:
    """Query-time options for C-SPANN vector search."""

    def __init__(self, beam_size: int | None = None):
        """Initialize query options.

        Args:
            beam_size: Search beam size (higher = more accurate, slower)
                      Default is CockroachDB's vector_search_beam_size setting
        """
        _check_positive_int(beam_size, "beam_size")
        self.beam_size = beam_size

    def get_session_settings(self) -> dict[str, int]:
        """Get session settings to apply before query.

        Returns:
            Dictionary of session variable names to values
        """
        settings = {}
        if self.beam_size is not None:
            settings["vector_search_beam_size"] = self.beam_size
        return settings


@dataclass
class _ExistingVectorIndex:
    """A vector index as SHOW CREATE TABLE renders it."""

    definition: str
    columns: list[str]
    options: dict[str, str]


def _find_vector_index(create_statement: str, name: str) -> _ExistingVectorIndex | None:
    """Find the vector index ``name`` in a SHOW CREATE TABLE statement.

    SHOW CREATE TABLE renders a line such as
    ``VECTOR INDEX idx ("tenant", ns, embedding vector_cosine_ops) WITH
    (min_partition_size=10, max_partition_size=50)`` on every supported
    version. Column names that are keywords come back quoted. Returns None
    when no such line exists.
    """
    match = re.search(
        rf'^\s*VECTOR INDEX "?{re.escape(name)}"? \((?P<columns>[^)]*)\)'
        r"(?: WITH \((?P<options>[^)]*)\))?",
        create_statement,
        re.MULTILINE | re.IGNORECASE,
    )
    if match is None:
        return None
    columns = [
        part.split()[0].strip('"').lower()
        for part in match.group("columns").split(",")
        if part.strip()
    ]
    options = {}
    for option in (match.group("options") or "").split(","):
        key, sep, value = option.partition("=")
        if sep:
            options[key.strip().lower()] = value.strip()
    return _ExistingVectorIndex(match.group(0), columns, options)
