"""Vector index abstractions for CockroachDB C-SPANN indexes."""

import re
from collections.abc import Sequence
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
    """A vector index as SHOW CREATE TABLE renders it.

    ``definition`` is its line up to the end of the WITH options.
    ``columns`` are its prefix columns, then its vector column, unquoted and
    in lower case. ``opclass`` is the vector column's operator class, and
    vector_l2_ops, the default, when none is shown (v25.4.14 and v26.3.4
    always show one). ``partial`` is set for an index with a WHERE clause.
    ``visible`` is cleared for an index that is NOT VISIBLE or only partly
    visible to the optimizer.
    """

    definition: str
    columns: list[str]
    options: dict[str, str]
    name: str = ""
    opclass: str = "vector_l2_ops"
    partial: bool = False
    visible: bool = True


# A vector index line of SHOW CREATE TABLE, such as
# VECTOR INDEX idx ("tenant", ns, embedding vector_cosine_ops) WITH
# (min_partition_size=10, max_partition_size=50), on every supported
# version. Names that need quotes come back quoted, with a quote inside
# doubled. After the columns and options, a partial index shows
# WHERE <predicate>, and an index the optimizer may skip shows NOT VISIBLE
# or VISIBILITY <fraction>.
_VECTOR_INDEX_LINE = re.compile(
    r'^(?P<definition>\s*VECTOR INDEX (?P<name>"(?:[^"]|"")+"|[^\s"(]+) '
    r"\((?P<columns>[^)]*)\)(?: WITH \((?P<options>[^)]*)\))?)(?P<rest>.*)$",
    re.MULTILINE | re.IGNORECASE,
)


def _unquote(name: str) -> str:
    if len(name) > 1 and name.startswith('"') and name.endswith('"'):
        return name[1:-1].replace('""', '"')
    return name


def _vector_indexes(create_statement: str) -> list[_ExistingVectorIndex]:
    """Every vector index in a SHOW CREATE TABLE statement, in the order shown."""
    found = []
    for match in _VECTOR_INDEX_LINE.finditer(create_statement):
        parts = [part.split() for part in match.group("columns").split(",") if part.strip()]
        options = {}
        for option in (match.group("options") or "").split(","):
            key, sep, value = option.partition("=")
            if sep:
                options[key.strip().lower()] = value.strip()
        rest = match.group("rest")
        found.append(
            _ExistingVectorIndex(
                definition=match.group("definition"),
                columns=[part[0].strip('"').lower() for part in parts],
                options=options,
                name=_unquote(match.group("name")),
                opclass=parts[-1][1].lower() if parts and len(parts[-1]) > 1 else "vector_l2_ops",
                partial=re.search(r"\bWHERE\b", rest, re.IGNORECASE) is not None,
                visible=re.search(r"\b(?:NOT VISIBLE|VISIBILITY)\b", rest, re.IGNORECASE) is None,
            )
        )
    return found


def _find_vector_index(create_statement: str, name: str) -> _ExistingVectorIndex | None:
    """Find the vector index ``name`` in a SHOW CREATE TABLE statement.

    Names compare without regard to case. Returns None when the table has no
    vector index of that name.
    """
    wanted = name.lower()
    for index in _vector_indexes(create_statement):
        if index.name.lower() == wanted:
            return index
    return None


def _usable_vector_index(
    create_statement: str, *, column: str, opclass: str, prefix: Sequence[str]
) -> str | None:
    """The name of a vector index that can serve a filtered search, or None.

    CockroachDB uses a vector index only when each of its prefix columns is
    constrained to a value, and accelerates a search with filters only when
    the filters match prefix columns. So the index must be on ``column``
    with ``opclass``, and its prefix columns must be exactly ``prefix``: none
    for a search of the whole table, the namespace column for a namespaced
    store. A partial index, or one the optimizer may skip, does not count.
    """
    wanted = [*(name.lower() for name in prefix), column.lower()]
    for index in _vector_indexes(create_statement):
        if (
            index.columns == wanted
            and index.opclass == opclass.lower()
            and not index.partial
            and index.visible
        ):
            return index.name
    return None
