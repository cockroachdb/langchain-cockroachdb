"""Vector index abstractions for CockroachDB C-SPANN indexes."""

from enum import Enum


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
        self.distance_strategy = distance_strategy
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.name = name

    def get_create_index_sql(
        self,
        table_name: str,
        column_name: str,
        *,
        schema: str = "public",
        prefix_columns: list[str] | None = None,
    ) -> str:
        """Generate CREATE VECTOR INDEX SQL.

        Args:
            table_name: Target table name
            column_name: Vector column name
            schema: Database schema
            prefix_columns: Optional prefix columns for multi-tenant indexes

        Returns:
            SQL statement to create the index
        """
        index_name = self.name or f"{table_name}_{column_name}_vector_idx"
        fqn = f"{schema}.{table_name}"

        columns = ", ".join(prefix_columns + [column_name]) if prefix_columns else column_name

        sql = f"CREATE VECTOR INDEX IF NOT EXISTS {index_name} ON {fqn} ({columns})"

        with_clauses = []
        if self.min_partition_size is not None:
            with_clauses.append(f"min_partition_size = {self.min_partition_size}")
        if self.max_partition_size is not None:
            with_clauses.append(f"max_partition_size = {self.max_partition_size}")

        if with_clauses:
            sql += f" WITH ({', '.join(with_clauses)})"

        return sql

    def get_drop_index_sql(
        self,
        table_name: str,
        column_name: str,
        *,
        schema: str = "public",
    ) -> str:
        """Generate DROP INDEX SQL.

        Args:
            table_name: Target table name
            column_name: Vector column name
            schema: Database schema

        Returns:
            SQL statement to drop the index
        """
        index_name = self.name or f"{table_name}_{column_name}_vector_idx"
        return f"DROP INDEX IF EXISTS {schema}.{index_name}"


class CSPANNQueryOptions:
    """Query-time options for C-SPANN vector search."""

    def __init__(self, beam_size: int | None = None):
        """Initialize query options.

        Args:
            beam_size: Search beam size (higher = more accurate, slower)
                      Default is CockroachDB's vector_search_beam_size setting
        """
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
