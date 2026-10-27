"""Store configuration types and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.embeddings import Embeddings
from langgraph.store.base import IndexConfig, TTLConfig
from langgraph.store.base.embed import ensure_embeddings, tokenize_path

from langchain_cockroachdb.indexes import DistanceStrategy


class CSPANNConfig(TypedDict, total=False):
    """C-SPANN index configuration.

    Attributes:
        min_partition_size: Minimum vectors per partition.
        max_partition_size: Maximum vectors per partition.
    """

    min_partition_size: int
    max_partition_size: int


# total=False keeps every added key optional. The ignore is for type checks
# run without langgraph installed, where the base class is untyped
class CockroachDBIndexConfig(IndexConfig, total=False):  # type: ignore[call-arg,unused-ignore]
    """CockroachDB-specific index configuration.

    Extends LangGraph IndexConfig with CockroachDB vector index options.

    Attributes:
        distance_type: Distance metric (cosine, l2, or inner_product). Default cosine.
        cspann: C-SPANN index parameters (both or neither).
        flat: Whether to use a flat (brute-force) index. Default False.
        beam_size: Beam width for C-SPANN search. Searches with a query set
            it with SET LOCAL in their own transaction. Inside a transaction
            the caller opened on a single connection shared with the store,
            it lasts until that transaction ends.
    """

    distance_type: Literal["cosine", "l2", "inner_product"]
    cspann: CSPANNConfig | None
    flat: bool
    beam_size: int


class CockroachDBTTLConfig(TTLConfig, total=False):  # type: ignore[call-arg,unused-ignore]
    """CockroachDB-specific TTL configuration.

    Extends LangGraph TTLConfig with CockroachDB TTL options.

    Attributes:
        omit_expired: Whether to exclude expired items from reads. Default True.
        ttl_job_cron: Cron expression for the TTL cleanup job.
    """

    omit_expired: bool
    ttl_job_cron: str


class PoolConfig(TypedDict, total=False):
    """Connection pool settings for ``CockroachDBStore.from_conn_string``.

    The same meaning as PostgresStore's PoolConfig.

    Attributes:
        min_size: Connections the pool keeps open. Default 1.
        max_size: Most connections the pool opens, or None for min_size.
        kwargs: Extra psycopg connection options for every pooled connection.
            autocommit, row_factory and prepare_threshold are set by the store.
    """

    min_size: int
    max_size: int | None
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class ResolvedIndex:
    """Validated and normalized index configuration.

    Attributes:
        dims: Vector dimensions.
        embeddings: Embeddings model.
        fields: Tuple of (path, tokens) for each indexed field.
        estimated_vectors: Estimated number of vectors per item.
        distance: Distance strategy.
        cspann: C-SPANN partition sizes (empty dict for no C-SPANN).
        flat: Whether to use flat index.
        beam_size: Beam width, or None.
    """

    dims: int
    embeddings: Embeddings
    fields: tuple[tuple[str, Literal["$"] | list[str]], ...]
    estimated_vectors: int
    distance: DistanceStrategy
    cspann: dict[str, int]
    flat: bool
    beam_size: int | None


@dataclass(frozen=True)
class ResolvedTTL:
    """Validated and normalized TTL configuration.

    Attributes:
        default_ttl: Default expiration in minutes, or None.
        refresh_on_read: Whether to refresh expiration on read.
        omit_expired: Whether to exclude expired items from reads.
        sweep_interval_minutes: TTL sweep interval, or None.
        cron: Cron expression for the TTL cleanup job.
    """

    default_ttl: float | None
    refresh_on_read: bool
    omit_expired: bool
    sweep_interval_minutes: float | None
    cron: str


def _validate_cspann(cspann_config: CSPANNConfig | None) -> dict[str, int]:
    """Validate C-SPANN configuration.

    Args:
        cspann_config: C-SPANN configuration, or None.

    Returns:
        Dictionary with min and max sizes, or empty dict.

    Raises:
        ValueError: If configuration is invalid.
    """
    if cspann_config is None:
        return {}

    min_size = cspann_config.get("min_partition_size")
    max_size = cspann_config.get("max_partition_size")

    # Both or neither
    if (min_size is None) != (max_size is None):
        raise ValueError(
            "cspann requires both min_partition_size and max_partition_size, or neither"
        )

    if min_size is None or max_size is None:
        return {}

    # Validate positive ints
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 1:
        raise ValueError("cspann min_partition_size must be a positive integer")
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
        raise ValueError("cspann max_partition_size must be a positive integer")

    # Validate max >= 4 * min
    if max_size < 4 * min_size:
        raise ValueError(
            f"cspann max_partition_size must be at least 4 times min_partition_size, "
            f"got max={max_size}, min={min_size}"
        )

    return {"min_partition_size": min_size, "max_partition_size": max_size}


def resolve_index_config(index: CockroachDBIndexConfig | None) -> ResolvedIndex | None:
    """Validate and normalize index configuration.

    Args:
        index: Index configuration, or None for no index.

    Returns:
        Resolved index configuration, or None if index is None.

    Raises:
        ValueError: If configuration is invalid (names the bad key).
    """
    if index is None:
        return None

    # Validate dims
    dims = index.get("dims")
    if dims is None:
        raise ValueError("dims is required")
    if isinstance(dims, bool) or not isinstance(dims, int) or dims < 1:
        raise ValueError("dims must be a positive integer")

    # Validate and resolve embeddings
    embeddings = ensure_embeddings(index.get("embed"))

    # Validate distance_type
    distance_type = index.get("distance_type", "cosine")
    if distance_type not in ("cosine", "l2", "inner_product"):
        raise ValueError(
            f"distance_type must be one of cosine, l2, inner_product, got {distance_type!r}"
        )

    distance_map = {
        "cosine": DistanceStrategy.COSINE,
        "l2": DistanceStrategy.EUCLIDEAN,
        "inner_product": DistanceStrategy.INNER_PRODUCT,
    }
    distance = distance_map[distance_type]

    # Validate C-SPANN config
    cspann_dict = _validate_cspann(index.get("cspann"))

    # Validate flat. A string such as "false" would be truthy and skip the index
    flat = index.get("flat", False)
    if not isinstance(flat, bool):
        raise ValueError(f"flat must be a bool, got {flat!r}")

    # Validate beam_size
    beam_size = index.get("beam_size")
    if beam_size is not None and (
        isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1
    ):
        raise ValueError("beam_size must be a positive integer")

    # Process fields
    raw_fields = index.get("fields") or ["$"]
    fields_list: list[tuple[str, Literal["$"] | list[str]]] = []
    estimated = 0

    for path in raw_fields:
        if path == "$":
            # Root path is stored as ("$", "$") and counts as 1
            fields_list.append(("$", "$"))
            estimated += 1
        else:
            # Other paths use tokenize_path
            tokens = tokenize_path(path)
            fields_list.append((path, tokens))
            estimated += len(tokens)

    return ResolvedIndex(
        dims=dims,
        embeddings=embeddings,
        fields=tuple(fields_list),
        estimated_vectors=estimated,
        distance=distance,
        cspann=cspann_dict,
        flat=flat,
        beam_size=beam_size,
    )


def ttl_job_cron_for(sweep_interval_minutes: float | None) -> str:
    """Generate a cron expression for the given sweep interval.

    Args:
        sweep_interval_minutes: TTL sweep interval in minutes, or None.

    Returns:
        Cron expression string.

    Raises:
        ValueError: If interval is invalid (non-integer below 60, or below 1).
    """
    if sweep_interval_minutes is None or sweep_interval_minutes >= 60:
        return "@hourly"

    if sweep_interval_minutes < 1:
        raise ValueError("sweep_interval_minutes must be at least 1 minute")

    if not float(sweep_interval_minutes).is_integer():
        raise ValueError("sweep_interval_minutes below 60 must be an integer")

    return f"*/{int(sweep_interval_minutes)} * * * *"


def resolve_ttl_config(ttl: TTLConfig | CockroachDBTTLConfig | None) -> ResolvedTTL:
    """Validate and normalize TTL configuration.

    Args:
        ttl: TTL configuration, or None for defaults.

    Returns:
        Resolved TTL configuration.

    Raises:
        ValueError: If ttl_job_cron is invalid.
    """
    if ttl is None:
        return ResolvedTTL(
            default_ttl=None,
            refresh_on_read=True,
            omit_expired=True,
            sweep_interval_minutes=None,
            cron="@hourly",
        )

    default_ttl = ttl.get("default_ttl")
    refresh_on_read = ttl.get("refresh_on_read", True)
    omit_expired = ttl.get("omit_expired", True)
    sweep_interval_minutes = ttl.get("sweep_interval_minutes")

    # Check for explicit ttl_job_cron
    ttl_job_cron = ttl.get("ttl_job_cron") if isinstance(ttl, dict) else None
    if ttl_job_cron is not None and not isinstance(ttl_job_cron, str):
        raise ValueError(f"ttl_job_cron must be a string, got {type(ttl_job_cron).__name__}")

    if ttl_job_cron is not None:
        # Validate cron expression to prevent SQL injection
        if not re.match(r"^[@\w\s\*/,-]+$", ttl_job_cron):
            raise ValueError(
                f"Invalid ttl_job_cron: {ttl_job_cron!r}. "
                "Must be a valid cron expression like '@daily', '* * * * *', etc."
            )
        cron = ttl_job_cron
    else:
        cron = ttl_job_cron_for(sweep_interval_minutes)

    return ResolvedTTL(
        default_ttl=default_ttl,
        refresh_on_read=refresh_on_read,
        omit_expired=omit_expired,
        sweep_interval_minutes=sweep_interval_minutes,
        cron=cron,
    )
