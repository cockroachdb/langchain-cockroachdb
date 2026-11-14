"""Optional observations of the vector store's search execution."""

from dataclasses import dataclass
from typing import Literal

SearchPath = Literal["empty", "direct", "filtered_exact", "filtered_index", "exact_fallback"]


@dataclass(frozen=True)
class SearchDiagnostics:
    """One completed vector search, without query text, vectors or tenant data.

    ``path`` describes the library's SQL strategy, not an EXPLAIN plan. A
    ``direct`` query leaves index selection to the database optimizer.
    ``elapsed_seconds`` includes index discovery, connection checkout and
    retries, but excludes embedding, MMR selection and hybrid fusion.
    ``window_candidates`` is unknown when an indexed window returns no matches,
    because that statement returns no row carrying the candidate count.
    ``window_matches`` counts rows returned by the filtered window (up to k),
    not every candidate that could satisfy the filter.
    """

    path: SearchPath
    requested_k: int
    returned_count: int
    elapsed_seconds: float
    window_limit: int | None = None
    window_candidates: int | None = None
    window_matches: int | None = None
