"""Hybrid search configuration for combining FTS and vector search."""

import re
from enum import Enum

# Text search configuration names like "english" or "pg_catalog.english".
# The language name gets inlined into SQL, so keep this strict.
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def validate_fts_language(language: str) -> str:
    """Validate a text search configuration name for safe use in SQL.

    Args:
        language: Text search config name, e.g. "english" or "simple"

    Returns:
        The validated language name

    Raises:
        ValueError: If the name is not a plain identifier
    """
    if not language or not _LANGUAGE_PATTERN.match(language):
        raise ValueError(
            f"Invalid FTS language {language!r}. Expected a plain identifier "
            "like 'english' or 'simple'."
        )
    return language


def min_max_normalize(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Rescale scores to [0, 1] so FTS and vector scores can be summed.

    ts_rank scores and vector similarities live on different scales, so a
    weighted sum is only meaningful after both sides are normalized. When all
    scores are equal (including a single result) everything maps to 1.0.

    Args:
        results: List of (id, score) tuples

    Returns:
        List of (id, score) tuples with scores in [0, 1], original order kept
    """
    if not results:
        return []

    scores = [score for _, score in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(doc_id, 1.0) for doc_id, _ in results]

    return [(doc_id, (score - lo) / (hi - lo)) for doc_id, score in results]


class FusionType(str, Enum):
    """Fusion methods for combining FTS and vector scores."""

    WEIGHTED_SUM = "weighted_sum"
    RRF = "reciprocal_rank_fusion"


class HybridSearchConfig:
    """Configuration for hybrid full-text + vector search."""

    def __init__(
        self,
        fts_weight: float = 0.5,
        vector_weight: float = 0.5,
        fusion_type: FusionType = FusionType.RRF,
        fts_query_language: str = "english",
        k: int = 60,
        fts_rank_normalization: int = 0,
    ):
        """Initialize hybrid search configuration.

        Args:
            fts_weight: Weight for full-text search scores (0-1)
            vector_weight: Weight for vector similarity scores (0-1)
            fusion_type: Method for combining scores. RRF is the default
                because it fuses by rank and is insensitive to score scales.
            fts_query_language: Language for full-text search
            k: Parameter for RRF (typically 60)
            fts_rank_normalization: ts_rank normalization bitmask (0-63).
                0 leaves ranks unnormalized. Useful values include 1 (divide
                by 1 + log of document length) and 32 (rank / (rank + 1)) to
                damp the bias toward long documents. See the PostgreSQL
                ts_rank docs for the full list of flags.
        """
        if abs(fts_weight + vector_weight - 1.0) > 0.001:
            raise ValueError("fts_weight + vector_weight must equal 1.0")

        if not isinstance(fts_rank_normalization, int) or isinstance(fts_rank_normalization, bool):
            raise ValueError("fts_rank_normalization must be an integer bitmask between 0 and 63")
        if not 0 <= fts_rank_normalization <= 63:
            raise ValueError("fts_rank_normalization must be an integer bitmask between 0 and 63")

        self.fts_weight = fts_weight
        self.vector_weight = vector_weight
        self.fusion_type = FusionType(fusion_type)
        self.fts_query_language = validate_fts_language(fts_query_language)
        self.k = k
        self.fts_rank_normalization = fts_rank_normalization

    def fuse_scores(
        self,
        fts_results: list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse FTS and vector search results.

        Args:
            fts_results: List of (id, score) from FTS
            vector_results: List of (id, score) from vector search

        Returns:
            Fused list of (id, score) sorted by final score
        """
        if self.fusion_type == FusionType.WEIGHTED_SUM:
            return self._weighted_sum_fusion(fts_results, vector_results)
        elif self.fusion_type == FusionType.RRF:
            return self._rrf_fusion(fts_results, vector_results)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def _weighted_sum_fusion(
        self,
        fts_results: list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Combine scores using weighted sum."""
        fts_dict = dict(fts_results)
        vector_dict = dict(vector_results)
        all_ids = set(fts_dict.keys()) | set(vector_dict.keys())

        combined = []
        for doc_id in all_ids:
            fts_score = fts_dict.get(doc_id, 0.0)
            vector_score = vector_dict.get(doc_id, 0.0)
            final_score = self.fts_weight * fts_score + self.vector_weight * vector_score
            combined.append((doc_id, final_score))

        return sorted(combined, key=lambda x: x[1], reverse=True)

    def _rrf_fusion(
        self,
        fts_results: list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Combine scores using Reciprocal Rank Fusion."""
        fts_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(fts_results, 1)}
        vector_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(vector_results, 1)}
        all_ids = set(fts_ranks.keys()) | set(vector_ranks.keys())

        combined = []
        for doc_id in all_ids:
            fts_rank = fts_ranks.get(doc_id, 0)
            vector_rank = vector_ranks.get(doc_id, 0)

            rrf_score = 0.0
            if fts_rank > 0:
                rrf_score += self.fts_weight / (self.k + fts_rank)
            if vector_rank > 0:
                rrf_score += self.vector_weight / (self.k + vector_rank)

            combined.append((doc_id, rrf_score))

        return sorted(combined, key=lambda x: x[1], reverse=True)
