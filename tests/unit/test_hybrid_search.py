"""Unit tests for hybrid search configuration."""

import pytest

from langchain_cockroachdb.hybrid_search_config import FusionType, HybridSearchConfig


class TestHybridSearchConfig:
    """Test hybrid search configuration."""

    def test_default_initialization(self) -> None:
        """Test default configuration."""
        config = HybridSearchConfig()
        assert config.fts_weight == 0.5
        assert config.vector_weight == 0.5
        assert config.fusion_type == FusionType.WEIGHTED_SUM
        assert config.fts_query_language == "english"
        assert config.k == 60

    def test_custom_initialization(self) -> None:
        """Test custom configuration."""
        config = HybridSearchConfig(
            fts_weight=0.3,
            vector_weight=0.7,
            fusion_type=FusionType.RRF,
            fts_query_language="spanish",
            k=100,
        )
        assert config.fts_weight == 0.3
        assert config.vector_weight == 0.7
        assert config.fusion_type == FusionType.RRF
        assert config.fts_query_language == "spanish"
        assert config.k == 100

    def test_weight_validation(self) -> None:
        """Test weight sum validation."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            HybridSearchConfig(fts_weight=0.3, vector_weight=0.5)

    def test_weighted_sum_fusion(self) -> None:
        """Test weighted sum fusion."""
        config = HybridSearchConfig(fts_weight=0.4, vector_weight=0.6)

        fts_results = [("doc1", 0.8), ("doc2", 0.6), ("doc3", 0.4)]
        vector_results = [("doc1", 0.9), ("doc3", 0.7), ("doc4", 0.5)]

        fused = config.fuse_scores(fts_results, vector_results)

        assert len(fused) == 4
        assert all(isinstance(item, tuple) and len(item) == 2 for item in fused)

        doc1_score = 0.4 * 0.8 + 0.6 * 0.9
        assert any(doc_id == "doc1" and abs(score - doc1_score) < 0.001 for doc_id, score in fused)

    def test_weighted_sum_fusion_no_overlap(self) -> None:
        """Test fusion with no overlapping documents."""
        config = HybridSearchConfig(fts_weight=0.5, vector_weight=0.5)

        fts_results = [("doc1", 0.8), ("doc2", 0.6)]
        vector_results = [("doc3", 0.9), ("doc4", 0.7)]

        fused = config.fuse_scores(fts_results, vector_results)

        assert len(fused) == 4
        doc_ids = {doc_id for doc_id, _ in fused}
        assert doc_ids == {"doc1", "doc2", "doc3", "doc4"}

    def test_rrf_fusion(self) -> None:
        """Test reciprocal rank fusion."""
        config = HybridSearchConfig(
            fts_weight=0.5,
            vector_weight=0.5,
            fusion_type=FusionType.RRF,
            k=60,
        )

        fts_results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        vector_results = [("doc1", 0.95), ("doc3", 0.8), ("doc4", 0.6)]

        fused = config.fuse_scores(fts_results, vector_results)

        assert len(fused) == 4

        doc1_rrf = 0.5 / (60 + 1) + 0.5 / (60 + 1)
        assert any(doc_id == "doc1" and abs(score - doc1_rrf) < 0.001 for doc_id, score in fused)

    def test_rrf_fusion_ordering(self) -> None:
        """Test that RRF fusion produces correct ordering."""
        config = HybridSearchConfig(
            fts_weight=0.5,
            vector_weight=0.5,
            fusion_type=FusionType.RRF,
        )

        fts_results = [("doc1", 1.0), ("doc2", 0.5)]
        vector_results = [("doc1", 1.0), ("doc3", 0.5)]

        fused = config.fuse_scores(fts_results, vector_results)

        assert fused[0][0] == "doc1"
        assert fused[0][1] > fused[1][1]

    def test_empty_results(self) -> None:
        """Test fusion with empty results."""
        config = HybridSearchConfig()

        assert config.fuse_scores([], []) == []

        fts_results = [("doc1", 0.8)]
        assert len(config.fuse_scores(fts_results, [])) == 1

        vector_results = [("doc2", 0.9)]
        assert len(config.fuse_scores([], vector_results)) == 1
