"""Integration tests for hybrid search execution (issue #7).

These tests verify that the FTS half of hybrid search actually runs against
the database and that its results get fused with vector results. The fake
embeddings are built so that pure vector search cannot find keyword matches,
which means any keyword doc showing up in results proves the FTS leg fired.
"""

import pytest
from langchain_core.embeddings import Embeddings

from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.hybrid_search_config import FusionType, HybridSearchConfig
from langchain_cockroachdb.vectorstores import CockroachDBVectorStore

COLLECTION = "test_hybrid_execution"


class KeywordBlindEmbeddings(Embeddings):
    """Embeddings where zebra docs live far away from every query.

    Queries and ordinary docs embed to [1, 0, 0]. Docs mentioning zebra embed
    to [0, 1, 0], so vector search alone will never rank them near the top.
    """

    def _embed(self, text: str) -> list[float]:
        if "zebra" in text.lower():
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


DOCS = [
    "The zebra grazes on the African savanna",
    "Databases store structured data reliably",
    "Python is a popular programming language",
    "Distributed systems scale horizontally",
]

METADATAS = [
    {"topic": "animals"},
    {"topic": "databases"},
    {"topic": "programming"},
    {"topic": "systems"},
]


async def _make_store(
    engine: CockroachDBEngine,
    config: HybridSearchConfig | None,
    **table_kwargs,
) -> AsyncCockroachDBVectorStore:
    await engine.ainit_vectorstore_table(
        table_name=COLLECTION,
        vector_dimension=3,
        create_tsvector=True,
        drop_if_exists=True,
        **table_kwargs,
    )
    store = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=KeywordBlindEmbeddings(),
        collection_name=COLLECTION,
        hybrid_search_config=config,
    )
    await store.aadd_texts(DOCS, metadatas=METADATAS)
    return store


@pytest.mark.asyncio
class TestHybridSearchExecution:
    """Hybrid search must actually execute FTS and fuse the results."""

    async def test_fts_leg_surfaces_keyword_match(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """A doc only findable by keyword must appear in hybrid results."""
        config = HybridSearchConfig(fts_weight=0.7, vector_weight=0.3)
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search("zebra", k=2)

        contents = [doc.page_content for doc in results]
        assert any("zebra" in c for c in contents), (
            "FTS leg did not run: zebra doc missing from hybrid results"
        )
        # With fts weighted higher, the keyword match should win outright
        assert "zebra" in results[0].page_content

    async def test_pure_vector_search_misses_keyword_match(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """Sanity check: without hybrid config the zebra doc is not in top 2."""
        store = await _make_store(cockroachdb_engine, None)

        results = await store.asimilarity_search("zebra", k=2)

        contents = [doc.page_content for doc in results]
        assert not any("zebra" in c for c in contents)

    async def test_rrf_fusion(self, cockroachdb_engine: CockroachDBEngine) -> None:
        """RRF fusion should rank the doc found by both legs highest."""
        config = HybridSearchConfig(fusion_type=FusionType.RRF)
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search_with_score("zebra", k=4)

        assert len(results) > 0
        contents = [doc.page_content for doc, _ in results]
        assert any("zebra" in c for c in contents)
        # Scores must be RRF scores, descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    async def test_hybrid_respects_metadata_filter(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """The FTS leg must apply the same metadata filter as the vector leg."""
        config = HybridSearchConfig(fts_weight=0.7, vector_weight=0.3)
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search(
            "zebra", k=4, filter={"topic": {"$eq": "databases"}}
        )

        for doc in results:
            assert doc.metadata["topic"] == "databases"

    async def test_hybrid_with_no_fts_matches_falls_back_to_vector(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """A query with zero keyword hits should still return vector results."""
        config = HybridSearchConfig()
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search("quantum entanglement", k=2)

        assert len(results) == 2

    async def test_backfills_docs_missing_from_vector_leg(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """FTS-only hits outside the vector candidate pool still come back whole."""
        config = HybridSearchConfig(fts_weight=0.7, vector_weight=0.3)
        store = await _make_store(cockroachdb_engine, config)

        # fetch_k=2 keeps the zebra doc out of the vector candidates entirely,
        # so it can only be materialized through the FTS backfill path
        results = await store.asimilarity_search("zebra", k=2, fetch_k=2)

        zebra_docs = [d for d in results if "zebra" in d.page_content]
        assert zebra_docs, "zebra doc should be backfilled from the FTS leg"
        assert zebra_docs[0].metadata == {"topic": "animals"}

    async def test_scores_returned_are_fused_scores(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> None:
        """Weighted sum scores land in [0, 1] and sort descending."""
        config = HybridSearchConfig(fusion_type=FusionType.WEIGHTED_SUM)
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search_with_score("zebra databases", k=4)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    async def test_default_fusion_is_rrf(self, cockroachdb_engine: CockroachDBEngine) -> None:
        """An untouched config fuses with RRF, so scores are reciprocal ranks."""
        config = HybridSearchConfig()
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search_with_score("zebra", k=4)

        assert results
        # RRF scores with k=60 can never exceed sum of weights / 61
        assert all(0.0 < score <= 2.0 / 61.0 for _, score in results)

    async def test_fts_rank_normalization(self, cockroachdb_engine: CockroachDBEngine) -> None:
        """A non-default ts_rank normalization must produce valid SQL."""
        config = HybridSearchConfig(
            fts_weight=0.7,
            vector_weight=0.3,
            fusion_type=FusionType.WEIGHTED_SUM,
            fts_rank_normalization=1 | 32,
        )
        store = await _make_store(cockroachdb_engine, config)

        results = await store.asimilarity_search("zebra", k=4)

        contents = [doc.page_content for doc in results]
        assert any("zebra" in c for c in contents)

    async def test_custom_fts_language(self, cockroachdb_engine: CockroachDBEngine) -> None:
        """Table and config can both use a non-default text search config."""
        config = HybridSearchConfig(fts_weight=0.7, vector_weight=0.3, fts_query_language="simple")
        store = await _make_store(cockroachdb_engine, config, fts_language="simple")

        results = await store.asimilarity_search("zebra", k=4)

        contents = [doc.page_content for doc in results]
        assert any("zebra" in c for c in contents)


class TestHybridSearchSyncWrapper:
    """The sync store must run hybrid search through the same path."""

    def test_sync_similarity_search_uses_hybrid(self, connection_string: str) -> None:
        engine = CockroachDBEngine.from_connection_string(connection_string)
        try:
            engine.init_vectorstore_table(
                table_name=COLLECTION,
                vector_dimension=3,
                create_tsvector=True,
                drop_if_exists=True,
            )
            store = CockroachDBVectorStore(
                engine=engine,
                embeddings=KeywordBlindEmbeddings(),
                collection_name=COLLECTION,
                hybrid_search_config=HybridSearchConfig(fts_weight=0.7, vector_weight=0.3),
            )
            store.add_texts(DOCS, metadatas=METADATAS)

            results = store.similarity_search("zebra", k=2)

            contents = [doc.page_content for doc in results]
            assert any("zebra" in c for c in contents)
        finally:
            engine.close()
