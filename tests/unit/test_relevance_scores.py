"""Relevance conversions and retriever thresholds preserve rank in both APIs."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CockroachDBVectorStore,
    DistanceStrategy,
    HybridSearchConfig,
)


@pytest.fixture(params=[AsyncCockroachDBVectorStore, CockroachDBVectorStore])
def store(request):
    engine = CockroachDBEngine.from_connection_string("cockroachdb://root@localhost:1/unused")
    try:
        yield request.param(engine, DeterministicFakeEmbedding(size=3), "docs")
    finally:
        engine.close()


@pytest.mark.parametrize(
    "strategy,distances,expected",
    [
        (DistanceStrategy.COSINE, [0, 1, 2], [1, 0.5, 0]),
        (DistanceStrategy.EUCLIDEAN, [0, 1, 9], [1, 0.5, 0.1]),
        (DistanceStrategy.INNER_PRODUCT, [-1000, 0, 1000], [1, 0.5, 0]),
    ],
)
async def test_scores_and_threshold_retrievers(store, strategy, distances, expected):
    store.distance_strategy = strategy
    rows = [(Document(page_content=str(i)), d) for i, d in enumerate(distances)]
    store.asimilarity_search_with_score = AsyncMock(return_value=rows)
    results = await store.asimilarity_search_with_relevance_scores("query")
    assert [score for _, score in results] == pytest.approx(expected)
    retriever = store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6}
    )
    assert await retriever.ainvoke("query") == [rows[0][0]]
    if isinstance(store, CockroachDBVectorStore):
        store.similarity_search_with_score = Mock(return_value=rows)
        assert store.similarity_search_with_relevance_scores("query") == results
        assert retriever.invoke("query") == [rows[0][0]]


async def test_hybrid_requires_explicit_conversion_before_search(store):
    store.hybrid_search_config = HybridSearchConfig()
    search = AsyncMock()
    store.asimilarity_search_with_score = search
    with pytest.raises(ValueError, match="Hybrid relevance searches require"):
        await store.asimilarity_search_with_relevance_scores("query")
    search.assert_not_called()
    if isinstance(store, CockroachDBVectorStore):
        with pytest.raises(ValueError, match="Hybrid relevance searches require"):
            store.similarity_search_with_relevance_scores("query")
    store._relevance_score_fn = lambda score: score * 2
    search.return_value = [(Document(page_content="doc"), 0.4)]
    assert (await store.asimilarity_search_with_relevance_scores("query"))[0][1] == 0.8


@pytest.mark.parametrize("ids", [None, [], ["one"]])
async def test_delete_rejects_unsupported_arguments_before_sql(store, ids):
    store._delete = AsyncMock()
    with pytest.raises(TypeError, match="IDs only"):
        await store.adelete(ids=ids, filter={"source": "manual.pdf"})
    if isinstance(store, CockroachDBVectorStore):
        with pytest.raises(TypeError, match="IDs only"):
            store.delete(ids=ids, filter={"source": "manual.pdf"})
    store._delete.assert_not_called()


def test_custom_conversion_constructor(store):
    convert = lambda distance: 0.75  # noqa: E731
    other = type(store)(store.engine, store.embeddings, "docs", relevance_score_fn=convert)
    assert other._select_relevance_score_fn()(123) == 0.75
    with pytest.raises(TypeError, match="must be callable"):
        type(store)(store.engine, store.embeddings, "docs", relevance_score_fn=42)
