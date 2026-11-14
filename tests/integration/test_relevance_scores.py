"""Threshold retrieval against actual CockroachDB distance operators."""

import uuid

import pytest
from langchain_core.embeddings import Embeddings

from langchain_cockroachdb import CockroachDBVectorStore, DistanceStrategy


class AxisEmbeddings(Embeddings):
    def embed_documents(self, texts):
        vectors = {"near": [1.0, 0.0], "orthogonal": [0.0, 1.0], "opposite": [-1.0, 0.0]}
        return [vectors[text] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.mark.parametrize("strategy", list(DistanceStrategy))
async def test_threshold_retrieval(cockroachdb_engine, strategy):
    table = "relevance_" + uuid.uuid4().hex
    await cockroachdb_engine.ainit_vectorstore_table(table, 2)
    store = CockroachDBVectorStore(
        cockroachdb_engine, AxisEmbeddings(), table, distance_strategy=strategy
    )
    try:
        await store.aadd_texts(["near", "orthogonal", "opposite"])
        rows = await store.asimilarity_search_with_relevance_scores("query", k=3)
        assert [doc.page_content for doc, _ in rows] == ["near", "orthogonal", "opposite"]
        assert all(0 <= score <= 1 for _, score in rows)
        threshold = (rows[0][1] + rows[1][1]) / 2
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": threshold},
        )
        assert [doc.page_content for doc in await retriever.ainvoke("query")] == ["near"]
        assert [doc.page_content for doc in retriever.invoke("query")] == ["near"]
    finally:
        await cockroachdb_engine.adrop_table(table)
