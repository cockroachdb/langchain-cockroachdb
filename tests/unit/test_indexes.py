"""Unit tests for C-SPANN index configuration."""

import pytest

from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions, DistanceStrategy


class TestDistanceStrategy:
    def test_get_operator(self) -> None:
        assert DistanceStrategy.COSINE.get_operator() == "<=>"
        assert DistanceStrategy.EUCLIDEAN.get_operator() == "<->"
        assert DistanceStrategy.INNER_PRODUCT.get_operator() == "<#>"

    def test_get_opclass(self) -> None:
        assert DistanceStrategy.COSINE.get_opclass() == "vector_cosine_ops"
        assert DistanceStrategy.EUCLIDEAN.get_opclass() == "vector_l2_ops"
        assert DistanceStrategy.INNER_PRODUCT.get_opclass() == "vector_ip_ops"


class TestCSPANNIndex:
    def test_defaults(self) -> None:
        index = CSPANNIndex()
        assert index.distance_strategy == DistanceStrategy.COSINE
        assert index.min_partition_size is None
        assert index.max_partition_size is None
        assert index.name is None

    @pytest.mark.parametrize("strategy", list(DistanceStrategy))
    def test_create_sql_names_the_operator_class(self, strategy: DistanceStrategy) -> None:
        sql = CSPANNIndex(distance_strategy=strategy).get_create_index_sql("docs", "embedding")
        assert sql == (
            "CREATE VECTOR INDEX IF NOT EXISTS docs_embedding_vector_idx "
            f"ON public.docs (embedding {strategy.get_opclass()})"
        )

    def test_create_sql_with_options(self) -> None:
        sql = CSPANNIndex(min_partition_size=10, max_partition_size=100).get_create_index_sql(
            "docs", "embedding"
        )
        assert sql.endswith(
            "(embedding vector_cosine_ops) WITH (min_partition_size = 10, max_partition_size = 100)"
        )

    def test_create_sql_with_prefix_columns(self) -> None:
        sql = CSPANNIndex().get_create_index_sql(
            "docs", "embedding", prefix_columns=["tenant_id", "collection_id"]
        )
        assert "(tenant_id, collection_id, embedding vector_cosine_ops)" in sql

    def test_create_sql_custom_schema_and_name(self) -> None:
        sql = CSPANNIndex(name="my_idx").get_create_index_sql("docs", "embedding", schema="app")
        assert sql == "CREATE VECTOR INDEX IF NOT EXISTS my_idx ON app.docs (embedding vector_cosine_ops)"

    def test_drop_sql(self) -> None:
        assert (
            CSPANNIndex().get_drop_index_sql("docs", "embedding")
            == "DROP INDEX IF EXISTS public.docs_embedding_vector_idx"
        )

    def test_index_name(self) -> None:
        assert CSPANNIndex().get_index_name("docs", "embedding") == "docs_embedding_vector_idx"
        assert CSPANNIndex(name="x").get_index_name("docs", "embedding") == "x"

    @pytest.mark.parametrize("kwargs", [{"name": "bad name"}, {"min_partition_size": 0}, {"max_partition_size": True}])
    def test_rejects_unsafe_values(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CSPANNIndex(**kwargs)


class TestCSPANNQueryOptions:
    def test_default_has_no_settings(self) -> None:
        assert CSPANNQueryOptions().get_session_settings() == {}

    def test_beam_size_setting(self) -> None:
        assert CSPANNQueryOptions(beam_size=64).get_session_settings() == {"vector_search_beam_size": 64}

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
    def test_rejects_invalid_beam_size(self, bad) -> None:
        with pytest.raises(ValueError):
            CSPANNQueryOptions(beam_size=bad)
