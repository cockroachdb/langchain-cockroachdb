"""Unit tests for index abstractions."""

from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions, DistanceStrategy


class TestDistanceStrategy:
    """Test distance strategy enum."""

    def test_get_operator(self) -> None:
        """Test operator mapping."""
        assert DistanceStrategy.EUCLIDEAN.get_operator() == "<->"
        assert DistanceStrategy.COSINE.get_operator() == "<=>"
        assert DistanceStrategy.INNER_PRODUCT.get_operator() == "<#>"

    def test_get_opclass(self) -> None:
        """Test opclass mapping."""
        assert DistanceStrategy.EUCLIDEAN.get_opclass() == "vector_l2_ops"
        assert DistanceStrategy.COSINE.get_opclass() == "vector_cosine_ops"
        assert DistanceStrategy.INNER_PRODUCT.get_opclass() == "vector_ip_ops"


class TestCSPANNIndex:
    """Test C-SPANN index configuration."""

    def test_default_initialization(self) -> None:
        """Test default index creation."""
        index = CSPANNIndex()
        assert index.distance_strategy == DistanceStrategy.COSINE
        assert index.min_partition_size is None
        assert index.max_partition_size is None
        assert index.name is None

    def test_custom_initialization(self) -> None:
        """Test custom index configuration."""
        index = CSPANNIndex(
            distance_strategy=DistanceStrategy.EUCLIDEAN,
            min_partition_size=100,
            max_partition_size=1000,
            name="custom_idx",
        )
        assert index.distance_strategy == DistanceStrategy.EUCLIDEAN
        assert index.min_partition_size == 100
        assert index.max_partition_size == 1000
        assert index.name == "custom_idx"

    def test_create_index_sql_basic(self) -> None:
        """Test basic CREATE INDEX SQL generation."""
        index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
        sql = index.get_create_index_sql("test_table", "embedding")

        assert "CREATE VECTOR INDEX IF NOT EXISTS" in sql
        assert "test_table_embedding_vector_idx" in sql
        assert "public.test_table" in sql
        assert "(embedding)" in sql

    def test_create_index_sql_with_options(self) -> None:
        """Test CREATE INDEX SQL with partition options."""
        index = CSPANNIndex(
            distance_strategy=DistanceStrategy.EUCLIDEAN,
            min_partition_size=50,
            max_partition_size=500,
        )
        sql = index.get_create_index_sql("test_table", "embedding")

        assert "WITH" in sql
        assert "min_partition_size = 50" in sql
        assert "max_partition_size = 500" in sql

    def test_create_index_sql_with_prefix_columns(self) -> None:
        """Test CREATE INDEX SQL with prefix columns for multi-tenancy."""
        index = CSPANNIndex()
        sql = index.get_create_index_sql(
            "test_table",
            "embedding",
            prefix_columns=["tenant_id", "collection_id"],
        )

        assert "tenant_id, collection_id, embedding" in sql

    def test_create_index_sql_custom_schema(self) -> None:
        """Test CREATE INDEX SQL with custom schema."""
        index = CSPANNIndex()
        sql = index.get_create_index_sql(
            "test_table",
            "embedding",
            schema="custom_schema",
        )

        assert "custom_schema.test_table" in sql

    def test_drop_index_sql(self) -> None:
        """Test DROP INDEX SQL generation."""
        index = CSPANNIndex()
        sql = index.get_drop_index_sql("test_table", "embedding")

        assert "DROP INDEX IF EXISTS" in sql
        assert "public.test_table_embedding_vector_idx" in sql


class TestCSPANNQueryOptions:
    """Test C-SPANN query options."""

    def test_default_initialization(self) -> None:
        """Test default query options."""
        options = CSPANNQueryOptions()
        assert options.beam_size is None
        assert options.get_session_settings() == {}

    def test_beam_size_setting(self) -> None:
        """Test beam size configuration."""
        options = CSPANNQueryOptions(beam_size=100)
        settings = options.get_session_settings()

        assert settings["vector_search_beam_size"] == 100

    def test_multiple_beam_sizes(self) -> None:
        """Test different beam size values."""
        for beam_size in [10, 50, 100, 500]:
            options = CSPANNQueryOptions(beam_size=beam_size)
            settings = options.get_session_settings()
            assert settings["vector_search_beam_size"] == beam_size
