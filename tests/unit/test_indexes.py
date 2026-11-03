"""Unit tests for C-SPANN index configuration."""

import pytest

from langchain_cockroachdb.indexes import (
    CSPANNIndex,
    CSPANNQueryOptions,
    DistanceStrategy,
    _find_vector_index,
    _index_leads_with,
    _usable_vector_index,
    _vector_indexes,
)


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
        assert (
            sql
            == "CREATE VECTOR INDEX IF NOT EXISTS my_idx ON app.docs (embedding vector_cosine_ops)"
        )

    def test_drop_sql(self) -> None:
        assert (
            CSPANNIndex().get_drop_index_sql("docs", "embedding")
            == "DROP INDEX IF EXISTS public.docs_embedding_vector_idx"
        )

    def test_index_name(self) -> None:
        assert CSPANNIndex().get_index_name("docs", "embedding") == "docs_embedding_vector_idx"
        assert CSPANNIndex(name="x").get_index_name("docs", "embedding") == "x"

    @pytest.mark.parametrize(
        "kwargs", [{"name": "bad name"}, {"min_partition_size": 0}, {"max_partition_size": True}]
    )
    def test_rejects_unsafe_values(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CSPANNIndex(**kwargs)


class TestCSPANNQueryOptions:
    def test_default_has_no_settings(self) -> None:
        assert CSPANNQueryOptions().get_session_settings() == {}

    def test_beam_size_setting(self) -> None:
        assert CSPANNQueryOptions(beam_size=64).get_session_settings() == {
            "vector_search_beam_size": 64
        }

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
    def test_rejects_invalid_beam_size(self, bad) -> None:
        with pytest.raises(ValueError):
            CSPANNQueryOptions(beam_size=bad)


def create_statement(
    *index_lines: str,
    namespace_index: bool = True,
    primary_key: str = "CONSTRAINT docs_pkey PRIMARY KEY (id ASC)",
) -> str:
    """A SHOW CREATE TABLE statement as v25.4.14 and v26.3.4 render it, with these index lines.

    ``namespace_index`` adds the index on the namespace column that
    ainit_vectorstore_table creates.
    """
    lines = [
        "\tid UUID NOT NULL DEFAULT gen_random_uuid(),",
        "\tnamespace STRING NOT NULL DEFAULT '':::STRING,",
        '\t"user" STRING NULL,',
        "\tcontent STRING NULL,",
        "\tembedding VECTOR(3) NULL,",
        "\tother VECTOR(3) NULL,",
        "\tmetadata JSONB NULL DEFAULT '{}':::JSONB,",
        f"\t{primary_key},",
        *(["\tINDEX docs_namespace_idx (namespace ASC)"] if namespace_index else []),
        *(f"\t{line}" for line in index_lines),
    ]
    body = ",\n".join(line.rstrip(",") for line in lines)
    return f"CREATE TABLE public.docs (\n{body}\n) WITH (schema_locked = true);"


def usable(statement: str, strategy: DistanceStrategy, prefix: list[str]) -> str | None:
    return _usable_vector_index(
        statement, column="embedding", opclass=strategy.get_opclass(), prefix=prefix
    )


COSINE = DistanceStrategy.COSINE


class TestVectorIndexParsing:
    def test_a_table_without_a_vector_index(self) -> None:
        statement = create_statement()
        assert _vector_indexes(statement) == []
        assert _find_vector_index(statement, "docs_embedding_vector_idx") is None
        assert usable(statement, COSINE, []) is None
        assert usable(statement, COSINE, ["namespace"]) is None

    def test_every_vector_index_with_its_parts(self) -> None:
        statement = create_statement(
            "VECTOR INDEX a_plain (embedding)",
            "VECTOR INDEX c_ns (namespace, embedding vector_ip_ops) "
            "WITH (min_partition_size=8, max_partition_size=64)",
            'VECTOR INDEX "D_Quoted" ("user", embedding vector_l2_ops)',
        )
        found = _vector_indexes(statement)
        assert [index.name for index in found] == ["a_plain", "c_ns", "D_Quoted"]
        assert [index.columns for index in found] == [
            ["embedding"],
            ["namespace", "embedding"],
            ["user", "embedding"],
        ]
        # No operator class means vector_l2_ops, the default. v25.4.14 and
        # v26.3.4 always show one, but a line without one reads the same
        assert [index.opclass for index in found] == [
            "vector_l2_ops",
            "vector_ip_ops",
            "vector_l2_ops",
        ]
        assert found[1].options == {"min_partition_size": "8", "max_partition_size": "64"}
        assert found[0].options == {}

    @pytest.mark.parametrize("strategy", list(DistanceStrategy))
    def test_the_operator_class_must_match_the_distance_strategy(
        self, strategy: DistanceStrategy
    ) -> None:
        for opclass in ("vector_cosine_ops", "vector_l2_ops", "vector_ip_ops"):
            statement = create_statement(f"VECTOR INDEX docs_idx (embedding {opclass})")
            expected = "docs_idx" if opclass == strategy.get_opclass() else None
            assert usable(statement, strategy, []) == expected
        # A line without an operator class has the default, vector_l2_ops
        statement = create_statement("VECTOR INDEX docs_idx (embedding)")
        expected = "docs_idx" if strategy is DistanceStrategy.EUCLIDEAN else None
        assert usable(statement, strategy, []) == expected

    def test_an_index_without_prefix_columns_serves_only_a_store_without_a_namespace(
        self,
    ) -> None:
        statement = create_statement("VECTOR INDEX plain (embedding vector_cosine_ops)")
        assert usable(statement, COSINE, []) == "plain"
        # namespace = ... is a filter too, and filters must match prefix columns
        assert usable(statement, COSINE, ["namespace"]) is None

    def test_a_namespace_prefix_serves_only_a_namespaced_store(self) -> None:
        statement = create_statement("VECTOR INDEX ns (namespace, embedding vector_cosine_ops)")
        assert usable(statement, COSINE, ["namespace"]) == "ns"
        # Every prefix column must be constrained to a value, and a store
        # without a namespace constrains none
        assert usable(statement, COSINE, []) is None
        # Column names compare without regard to case
        assert usable(statement, COSINE, ["NameSpace"]) == "ns"

    def test_other_prefix_columns_serve_neither(self) -> None:
        statement = create_statement(
            'VECTOR INDEX by_user ("user", embedding vector_cosine_ops)',
            'VECTOR INDEX two (namespace, "user", embedding vector_cosine_ops)',
        )
        assert usable(statement, COSINE, []) is None
        assert usable(statement, COSINE, ["namespace"]) is None
        assert usable(statement, COSINE, ["user"]) == "by_user"

    def test_an_index_on_another_vector_column_does_not_count(self) -> None:
        statement = create_statement("VECTOR INDEX docs_other (other vector_cosine_ops)")
        assert usable(statement, COSINE, []) is None

    def test_quoted_names(self) -> None:
        statement = create_statement(
            'VECTOR INDEX "select" ("namespace", "embedding" vector_cosine_ops)',
            'VECTOR INDEX "Mixed ""Case"" idx" (embedding vector_cosine_ops)',
        )
        found = _vector_indexes(statement)
        assert [index.name for index in found] == ["select", 'Mixed "Case" idx']
        assert found[0].columns == ["namespace", "embedding"]
        assert usable(statement, COSINE, ["namespace"]) == "select"
        assert usable(statement, COSINE, []) == 'Mixed "Case" idx'

    def test_several_indexes_pick_one_that_serves(self) -> None:
        statement = create_statement(
            "VECTOR INDEX old_l2 (embedding vector_l2_ops)",
            "VECTOR INDEX by_ns (namespace, embedding vector_cosine_ops)",
            "VECTOR INDEX whole (embedding vector_cosine_ops)",
        )
        assert usable(statement, COSINE, []) == "whole"
        assert usable(statement, COSINE, ["namespace"]) == "by_ns"
        assert usable(statement, DistanceStrategy.EUCLIDEAN, []) == "old_l2"
        assert usable(statement, DistanceStrategy.INNER_PRODUCT, []) is None

    def test_partial_and_invisible_indexes_do_not_serve(self) -> None:
        statement = create_statement(
            "VECTOR INDEX partial (embedding vector_cosine_ops) WHERE namespace = 'x':::STRING",
            "VECTOR INDEX hidden (embedding vector_cosine_ops) NOT VISIBLE",
            "VECTOR INDEX half (embedding vector_cosine_ops) VISIBILITY 0.50",
        )
        found = _vector_indexes(statement)
        assert [index.name for index in found] == ["partial", "hidden", "half"]
        assert [index.partial for index in found] == [True, False, False]
        assert [index.visible for index in found] == [True, False, False]
        assert usable(statement, COSINE, []) is None

    def test_find_vector_index_by_name(self) -> None:
        statement = create_statement(
            "VECTOR INDEX docs_embedding_vector_idx (namespace, embedding vector_cosine_ops) "
            "WITH (min_partition_size=10, max_partition_size=50)",
            'VECTOR INDEX "Quoted" (embedding vector_l2_ops) NOT VISIBLE',
        )
        found = _find_vector_index(statement, "docs_embedding_vector_idx")
        assert found is not None
        assert found.columns == ["namespace", "embedding"]
        assert found.options == {"min_partition_size": "10", "max_partition_size": "50"}
        assert found.definition == (
            "\tVECTOR INDEX docs_embedding_vector_idx (namespace, embedding vector_cosine_ops) "
            "WITH (min_partition_size=10, max_partition_size=50)"
        )
        # Names compare without regard to case, quoted or not
        assert _find_vector_index(statement, "DOCS_EMBEDDING_VECTOR_IDX") is not None
        quoted = _find_vector_index(statement, "quoted")
        assert quoted is not None and quoted.opclass == "vector_l2_ops"
        assert _find_vector_index(statement, "docs_embedding") is None


class TestIndexLeadingWith:
    """Whether the primary key or a forward index can serve namespace = ... with a LIMIT."""

    def test_the_index_ainit_vectorstore_table_creates(self) -> None:
        assert _index_leads_with(create_statement(), "namespace")
        assert _index_leads_with(create_statement(), "NAMESPACE")
        assert not _index_leads_with(create_statement(namespace_index=False), "namespace")

    def test_a_vector_or_inverted_index_does_not_count(self) -> None:
        statement = create_statement(
            "VECTOR INDEX docs_ns_vec (namespace, embedding vector_cosine_ops)",
            "INVERTED INDEX docs_ns_meta (namespace, metadata)",
            namespace_index=False,
        )
        assert not _index_leads_with(statement, "namespace")

    def test_the_primary_key(self) -> None:
        statement = create_statement(
            namespace_index=False,
            primary_key="CONSTRAINT docs_pkey PRIMARY KEY (namespace ASC, id ASC)",
        )
        assert _index_leads_with(statement, "namespace")
        assert _index_leads_with(create_statement(namespace_index=False), "id")

    @pytest.mark.parametrize(
        ("line", "leads"),
        [
            ("UNIQUE INDEX docs_u (namespace ASC, id ASC)", True),
            ("INDEX docs_s (namespace ASC) STORING (content)", True),
            ("INDEX docs_h (namespace ASC) USING HASH WITH (bucket_count=16)", True),
            ('INDEX "Docs Quoted" ("namespace" ASC)', True),
            ("INDEX docs_second (id ASC, namespace ASC)", False),
            ("INDEX docs_p (namespace ASC) WHERE content IS NOT NULL", False),
            ("INDEX docs_hidden (namespace ASC) NOT VISIBLE", False),
            ("INDEX docs_half (namespace ASC) VISIBILITY 0.50", False),
            ("INDEX docs_expr ((lower(namespace)) ASC)", False),
            ("INDEX docs_ns_extra (namespace_id ASC)", False),
        ],
    )
    def test_forward_indexes(self, line: str, leads: bool) -> None:
        statement = create_statement(line, namespace_index=False)
        assert _index_leads_with(statement, "namespace") is leads
