"""Correctness of the vector store's SQL: binding, filters, beam size and indexes."""

import datetime
import logging
import uuid

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CSPANNIndex,
    CSPANNQueryOptions,
    DistanceStrategy,
    get_sqlstate,
    is_retryable_error,
)
from tests.integration.faults import FaultInjector

DIM = 8


@pytest_asyncio.fixture
async def store(engine_factory):
    engine = engine_factory()
    table = f"hardening_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    yield AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await engine.adrop_table(table)


async def names(store: AsyncCockroachDBVectorStore, filter: dict) -> list[str]:
    return sorted(d.page_content for d in await store.asimilarity_search("q", k=10, filter=filter))


async def test_special_characters_round_trip(store) -> None:
    content = '100% O\'Brien \\ said "hi" :colon 🪳'
    metadata = {"who": "Trader Joe's", "a:b": "x:y", "pct": "50%", "quote": 'say "x"'}
    ids = await store.aadd_texts([content, "other"], metadatas=[metadata, {"who": "else"}])
    [doc] = await store.aget_by_ids([ids[0]])
    assert doc.page_content == content
    assert doc.metadata == metadata
    for filter in (
        {"who": "Trader Joe's"},
        {"a:b": "x:y"},
        {"pct": {"$like": "50%"}},
        {"quote": 'say "x"'},
    ):
        assert await names(store, filter) == [content], filter


async def test_bool_and_null_filters(store) -> None:
    await store.aadd_texts(
        ["t", "f", "n"], metadatas=[{"flag": True}, {"flag": False}, {"flag": None}]
    )
    assert await names(store, {"flag": True}) == ["t"]
    assert await names(store, {"flag": None}) == ["n"]


async def test_every_operator_against_the_database(store) -> None:
    await store.aadd_texts(
        ["a", "b", "c", "d"],
        metadatas=[
            {"n": 1, "s": "apple", "tag": "x", "day": "2024-01-01"},
            {"n": 5, "s": "banana", "tag": "y", "day": "2024-06-01"},
            {"n": "text", "s": "cherry", "tag": "x"},
            {"s": "date"},
        ],
    )
    assert await names(store, {"n": {"$gt": 2}}) == ["b"]
    assert await names(store, {"n": {"$between": [1, 5]}}) == ["a", "b"]
    assert await names(store, {"tag": {"$in": ["x"]}}) == ["a", "c"]
    assert await names(store, {"tag": {"$nin": ["x"]}}) == ["b"]
    assert await names(store, {"n": {"$exists": False}}) == ["d"]
    assert await names(store, {"s": {"$ilike": "%AN%"}}) == ["b"]
    assert await names(store, {"day": {"$gte": "2024-03-01"}}) == ["b"]
    assert await names(store, {"$or": [{"s": "apple"}, {"s": "date"}]}) == ["a", "d"]
    assert await names(store, {"$not": {"tag": "x"}}) == ["b"]
    assert await names(store, {"$or": [{"tag": "y"}, {"tag": "x"}], "s": "banana"}) == ["b"]


async def test_invalid_uuid_id_fails_once(store, unsafe_internals) -> None:
    faults = FaultInjector(store.engine.engine, unsafe_internals=unsafe_internals)
    try:
        with pytest.raises(Exception) as info:
            await store.aadd_texts(["x"], ids=["doc-1"])
        assert (get_sqlstate(info.value) or "").startswith("22")
        assert not is_retryable_error(info.value)
        assert faults.count("insert into") == 1
    finally:
        faults.close()


async def test_unserializable_metadata_fails_before_writing(store) -> None:
    metadatas = [{"i": i} for i in range(5)] + [{"when": datetime.datetime.now()}]
    with pytest.raises(TypeError, match="text 5"):
        await store.aadd_texts([f"t{i}" for i in range(6)], metadatas=metadatas, batch_size=2)
    assert await store.asimilarity_search("q", k=10) == []
    with pytest.raises(TypeError, match="text 1"):
        await store.aadd_texts(["a", "b"], metadatas=[{}, {"nan": float("nan")}], batch_size=1)
    assert await store.asimilarity_search("q", k=10) == []
    with pytest.raises(TypeError, match="text 0"):
        await store.aadd_texts(["a"], metadatas=[{"inf": float("inf")}])
    assert await store.asimilarity_search("q", k=10) == []


async def test_invalid_batch_size_is_rejected(store) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        await store.aadd_texts(["a"], batch_size=0)


async def test_mismatched_lengths_are_rejected(store) -> None:
    with pytest.raises(ValueError):
        await store.aadd_texts(["a", "b"], ids=[str(uuid.uuid4())])
    with pytest.raises(ValueError):
        await store.aadd_texts(["a", "b"], metadatas=[{}])


async def test_duplicate_ids_are_rejected_before_writing(store) -> None:
    dup_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match=f"Duplicate id in one add call: {dup_id}"):
        await store.aadd_texts(
            ["a", "b", "c", "d"],
            ids=[str(uuid.uuid4()), dup_id, str(uuid.uuid4()), dup_id],
            batch_size=2,
        )
    assert await store.asimilarity_search("q", k=10) == []


async def test_beam_size_does_not_leak_into_the_pool(engine_factory) -> None:
    engine = engine_factory(pool_size=1, max_overflow=0)
    table = f"beam_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aadd_texts(["a"])
    async with engine.engine.connect() as conn:
        default = (await conn.execute(text("SHOW vector_search_beam_size"))).scalar()
    await store.asimilarity_search_with_score(
        "a", k=1, query_options=CSPANNQueryOptions(beam_size=77)
    )
    async with engine.engine.connect() as conn:
        assert (await conn.execute(text("SHOW vector_search_beam_size"))).scalar() == default
    await engine.adrop_table(table)


@pytest.mark.parametrize("strategy", list(DistanceStrategy))
async def test_vector_index_is_used_for_its_distance_strategy(engine_factory, strategy) -> None:
    engine = engine_factory()
    table = f"idx_{strategy.value}_{uuid.uuid4().hex[:6]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(
        engine, DeterministicFakeEmbedding(size=DIM), table, distance_strategy=strategy
    )
    await store.aadd_texts(["a", "b"])
    await store.aapply_vector_index(CSPANNIndex(distance_strategy=strategy))
    query_vector = "[" + ",".join(["0.1"] * DIM) + "]"
    async with engine.engine.connect() as conn:
        rows = await conn.execute(
            text(
                f"EXPLAIN SELECT id FROM {table} ORDER BY embedding {strategy.get_operator()} CAST(:q AS VECTOR) LIMIT 2"
            ),
            {"q": query_vector},
        )
        plan = "\n".join(row[0] for row in rows.fetchall())
    assert "vector search" in plan
    await engine.adrop_table(table)


async def test_existing_index_with_the_wrong_operator_class_warns(engine_factory, caplog) -> None:
    engine = engine_factory()
    table = f"oldidx_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    async with engine.engine.begin() as conn:
        # What 0.3.0 created for a cosine store: the default vector_l2_ops
        await conn.execute(
            text(f"CREATE VECTOR INDEX {table}_embedding_vector_idx ON {table} (embedding)")
        )
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex())
    assert "vector_cosine_ops" in caplog.text and "Drop it" in caplog.text
    await store.adrop_vector_index(CSPANNIndex())
    await store.aapply_vector_index(CSPANNIndex())
    async with engine.engine.connect() as conn:
        indexdef = (
            await conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE tablename = :t AND indexname = :n"),
                {"t": table, "n": f"{table}_embedding_vector_idx"},
            )
        ).scalar()
    assert "vector_cosine_ops" in indexdef
    await engine.adrop_table(table)


async def test_existing_index_with_wrong_opclass_warns_for_mixed_case(
    engine_factory, caplog
) -> None:
    engine = engine_factory()
    table = f"T8Rev_{uuid.uuid4().hex[:6]}"
    await engine.ainit_vectorstore_table(table, DIM)
    async with engine.engine.begin() as conn:
        await conn.execute(
            text(f"CREATE VECTOR INDEX {table}_embedding_vector_idx ON {table} (embedding)")
        )
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex())
    assert "vector_cosine_ops" in caplog.text and "Drop it" in caplog.text
    await engine.adrop_table(table)


def test_unsafe_identifiers_are_rejected(engine_factory) -> None:
    engine = engine_factory()
    with pytest.raises(ValueError):
        AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), "bad-name")
    with pytest.raises(ValueError):
        AsyncCockroachDBVectorStore(
            engine, DeterministicFakeEmbedding(size=DIM), "ok", content_column="x; drop"
        )


async def test_index_ddl_commits_with_autocommit_before_ddl_off(engine_factory) -> None:
    engine = engine_factory(connect_args={"options": "-c autocommit_before_ddl=off"})
    table = f"ddl_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aadd_texts(["a"])
    await store.aapply_vector_index(CSPANNIndex())
    async with engine.engine.connect() as conn:
        indexdef = (
            await conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE tablename = :t AND indexname = :n"),
                {"t": table, "n": f"{table}_embedding_vector_idx"},
            )
        ).scalar()
    assert indexdef is not None
    assert "vector_cosine_ops" in indexdef
    await store.adrop_vector_index(CSPANNIndex())
    async with engine.engine.connect() as conn:
        exists = (
            await conn.execute(
                text("SELECT 1 FROM pg_indexes WHERE tablename = :t AND indexname = :n"),
                {"t": table, "n": f"{table}_embedding_vector_idx"},
            )
        ).scalar()
    assert exists is None
    await engine.adrop_table(table)


async def test_index_warns_on_prefix_column_difference(engine_factory, caplog) -> None:
    """Applying an index with prefix columns when one without them exists warns."""
    engine = engine_factory()
    table = f"prefix_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aadd_texts(["a"])
    await store.aapply_vector_index(CSPANNIndex())
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex(), prefix_columns=["id"])
    assert any(
        "prefix columns = id (index has none)" in record.message for record in caplog.records
    )
    assert any("adrop_vector_index" in record.message for record in caplog.records)
    await engine.adrop_table(table)


async def test_index_warns_on_partition_size_difference(engine_factory, caplog) -> None:
    """Applying an index with different partition sizes when one exists warns."""
    engine = engine_factory()
    table = f"partition_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aadd_texts(["a"])
    await store.aapply_vector_index(CSPANNIndex(min_partition_size=100))
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex(min_partition_size=200))
    assert any("min_partition_size = 200" in record.message for record in caplog.records)
    assert any("adrop_vector_index" in record.message for record in caplog.records)
    await engine.adrop_table(table)


async def _prefixed_store(engine_factory, prefix: str) -> AsyncCockroachDBVectorStore:
    """A store whose table has two extra columns, tenant and ns, for prefix tests."""
    engine = engine_factory()
    table = f"{prefix}_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM, namespace_column="tenant")
    async with engine.engine.begin() as conn:
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS ns STRING"))
    return AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)


def _index_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "langchain_cockroachdb.async_vectorstore"
        and record.levelno >= logging.WARNING
    ]


async def test_identical_index_with_partition_sizes_reapplies_silently(
    engine_factory, caplog
) -> None:
    engine = engine_factory()
    table = f"samepart_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    index = CSPANNIndex(min_partition_size=10, max_partition_size=50)
    await store.aapply_vector_index(index)
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex(min_partition_size=10, max_partition_size=50))
        # A size the caller leaves unset accepts whatever the index has
        await store.aapply_vector_index(CSPANNIndex(min_partition_size=10))
        await store.aapply_vector_index(CSPANNIndex())
    assert _index_warnings(caplog) == []
    await engine.adrop_table(table)


async def test_identical_index_with_two_prefix_columns_reapplies_silently(
    engine_factory, caplog
) -> None:
    store = await _prefixed_store(engine_factory, "sameprefix")
    index = CSPANNIndex(min_partition_size=10, max_partition_size=50)
    await store.aapply_vector_index(index, prefix_columns=["tenant", "ns"])
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(index, prefix_columns=["tenant", "ns"])
    assert _index_warnings(caplog) == []
    await store.engine.adrop_table(store.collection_name)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (["ns", "tenant"], "prefix columns = ns, tenant (index has tenant, ns)"),
        (["tenant"], "prefix columns = tenant (index has tenant, ns)"),
        (["tenant", "ns", "id"], "prefix columns = tenant, ns, id (index has tenant, ns)"),
        (None, "prefix columns = none (index has tenant, ns)"),
    ],
)
async def test_index_warns_on_prefix_column_order_and_count(
    engine_factory, caplog, requested, expected
) -> None:
    store = await _prefixed_store(engine_factory, "diffprefix")
    await store.aapply_vector_index(CSPANNIndex(), prefix_columns=["tenant", "ns"])
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex(), prefix_columns=requested)
    warnings = _index_warnings(caplog)
    assert len(warnings) == 1 and expected in warnings[0], warnings
    assert "adrop_vector_index" in warnings[0]
    await store.engine.adrop_table(store.collection_name)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (
            CSPANNIndex(min_partition_size=20, max_partition_size=100),
            "min_partition_size = 20 (index has 10)",
        ),
        (
            CSPANNIndex(min_partition_size=10, max_partition_size=200),
            "max_partition_size = 200 (index has 100)",
        ),
        # A substring check would let 10 match an existing 100
        (CSPANNIndex(max_partition_size=10), "max_partition_size = 10 (index has 100)"),
    ],
)
async def test_index_warns_on_each_partition_size_difference(
    engine_factory, caplog, requested, expected
) -> None:
    engine = engine_factory()
    table = f"diffpart_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aapply_vector_index(CSPANNIndex(min_partition_size=10, max_partition_size=100))
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(requested)
    warnings = _index_warnings(caplog)
    assert len(warnings) == 1 and expected in warnings[0], warnings
    await engine.adrop_table(table)


async def test_index_warns_when_a_requested_size_is_unset_on_the_index(
    engine_factory, caplog
) -> None:
    engine = engine_factory()
    table = f"unsetpart_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aapply_vector_index(CSPANNIndex())
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex(min_partition_size=10))
    warnings = _index_warnings(caplog)
    assert len(warnings) == 1 and "min_partition_size = 10 (index has the default)" in warnings[0]
    await engine.adrop_table(table)
