"""Vector store operations survive real CockroachDB faults."""

import asyncio
import logging
import time
import uuid

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import event, text

from langchain_cockroachdb import (
    RETRYABLE_SQLSTATES,
    AsyncCockroachDBVectorStore,
    HybridSearchConfig,
    RetryPolicy,
    get_sqlstate,
    is_retryable_error,
)
from tests.integration.faults import FaultInjector, hold_row_lock, kill_sessions

DIM = 8
RETRYABLE_CODES = ["40001", "40003", "57P01", "58C00", "58C01", "53300", "08006", "08C00"]
NON_RETRYABLE_CODES = ["57014", "XXA00", "42P01"]
FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.05)


async def make_store(engine, **kwargs) -> AsyncCockroachDBVectorStore:
    table = f"faults_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM, create_tsvector=True)
    return AsyncCockroachDBVectorStore(
        engine, DeterministicFakeEmbedding(size=DIM), table, **kwargs
    )


@pytest_asyncio.fixture
async def faulty(engine_factory, unsafe_internals):
    engine = engine_factory(retry_policy=FAST)
    store = await make_store(engine)
    faults = FaultInjector(engine.engine, unsafe_internals=unsafe_internals)
    yield store, faults
    faults.close()
    await engine.adrop_table(store.collection_name)


@pytest.mark.parametrize("code", RETRYABLE_CODES)
async def test_insert_recovers_from_retryable_errors(faulty, code: str) -> None:
    store, faults = faulty
    faults.fail_next(code, match="insert into")
    ids = await store.aadd_texts(["hello"])
    assert faults.injected == [code]
    assert faults.count("insert into") == 2
    assert [d.page_content for d in await store.aget_by_ids(ids)] == ["hello"]


async def test_search_get_and_delete_recover(faulty) -> None:
    store, faults = faulty
    ids = await store.aadd_texts(["hello", "world"])
    faults.fail_next("40001", match="as distance")
    assert len(await store.asimilarity_search("hello", k=2)) == 2
    faults.fail_next("58C01", match="select id, content, metadata from")
    assert len(await store.aget_by_ids(ids)) == 2
    faults.fail_next("40003", match="delete from")
    assert await store.adelete([ids[0]]) is True
    assert len(await store.aget_by_ids(ids)) == 1
    assert faults.injected == ["40001", "58C01", "40003"]


async def test_hybrid_full_text_leg_recovers(faulty) -> None:
    store, faults = faulty
    hybrid = AsyncCockroachDBVectorStore(
        store.engine,
        store.embeddings,
        store.collection_name,
        hybrid_search_config=HybridSearchConfig(),
    )
    await hybrid.aadd_texts(["cockroach database", "unrelated"])
    faults.fail_next("57P01", match="plainto_tsquery")
    assert await hybrid.asimilarity_search("cockroach", k=2)
    assert faults.injected == ["57P01"]


@pytest.mark.parametrize("code", NON_RETRYABLE_CODES)
async def test_non_retryable_errors_are_raised_after_one_attempt(faulty, code: str) -> None:
    store, faults = faulty
    faults.fail_next(code, match="insert into")
    with pytest.raises(Exception) as info:
        await store.aadd_texts(["hello"])
    assert get_sqlstate(info.value) == code
    assert faults.count("insert into") == 1


async def test_gives_up_after_max_attempts(faulty, caplog) -> None:
    store, faults = faulty
    faults.fail_next("40001", times=5, match="insert into")
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        pytest.raises(Exception) as info,
    ):
        await store.aadd_texts(["hello"])
    assert get_sqlstate(info.value) == "40001"
    assert faults.count("insert into") == 5
    assert "Giving up" in caplog.text


async def test_real_serialization_failure_from_cockroachdb(engine_factory) -> None:
    """inject_retry_errors_enabled makes CockroachDB itself return 40001 in explicit transactions."""
    engine = engine_factory(pool_size=1, max_overflow=0, retry_policy=FAST)
    store = await make_store(engine)
    state = {"failures": 0, "turn_off": False}

    def on_error(ctx) -> None:  # type: ignore[no-untyped-def]
        if get_sqlstate(ctx.original_exception) == "40001":
            state["failures"] += 1
            state["turn_off"] = True

    def before(conn, cursor, statement, params, context, executemany) -> None:  # type: ignore[no-untyped-def]
        if state["turn_off"] and not statement.lstrip().upper().startswith("SET"):
            cursor.execute("SET inject_retry_errors_enabled = 'false'")
            state["turn_off"] = False

    sync_engine = engine.engine.sync_engine
    event.listen(sync_engine, "handle_error", on_error)
    event.listen(sync_engine, "before_cursor_execute", before)
    try:
        async with engine.engine.connect() as conn:
            await conn.execute(text("SET inject_retry_errors_enabled = 'true'"))
            await conn.commit()
        await store.aadd_texts(["written once"])
        assert state["failures"] == 1
        async with engine.engine.connect() as conn:
            assert (await conn.execute(text(f"SELECT count(*) FROM {store._fqn}"))).scalar() == 1
    finally:
        event.remove(sync_engine, "handle_error", on_error)
        event.remove(sync_engine, "before_cursor_execute", before)


async def test_recovers_from_killed_connections(engine_factory, libpq_url) -> None:
    app = f"vs_kill_{uuid.uuid4().hex[:6]}"
    engine = engine_factory(
        app, pool_pre_ping=False, pool_size=2, max_overflow=0, retry_policy=FAST
    )
    store = await make_store(engine)
    ids = await store.aadd_texts(["before"])
    assert kill_sessions(libpq_url, app) >= 1
    await asyncio.sleep(0.3)
    assert [d.page_content for d in await store.aget_by_ids(ids)] == ["before"]
    assert kill_sessions(libpq_url, app) >= 1
    await asyncio.sleep(0.3)
    await store.aadd_texts(["after"])
    assert len(await store.asimilarity_search("x", k=5)) == 2


async def test_attempt_timeout_recovers_once_the_lock_is_released(
    engine_factory, libpq_url, caplog
) -> None:
    engine = engine_factory(
        retry_policy=RetryPolicy(attempt_timeout=0.5, initial_backoff=0.05, max_attempts=8)
    )
    store = await make_store(engine)
    ids = await store.aadd_texts(["original"])
    started = time.monotonic()
    lock_sql = f"UPDATE {store._fqn} SET content = 'locked' WHERE id = %s"
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        hold_row_lock(libpq_url, lock_sql, (ids[0],), release_after=1.2),
    ):
        await store.aadd_texts(["updated"], ids=ids)
    assert time.monotonic() - started >= 1.0
    assert "attempt timeout" in caplog.text
    [doc] = await store.aget_by_ids(ids)
    assert doc.page_content == "updated"


async def test_attempt_timeout_gives_up_with_clear_log(engine_factory, libpq_url, caplog) -> None:
    engine = engine_factory(
        retry_policy=RetryPolicy(attempt_timeout=0.3, initial_backoff=0.01, max_attempts=2)
    )
    store = await make_store(engine)
    ids = await store.aadd_texts(["original"])
    lock_sql = f"UPDATE {store._fqn} SET content = 'locked' WHERE id = %s"
    with (
        caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.retry"),
        hold_row_lock(libpq_url, lock_sql, (ids[0],)),
        pytest.raises(Exception) as info,
    ):
        await store.aadd_texts(["updated"], ids=ids)
    assert is_retryable_error(info.value)
    assert caplog.text.count("attempt timeout") == 2
    assert "Giving up" in caplog.text


async def test_statement_timeouts_are_not_retried_by_default(
    engine_factory, libpq_url, unsafe_internals
) -> None:
    setup = engine_factory()
    store = await make_store(setup)
    ids = await store.aadd_texts(["original"])
    engine = engine_factory(
        connect_args={"options": "-c statement_timeout=300ms"}, retry_policy=FAST
    )
    timed = AsyncCockroachDBVectorStore(engine, store.embeddings, store.collection_name)
    faults = FaultInjector(engine.engine, unsafe_internals=unsafe_internals)
    lock_sql = f"UPDATE {store._fqn} SET content = 'locked' WHERE id = %s"
    try:
        with hold_row_lock(libpq_url, lock_sql, (ids[0],)), pytest.raises(Exception) as info:
            await timed.aadd_texts(["updated"], ids=ids)
        assert get_sqlstate(info.value) == "57014"
        assert faults.count("insert into") == 1
    finally:
        faults.close()


async def test_statement_timeouts_can_be_opted_in(engine_factory, libpq_url) -> None:
    setup = engine_factory()
    store = await make_store(setup)
    ids = await store.aadd_texts(["original"])
    policy = RetryPolicy(
        retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"}, initial_backoff=0.05, max_attempts=10
    )
    engine = engine_factory(
        connect_args={"options": "-c statement_timeout=300ms"}, retry_policy=policy
    )
    timed = AsyncCockroachDBVectorStore(engine, store.embeddings, store.collection_name)
    lock_sql = f"UPDATE {store._fqn} SET content = 'locked' WHERE id = %s"
    with hold_row_lock(libpq_url, lock_sql, (ids[0],), release_after=1.0):
        await timed.aadd_texts(["updated"], ids=ids)
    [doc] = await store.aget_by_ids(ids)
    assert doc.page_content == "updated"


def test_vector_store_inherits_and_overrides_the_engine_policy(engine_factory) -> None:
    engine = engine_factory(retry_policy=RetryPolicy(max_attempts=9))
    inherited = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), "t")
    assert inherited.retry_policy is engine.retry_policy
    overridden = AsyncCockroachDBVectorStore(
        engine, DeterministicFakeEmbedding(size=DIM), "t", retry_max_attempts=2
    )
    assert overridden.retry_max_attempts == 2
    assert overridden.retry_policy.retryable_sqlstates == engine.retry_policy.retryable_sqlstates
    with pytest.raises(ValueError):
        AsyncCockroachDBVectorStore(
            engine,
            DeterministicFakeEmbedding(size=DIM),
            "t",
            retry_policy=RetryPolicy(),
            retry_max_attempts=2,
        )


async def test_hybrid_search_fts_leg_retries_and_recovers(faulty) -> None:
    """Full-text search leg retries transient errors and hybrid search succeeds."""
    store, faults = faulty
    hybrid = AsyncCockroachDBVectorStore(
        store.engine,
        store.embeddings,
        store.collection_name,
        hybrid_search_config=HybridSearchConfig(),
    )
    await hybrid.aadd_texts(["cockroach database", "unrelated"])
    faults.fail_next("40001", match="plainto_tsquery")
    results = await hybrid.asimilarity_search("cockroach", k=2)
    assert len(results) > 0
    assert faults.count("plainto_tsquery") == 2
    assert faults.injected == ["40001"]


async def test_opclass_mismatch_warning_does_not_change_index(engine_factory, caplog) -> None:
    """After the opclass mismatch warning, the existing index is unchanged."""
    from langchain_cockroachdb import CSPANNIndex, DistanceStrategy

    engine = engine_factory()
    table = f"opclass_{uuid.uuid4().hex[:8]}"
    await engine.ainit_vectorstore_table(table, DIM)
    store = AsyncCockroachDBVectorStore(engine, DeterministicFakeEmbedding(size=DIM), table)
    await store.aadd_texts(["a"])
    await store.aapply_vector_index(CSPANNIndex(distance_strategy=DistanceStrategy.EUCLIDEAN))
    index_name = f"{table}_embedding_vector_idx"
    async with engine.engine.connect() as conn:
        indexdef_before = (
            await conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
                {"name": index_name},
            )
        ).scalar()
    assert "vector_l2_ops" in indexdef_before
    with caplog.at_level(logging.WARNING, logger="langchain_cockroachdb.async_vectorstore"):
        await store.aapply_vector_index(CSPANNIndex(distance_strategy=DistanceStrategy.COSINE))
    assert any("vector_cosine_ops" in record.message for record in caplog.records)
    async with engine.engine.connect() as conn:
        indexdef_after = (
            await conn.execute(
                text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
                {"name": index_name},
            )
        ).scalar()
    assert indexdef_after == indexdef_before
    assert "vector_l2_ops" in indexdef_after
    await engine.adrop_table(table)


async def test_ambiguous_commit_does_not_duplicate_rows(faulty) -> None:
    """Write applies on server, client sees ambiguous error, retry doesn't duplicate."""
    store, faults = faulty
    faults.fail_next("40003", match="insert into")
    ids = await store.aadd_texts(["once"])
    assert faults.injected == ["40003"]
    assert faults.count("insert into") == 2
    async with store.engine.engine.connect() as conn:
        count = (await conn.execute(text(f"SELECT count(*) FROM {store._fqn}"))).scalar()
    assert count == 1
    [doc] = await store.aget_by_ids(ids)
    assert doc.page_content == "once"
