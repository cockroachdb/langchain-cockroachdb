"""Unit tests for vectorstore methods that don't require a database."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg
import pytest
from psycopg import errors as pg_errors
from sqlalchemy.exc import OperationalError as SAOperationalError
from sqlalchemy.exc import ProgrammingError as SAProgrammingError

import langchain_cockroachdb
from langchain_cockroachdb import (
    VECTOR_INDEX_REFRESH_SECONDS,
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CockroachDBVectorStore,
    CSPANNIndex,
    CSPANNQueryOptions,
    DistanceStrategy,
    HybridSearchConfig,
)
from langchain_cockroachdb import async_vectorstore as vectorstore_module
from langchain_cockroachdb._window import window_needs_exact_search
from langchain_cockroachdb.async_vectorstore import _WINDOW_FACTOR, _check_batch_size
from tests.unit.test_indexes import create_statement


class TestBatchSizeValidation:
    """Test batch_size validation."""

    def test_max_batch_size_boundary(self) -> None:
        """Test that batch_size is capped at 16,383 (65,535 params / 4 params per row)."""
        # The limit is derived from CockroachDB's 65,535 parameter cap
        # Each inserted row uses 4 parameters
        max_allowed = 65535 // 4  # Should be 16,383

        # At the boundary should be OK
        _check_batch_size(max_allowed)

        # One over the boundary should fail
        with pytest.raises(
            ValueError,
            match=r"batch_size \d+ exceeds the maximum of 16383.*65,535 bound parameters",
        ):
            _check_batch_size(max_allowed + 1)

        # Way over should also fail
        with pytest.raises(
            ValueError,
            match=r"batch_size \d+ exceeds the maximum of 16383.*65,535 bound parameters",
        ):
            _check_batch_size(50000)


class TestIdNormalization:
    """Test id normalization in duplicate detection."""

    def test_duplicate_uuid_spellings_rejected(self) -> None:
        """Test that different spellings of the same UUID are rejected as duplicates."""
        import uuid

        from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore

        # Create a UUID and two spellings (lowercase and uppercase)
        u = str(uuid.uuid4())

        # Create a minimal instance with required methods
        class MinimalStore:
            batch_size = 100

            @staticmethod
            def _normalize_id(id_str):
                return AsyncCockroachDBVectorStore._normalize_id(id_str)

            def _validate_inputs(self, *args):
                return AsyncCockroachDBVectorStore._validate_inputs(self, *args)

        mock_store = MinimalStore()

        # This should raise ValueError before writing anything
        with pytest.raises(ValueError, match=r"[Dd]uplicate.*id"):
            AsyncCockroachDBVectorStore._prepare_rows(
                mock_store,
                texts=["text1", "text2"],
                embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                metadatas=[{}, {}],
                ids=[u.lower(), u.upper()],
                kwargs={},
            )


# Filtered search through the C-SPANN index. The store's _execute is replaced
# by FakeDatabase, which answers SHOW CREATE TABLE, the windowed statement and
# today's exact statement, and records what was sent.

UNREACHABLE = "cockroachdb://root@localhost:1/none"
QUERY = [0.1, 0.2, 0.3]
FILTER = {"kind": "x"}
COSINE_INDEX = "VECTOR INDEX docs_embedding_vector_idx (embedding vector_cosine_ops)"
NAMESPACE_INDEX = "VECTOR INDEX docs_ns_idx (namespace, embedding vector_cosine_ops)"


@dataclass
class Sent:
    sql: str
    params: dict[str, Any]
    label: str
    settings: dict[str, int] | None


class FakeDatabase:
    """Stands in for the store's _execute and records every statement."""

    def __init__(self, create: str | None = None) -> None:
        self.create = create
        self.read_error: BaseException | None = None
        self.on_read: Callable[[], None] | None = None
        self.window_rows: list[tuple] = []
        self.exact_rows: list[tuple] = []
        self.sent: list[Sent] = []

    async def execute(
        self,
        statement: Any,
        params: dict[str, Any] | None = None,
        *,
        label: str,
        begin: bool = False,
        ddl: bool = False,
        settings: dict[str, int] | None = None,
        fetch: bool = False,
    ) -> list[Any]:
        sql = statement.text
        self.sent.append(Sent(sql, dict(params or {}), label, settings))
        if "SHOW CREATE TABLE" in sql:
            if self.on_read is not None:
                self.on_read()
            if self.read_error is not None:
                raise self.read_error
            return [(self.create,)] if self.create is not None else []
        if "ts_rank" in sql:
            return []
        if "count(*) OVER ()" in sql:
            return list(self.window_rows)
        if "ORDER BY" in sql:
            return list(self.exact_rows)
        return []

    def reads(self) -> list[Sent]:
        return [s for s in self.sent if "SHOW CREATE TABLE" in s.sql]

    def windowed(self) -> list[Sent]:
        return [s for s in self.sent if "count(*) OVER ()" in s.sql]

    def exact(self) -> list[Sent]:
        return [
            s
            for s in self.sent
            if "AS VECTOR)" in s.sql and "ORDER BY" in s.sql and "OVER ()" not in s.sql
        ]


class QueryEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return list(QUERY)

    async def aembed_query(self, text: str) -> list[float]:
        return list(QUERY)


def hit(i: int, *rest: Any) -> tuple:
    return (f"id{i}", f"text {i}", {"kind": "x"}, round(0.1 * (i + 1), 3), *rest)


def window_rows(count: int, *, window_size: int, scope_rows: int) -> list[tuple]:
    return [hit(i, window_size, scope_rows) for i in range(count)]


def todays_statement(*, namespace: bool, k: int = 4, with_embeddings: bool = False) -> str:
    """The exact statement as 0.5.1 sends it, which the fallback keeps unchanged."""
    distance = "embedding <=> CAST(:query_vector AS VECTOR)"
    if namespace:
        where = "WHERE namespace = :f0 AND metadata->CAST(:f1 AS STRING) = CAST(:f2 AS JSONB)"
    else:
        where = "WHERE metadata->CAST(:f0 AS STRING) = CAST(:f1 AS JSONB)"
    extra = ", embedding" if with_embeddings else ""
    return (
        f"SELECT id, content, metadata, {distance} AS distance{extra} FROM public.docs {where} "
        f"ORDER BY {distance} LIMIT {k}"
    )


def flat(sql: str) -> str:
    """``sql`` on one line, with single spaces and none just inside parentheses."""
    return " ".join(sql.split()).replace("( ", "(").replace(" )", ")")


class FakeClock:
    """Stands in for the time module in async_vectorstore: monotonic() returns ``now``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(vectorstore_module, "time", fake)
    return fake


@pytest.fixture
def engine():
    engine = CockroachDBEngine.from_connection_string(UNREACHABLE)
    yield engine
    engine.close()


@pytest.fixture
def make_store(engine, monkeypatch: pytest.MonkeyPatch):
    def make(
        db: FakeDatabase, cls: type = AsyncCockroachDBVectorStore, **kwargs: Any
    ) -> AsyncCockroachDBVectorStore:
        store = cls(engine, QueryEmbeddings(), "docs", **kwargs)
        monkeypatch.setattr(store, "_execute", db.execute)
        return store

    return make


def server_error(code: str, message: str = "boom") -> psycopg.Error:
    try:
        error = pg_errors.lookup(code)(message)
    except KeyError:
        error = psycopg.OperationalError(message)
    error.sqlstate = code
    return error


class TestWindowRule:
    """The rule both stores use to decide whether a windowed search stands."""

    def test_the_window_factor(self) -> None:
        assert _WINDOW_FACTOR == 2

    @pytest.mark.parametrize(
        ("found", "window_size", "scope_rows", "exact"),
        [
            (4, 9, 9, False),  # k rows: complete
            (4, 9, 40, False),  # complete, however large the scope
            (0, 0, 0, True),  # no rows: the sizes are unknown
            (2, 9, 9, True),  # a full window may have left matches out
            (2, 9, 40, True),
            (2, 6, 9, True),  # the index cut the window short of the scope
            (2, 5, 5, False),  # the window holds the whole scope, so the short result is exact
        ],
    )
    def test_cases(self, found: int, window_size: int, scope_rows: int, exact: bool) -> None:
        assert (
            window_needs_exact_search(
                found, limit=4, window=9, window_size=window_size, scope_rows=scope_rows
            )
            is exact
        )

    def test_the_memory_store_keeps_its_name_for_the_rule(self) -> None:
        from langchain_cockroachdb.store.base import needs_exact_search

        def row(window_size: int, held: int) -> dict[str, int]:
            return {"window_size": window_size, "namespace_vectors": held}

        assert not needs_exact_search([row(5, 5)] * 4, limit=4, over_fetch=9)
        assert needs_exact_search([], limit=4, over_fetch=9)
        assert needs_exact_search([row(9, 9)], limit=4, over_fetch=9)
        assert needs_exact_search([row(6, 9)], limit=4, over_fetch=9)
        assert not needs_exact_search([row(5, 5)], limit=4, over_fetch=9)


class TestWindowedStatement:
    async def test_shape_without_a_namespace(self, make_store) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db)

        results = await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)

        assert [doc.id for doc, _ in results] == ["id0", "id1", "id2", "id3"]
        assert [distance for _, distance in results] == [0.1, 0.2, 0.3, 0.4]
        [sent] = db.windowed()
        assert db.exact() == []
        sql = flat(sent.sql)
        distance = "embedding <=> CAST(:query_vector AS VECTOR)"
        # The inner query is what the index serves: no filter, ORDER BY and LIMIT
        assert (
            f"SELECT id, content, metadata, {distance} AS distance FROM public.docs "
            f"ORDER BY {distance} LIMIT :window" in sql
        )
        # Both counts: the rows the window kept, and the table's rows up to the window
        assert "count(*) OVER () AS window_size" in sql
        assert (
            "(SELECT count(*) FROM (SELECT 1 FROM public.docs LIMIT :window) AS scope) "
            "AS scope_rows" in sql
        )
        # The filter applies to the window, outside it
        assert sql.endswith(
            "WHERE metadata->CAST(:f0 AS STRING) = CAST(:f1 AS JSONB) ORDER BY distance LIMIT :k"
        )
        assert sql.count("metadata->CAST(") == 1
        assert sql.count("FROM public.docs") == 2
        # Every value is a bound parameter
        assert "::" not in sql
        assert '"x"' not in sql and "'x'" not in sql
        assert sent.params == {
            "f0": "kind",
            "f1": '"x"',
            "query_vector": "[0.1,0.2,0.3]",
            "window": _WINDOW_FACTOR * 4 + 1,
            "k": 4,
        }

    async def test_shape_with_a_namespace(self, make_store) -> None:
        db = FakeDatabase(create_statement(NAMESPACE_INDEX))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db, namespace="tenant-a")

        await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)

        [sent] = db.windowed()
        sql = flat(sent.sql)
        distance = "embedding <=> CAST(:query_vector AS VECTOR)"
        # The namespace is the scope of the window and of the count, and the
        # index serves it through its prefix column
        assert f"FROM public.docs WHERE namespace = :f0 ORDER BY {distance} LIMIT :window" in sql
        assert "SELECT 1 FROM public.docs WHERE namespace = :f0 LIMIT :window" in sql
        assert sql.count("namespace = :f0") == 2
        assert sql.endswith(
            "WHERE metadata->CAST(:f1 AS STRING) = CAST(:f2 AS JSONB) ORDER BY distance LIMIT :k"
        )
        assert "::" not in sql
        assert sent.params["f0"] == "tenant-a"
        assert (sent.params["window"], sent.params["k"]) == (9, 4)

    async def test_with_embeddings_selects_them_through_the_window(self, make_store) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = [hit(i, "[0.1,0.2,0.3]", 9, 9) for i in range(4)]
        store = make_store(db)

        docs = await store.amax_marginal_relevance_search_by_vector(
            QUERY, k=2, fetch_k=4, filter=FILTER
        )

        assert len(docs) == 2
        [sent] = db.windowed()
        sql = flat(sent.sql)
        assert "AS distance, embedding FROM public.docs" in sql
        assert "SELECT id, content, metadata, distance, embedding, window_size" in sql
        assert (sent.params["window"], sent.params["k"]) == (9, 4)

    async def test_beam_size_reaches_both_statements(self, make_store) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(1, window_size=9, scope_rows=9)
        store = make_store(db)

        await store.asimilarity_search_with_score_by_vector(
            QUERY, k=4, filter=FILTER, query_options=CSPANNQueryOptions(beam_size=7)
        )

        assert [s.settings for s in db.windowed() + db.exact()] == [
            {"vector_search_beam_size": 7},
            {"vector_search_beam_size": 7},
        ]


class TestWindowDecision:
    @pytest.mark.parametrize(
        ("rows", "falls_back"),
        [
            (window_rows(4, window_size=9, scope_rows=9), False),
            (window_rows(2, window_size=5, scope_rows=5), False),
            (window_rows(2, window_size=9, scope_rows=9), True),
            (window_rows(2, window_size=6, scope_rows=9), True),
            ([], True),
        ],
        ids=["complete", "whole-scope", "full-window", "cut-short", "empty"],
    )
    async def test_the_store_runs_todays_statement_when_the_window_may_miss_matches(
        self, make_store, rows: list[tuple], falls_back: bool
    ) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = rows
        db.exact_rows = [hit(7)]
        store = make_store(db)

        results = await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)

        assert len(db.windowed()) == 1
        if falls_back:
            [exact] = db.exact()
            assert exact.sql == todays_statement(namespace=False)
            assert [doc.id for doc, _ in results] == ["id7"]
        else:
            assert db.exact() == []
            assert [doc.id for doc, _ in results] == [f"id{i}" for i in range(len(rows))]

    @pytest.mark.parametrize("empty", [None, {}])
    async def test_without_a_filter_nothing_new_runs(self, make_store, empty) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.exact_rows = [hit(0)]
        store = make_store(db)

        await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=empty)

        assert db.reads() == [] and db.windowed() == []
        [exact] = db.exact()
        assert "WHERE" not in exact.sql

    @pytest.mark.parametrize(
        ("namespace", "index", "windowed"),
        [
            (None, None, False),
            (None, "VECTOR INDEX old (embedding vector_l2_ops)", False),
            (None, "VECTOR INDEX old (embedding)", False),
            (None, NAMESPACE_INDEX, False),
            (None, COSINE_INDEX, True),
            ("tenant-a", COSINE_INDEX, False),
            ("tenant-a", NAMESPACE_INDEX, True),
            ("tenant-a", "VECTOR INDEX t (namespace, embedding vector_ip_ops)", False),
        ],
    )
    async def test_only_a_usable_index_takes_the_window(
        self, make_store, namespace: str | None, index: str | None, windowed: bool
    ) -> None:
        db = FakeDatabase(create_statement(*([index] if index else [])))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db, namespace=namespace)

        await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)

        assert len(db.reads()) == 1
        if windowed:
            assert len(db.windowed()) == 1 and db.exact() == []
        else:
            assert db.windowed() == []
            [exact] = db.exact()
            assert exact.sql == todays_statement(namespace=namespace is not None)

    @pytest.mark.parametrize("column", ["content_column", "metadata_column", "id_column"])
    @pytest.mark.parametrize("name", ["distance", "Window_Size", "scope_rows"])
    async def test_a_column_named_like_the_windows_own_keeps_todays_statement(
        self, make_store, column: str, name: str
    ) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db, **{column: name})

        await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)

        # The windowed statement would make the name ambiguous
        assert db.reads() == [] and db.windowed() == []
        assert len(db.exact()) == 1

    async def test_the_distance_strategy_picks_the_operator_class(self, make_store) -> None:
        db = FakeDatabase(create_statement("VECTOR INDEX old (embedding)"))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db, distance_strategy=DistanceStrategy.EUCLIDEAN)

        await store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)

        [sent] = db.windowed()
        assert "embedding <-> CAST(:query_vector AS VECTOR)" in sent.sql


class TestVectorIndexCache:
    async def test_read_once_per_interval(self, make_store, clock: FakeClock) -> None:
        assert VECTOR_INDEX_REFRESH_SECONDS == 60.0
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db)

        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 1
        clock.now += VECTOR_INDEX_REFRESH_SECONDS - 0.5
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 1
        clock.now += 0.5
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 2

    async def test_an_index_made_by_another_process_is_used_after_the_interval(
        self, make_store, clock: FakeClock
    ) -> None:
        db = FakeDatabase(create_statement())
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db)

        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        db.create = create_statement(COSINE_INDEX)
        clock.now += 30
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert db.windowed() == [] and len(db.exact()) == 2
        clock.now += 30
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.windowed()) == 1 and len(db.exact()) == 2

    async def test_applying_or_dropping_an_index_reads_again(
        self, make_store, clock: FakeClock
    ) -> None:
        db = FakeDatabase(create_statement())
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db)

        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert db.windowed() == []

        # aapply_vector_index reads the table for itself, then creates the index
        await store.aapply_vector_index(CSPANNIndex())
        assert len(db.reads()) == 2
        assert any(s.sql.startswith("CREATE VECTOR INDEX") for s in db.sent)
        db.create = create_statement(COSINE_INDEX)
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 3 and len(db.windowed()) == 1

        # Applying an index that exists reads again too
        await store.aapply_vector_index(CSPANNIndex())
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 5 and len(db.windowed()) == 2

        await store.adrop_vector_index()
        db.create = create_statement()
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 6 and len(db.windowed()) == 2

    async def test_a_failed_index_statement_reads_again(self, make_store, clock: FakeClock) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db)
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        original = db.execute

        async def failing_ddl(statement: Any, params: Any = None, **kwargs: Any) -> list[Any]:
            if kwargs.get("ddl"):
                raise SAOperationalError(statement.text, {}, server_error("40003"))
            return await original(statement, params, **kwargs)

        store._execute = failing_ddl  # type: ignore[method-assign]
        # An ambiguous result: the drop may have committed
        with pytest.raises(SAOperationalError):
            await store.adrop_vector_index()
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.reads()) == 2

    async def test_a_read_that_started_before_a_reset_is_not_kept(
        self, make_store, clock: FakeClock
    ) -> None:
        db = FakeDatabase(create_statement())
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        store = make_store(db)
        # The index is applied while the first read is on its way back
        db.on_read = store._forget_vector_index

        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        db.on_read = None
        db.create = create_statement(COSINE_INDEX)
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)

        assert len(db.reads()) == 2
        assert len(db.windowed()) == 1

    async def test_a_read_error_the_policy_does_not_retry_logs_once_at_info(
        self, make_store, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(4, window_size=9, scope_rows=9)
        db.exact_rows = [hit(0)]
        db.read_error = SAProgrammingError(
            "SHOW CREATE TABLE", {}, server_error("42501", "user has no privileges on docs")
        )
        store = make_store(db)

        with caplog.at_level(logging.INFO, logger="langchain_cockroachdb.async_vectorstore"):
            results = await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
            assert [doc.id for doc in results] == ["id0"]
            await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
            assert len(db.reads()) == 1
            clock.now += VECTOR_INDEX_REFRESH_SECONDS
            await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
            assert len(db.reads()) == 2

        records = [r for r in caplog.records if r.name == "langchain_cockroachdb.async_vectorstore"]
        assert [r.levelno for r in records] == [logging.INFO]
        message = records[0].getMessage()
        assert "public.docs" in message and "42501" in message
        # The log carries the driver's message, not SQLAlchemy's dump of the
        # statement and its parameters
        assert "user has no privileges on docs" in message
        assert "[SQL:" not in message
        assert db.windowed() == [] and len(db.exact()) == 3

        # Once the read works again, the window is used
        db.read_error = None
        clock.now += VECTOR_INDEX_REFRESH_SECONDS
        await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert len(db.windowed()) == 1

    async def test_a_retryable_read_error_is_raised(self, make_store) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.read_error = SAOperationalError("SHOW CREATE TABLE", {}, server_error("40001"))
        store = make_store(db)

        # The policy gave up on it, so the search fails as its own statement would
        with pytest.raises(SAOperationalError):
            await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert db.exact() == [] and db.windowed() == []

    async def test_an_error_from_outside_the_database_is_raised(self, make_store) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.read_error = RuntimeError("Event loop is closed")
        store = make_store(db)

        with pytest.raises(RuntimeError):
            await store.asimilarity_search_by_vector(QUERY, k=4, filter=FILTER)
        assert db.exact() == []


class TestSyncParity:
    async def test_sync_searches_send_what_async_searches_send(self, make_store) -> None:
        rows = [hit(i, "[0.1,0.2,0.3]", 9, 9) for i in range(4)]
        dbs = [FakeDatabase(create_statement(COSINE_INDEX)) for _ in range(2)]
        for db in dbs:
            db.window_rows = rows
        async_store = make_store(dbs[0], namespace="tenant-a")
        sync_store = make_store(dbs[1], cls=CockroachDBVectorStore, namespace="tenant-a")
        dbs[0].create = dbs[1].create = create_statement(NAMESPACE_INDEX)

        mine = await async_store.asimilarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)
        theirs = sync_store.similarity_search_with_score_by_vector(QUERY, k=4, filter=FILTER)
        assert mine == theirs
        mmr = await async_store.amax_marginal_relevance_search_by_vector(
            QUERY, k=2, fetch_k=4, filter=FILTER
        )
        sync_mmr = sync_store.max_marginal_relevance_search_by_vector(
            QUERY, k=2, fetch_k=4, filter=FILTER
        )
        assert mmr == sync_mmr

        assert [(s.sql, s.params, s.label) for s in dbs[0].sent] == [
            (s.sql, s.params, s.label) for s in dbs[1].sent
        ]
        assert len(dbs[1].windowed()) == 2 and dbs[1].exact() == []


class TestHybridVectorLeg:
    async def test_the_candidate_pool_is_the_vector_legs_k(self, make_store) -> None:
        db = FakeDatabase(create_statement(COSINE_INDEX))
        db.window_rows = window_rows(3, window_size=3, scope_rows=3)
        store = make_store(db, hybrid_search_config=HybridSearchConfig())

        results = await store.asimilarity_search("query", k=2, filter=FILTER)

        assert len(results) == 2
        [sent] = db.windowed()
        pool = max(2 * 4, 20)
        assert (sent.params["k"], sent.params["window"]) == (pool, _WINDOW_FACTOR * pool + 1)
        assert db.exact() == []


def test_the_refresh_interval_is_exported() -> None:
    assert langchain_cockroachdb.VECTOR_INDEX_REFRESH_SECONDS == 60.0
    assert "VECTOR_INDEX_REFRESH_SECONDS" in langchain_cockroachdb.__all__
