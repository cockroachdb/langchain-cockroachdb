"""Integration tests for vector store."""

import uuid

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.indexes import CSPANNIndex, CSPANNQueryOptions, DistanceStrategy


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        return [[float(i), float(i + 1), float(i + 2)] for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed query."""
        return [1.0, 2.0, 3.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed query."""
        return self.embed_query(text)


@pytest.mark.asyncio
class TestAsyncCockroachDBVectorStore:
    """Test async vector store with real database."""

    @pytest.fixture
    async def vectorstore(
        self, cockroachdb_engine: CockroachDBEngine
    ) -> AsyncCockroachDBVectorStore:
        """Create vector store for testing."""
        embeddings = FakeEmbeddings()
        collection_name = "test_collection"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=collection_name,
            vector_dimension=3,
            drop_if_exists=True,
        )

        return AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=collection_name,
        )

    async def test_aadd_texts(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test adding texts."""
        ids = await vectorstore.aadd_texts(sample_texts)

        assert len(ids) == len(sample_texts)
        assert all(isinstance(id, str) for id in ids)

    async def test_aadd_texts_with_metadata(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test adding texts with metadata."""
        ids = await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)

        assert len(ids) == len(sample_texts)

    async def test_aadd_texts_with_ids(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test adding texts with custom IDs."""
        import uuid

        custom_ids = [str(uuid.uuid4()) for _ in range(len(sample_texts))]
        ids = await vectorstore.aadd_texts(sample_texts, ids=custom_ids)

        assert ids == custom_ids

    async def test_asimilarity_search(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test similarity search."""
        await vectorstore.aadd_texts(sample_texts)

        results = await vectorstore.asimilarity_search("database", k=3)

        assert len(results) <= 3
        assert all(hasattr(doc, "page_content") for doc in results)

    async def test_asimilarity_search_with_score(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test similarity search with scores."""
        await vectorstore.aadd_texts(sample_texts)

        results = await vectorstore.asimilarity_search_with_score("database", k=3)

        assert len(results) <= 3
        assert all(isinstance(score, float) for _, score in results)

    async def test_asimilarity_search_with_filter(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test similarity search with metadata filter."""
        await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)

        filter_dict = {"category": {"$eq": "database"}}
        results = await vectorstore.asimilarity_search("query", k=5, filter=filter_dict)

        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("category") == "database"

    async def test_asimilarity_search_with_and_filter(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test similarity search with AND filter."""
        await vectorstore.aadd_texts(sample_texts, metadatas=sample_metadatas)

        filter_dict = {
            "$and": [
                {"category": {"$eq": "database"}},
                {"page": {"$gt": 2}},
            ]
        }
        results = await vectorstore.asimilarity_search("query", k=5, filter=filter_dict)

        for doc in results:
            assert doc.metadata.get("category") == "database"
            assert doc.metadata.get("page", 0) > 2

    async def test_adelete(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test deleting documents."""
        ids = await vectorstore.aadd_texts(sample_texts)

        delete_ids = ids[:2]
        result = await vectorstore.adelete(delete_ids)

        assert result is True

        remaining = await vectorstore.asimilarity_search("", k=10)
        assert len(remaining) == len(sample_texts) - len(delete_ids)

    async def test_aapply_vector_index(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test creating vector index."""
        await vectorstore.aadd_texts(sample_texts)

        index = CSPANNIndex(
            distance_strategy=DistanceStrategy.COSINE,
            min_partition_size=10,
        )

        await vectorstore.aapply_vector_index(index)

    async def test_query_with_beam_size(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test query with beam size option."""
        await vectorstore.aadd_texts(sample_texts)

        index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
        await vectorstore.aapply_vector_index(index)

        query_options = CSPANNQueryOptions(beam_size=100)
        results = await vectorstore.asimilarity_search_with_score(
            "database",
            k=3,
            query_options=query_options,
        )

        assert len(results) <= 3

    async def test_amax_marginal_relevance_search(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
        sample_texts: list[str],
    ) -> None:
        """Test MMR search."""
        await vectorstore.aadd_texts(sample_texts)

        results = await vectorstore.amax_marginal_relevance_search(
            "database",
            k=3,
            fetch_k=5,
            lambda_mult=0.5,
        )

        assert len(results) <= 3

    async def test_batch_insert_with_custom_batch_size(
        self,
        vectorstore: AsyncCockroachDBVectorStore,
    ) -> None:
        """Test batch insert with custom batch size."""
        texts = [f"text_{i}" for i in range(250)]

        ids = await vectorstore.aadd_texts(texts, batch_size=50)

        assert len(ids) == 250

    async def test_afrom_texts(
        self,
        cockroachdb_engine: CockroachDBEngine,
        sample_texts: list[str],
    ) -> None:
        """Test creating vectorstore from texts."""
        embeddings = FakeEmbeddings()

        vectorstore = await AsyncCockroachDBVectorStore.afrom_texts(
            texts=sample_texts,
            embedding=embeddings,
            engine=cockroachdb_engine,
            collection_name="from_texts_test",
        )

        assert vectorstore is not None

        results = await vectorstore.asimilarity_search("database", k=3)
        assert len(results) <= 3

    async def test_multiple_distance_strategies(
        self,
        cockroachdb_engine: CockroachDBEngine,
        sample_texts: list[str],
    ) -> None:
        """Test different distance strategies."""
        embeddings = FakeEmbeddings()

        for strategy in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN,
            DistanceStrategy.INNER_PRODUCT,
        ]:
            collection_name = f"test_{strategy.value}"

            await cockroachdb_engine.ainit_vectorstore_table(
                table_name=collection_name,
                vector_dimension=3,
                drop_if_exists=True,
            )

            vectorstore = AsyncCockroachDBVectorStore(
                engine=cockroachdb_engine,
                embeddings=embeddings,
                collection_name=collection_name,
                distance_strategy=strategy,
            )

            await vectorstore.aadd_texts(sample_texts)

            results = await vectorstore.asimilarity_search("database", k=3)
            assert len(results) <= 3

    async def test_namespace_isolation(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that namespaces isolate documents from each other."""
        embeddings = FakeEmbeddings()
        table = "test_ns_isolation"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        await store_a.aadd_texts(["doc for tenant A"], ids=[id_a])
        await store_b.aadd_texts(["doc for tenant B"], ids=[id_b])

        results_a = await store_a.asimilarity_search("doc", k=10)
        results_b = await store_b.asimilarity_search("doc", k=10)

        assert len(results_a) == 1
        assert results_a[0].page_content == "doc for tenant A"
        assert len(results_b) == 1
        assert results_b[0].page_content == "doc for tenant B"

    async def test_namespace_delete_isolation(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that delete in one namespace doesn't affect another."""
        embeddings = FakeEmbeddings()
        table = "test_ns_delete"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        await store_a.aadd_texts(["doc A"], ids=[id_a])
        await store_b.aadd_texts(["doc B"], ids=[id_b])

        await store_a.adelete([id_a])

        results_a = await store_a.asimilarity_search("doc", k=10)
        results_b = await store_b.asimilarity_search("doc", k=10)

        assert len(results_a) == 0
        assert len(results_b) == 1

    async def test_namespace_get_by_ids_isolation(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that get_by_ids respects namespace."""
        embeddings = FakeEmbeddings()
        table = "test_ns_getbyids"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        await store_a.aadd_texts(["doc A"], ids=[id_a])
        await store_b.aadd_texts(["doc B"], ids=[id_b])

        # Tenant A can see id_a but not id_b
        assert len(await store_a.aget_by_ids([id_a])) == 1
        assert len(await store_a.aget_by_ids([id_b])) == 0

        # Tenant B can see id_b but not id_a
        assert len(await store_b.aget_by_ids([id_b])) == 1
        assert len(await store_b.aget_by_ids([id_a])) == 0

    async def test_no_namespace_sees_all(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that no namespace (None) sees all documents -- backward compatible."""
        embeddings = FakeEmbeddings()
        table = "test_ns_all"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )
        store_all = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            # no namespace -- sees everything
        )

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        await store_a.aadd_texts(["doc A"], ids=[id_a])
        await store_b.aadd_texts(["doc B"], ids=[id_b])

        results = await store_all.asimilarity_search("doc", k=10)
        assert len(results) == 2

    async def test_namespace_rejects_ids_owned_by_another_namespace(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that adding a document with an id owned by another namespace raises ValueError."""
        embeddings = FakeEmbeddings()
        table = "test_ns_reject_ids"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )

        # Tenant A adds a document
        id_x = str(uuid.uuid4())
        await store_a.aadd_texts(["A original"], ids=[id_x])

        # Tenant B tries to add documents with id_x and id_y
        id_y = str(uuid.uuid4())
        with pytest.raises(ValueError, match=str(id_x)):
            await store_b.aadd_texts(["B attempt", "B other"], ids=[id_x, id_y])

        # Verify A's document is unchanged
        docs_a = await store_a.aget_by_ids([id_x])
        assert len(docs_a) == 1
        assert docs_a[0].page_content == "A original"

        # Verify B has nothing (id_y should not be written)
        docs_b = await store_b.aget_by_ids([id_x, id_y])
        assert len(docs_b) == 0

    async def test_namespace_reupsert_in_same_namespace_updates(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that upserting in the same namespace updates the content."""
        embeddings = FakeEmbeddings()
        table = "test_ns_reupsert"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )

        id_x = str(uuid.uuid4())
        await store_a.aadd_texts(["v1"], ids=[id_x])
        await store_a.aadd_texts(["v2"], ids=[id_x])

        docs = await store_a.aget_by_ids([id_x])
        assert len(docs) == 1
        assert docs[0].page_content == "v2"

    async def test_guarded_upsert_never_crosses_namespaces(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that the guarded upsert rejects cross-namespace writes."""
        embeddings = FakeEmbeddings()
        table = "test_ns_guarded"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )

        # Tenant A adds a document
        id_x = str(uuid.uuid4())
        await store_a.aadd_texts(["A doc"], ids=[id_x])

        # Call store B's _insert_batch directly to bypass the check
        id_z = str(uuid.uuid4())
        emb_x = await embeddings.aembed_documents(["B attempt on X"])
        emb_z = await embeddings.aembed_documents(["B doc Z"])
        with pytest.raises(ValueError, match=str(id_x)):
            await store_b._insert_batch(
                ["B attempt on X", "B doc Z"],
                emb_x + emb_z,
                ["{}", "{}"],
                [id_x, id_z],
            )

        # Verify A's row is unchanged and still in namespace A
        docs_a = await store_a.aget_by_ids([id_x])
        assert len(docs_a) == 1
        assert docs_a[0].page_content == "A doc"

        # Verify Z is in namespace B
        docs_b = await store_b.aget_by_ids([id_z])
        assert len(docs_b) == 1
        assert docs_b[0].page_content == "B doc Z"

    async def test_uppercase_uuid_ids_are_not_reported_as_skipped(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that uppercase UUID ids work correctly without being reported as skipped."""
        embeddings = FakeEmbeddings()
        table = "test_ns_uppercase"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )

        # Add document with uppercase UUID
        id_upper = str(uuid.uuid4()).upper()
        await store_a.aadd_texts(["v1"], ids=[id_upper])

        # Upsert with new content (should not raise an error)
        await store_a.aadd_texts(["v2"], ids=[id_upper])

        # Verify content is updated
        docs = await store_a.aget_by_ids([id_upper])
        assert len(docs) == 1
        assert docs[0].page_content == "v2"

    async def test_store_without_namespace_keeps_upsert_behavior(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that stores without a namespace keep the same upsert behavior."""
        embeddings = FakeEmbeddings()
        table = "test_no_ns_upsert"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            # no namespace
        )

        id_x = str(uuid.uuid4())
        await store.aadd_texts(["v1"], ids=[id_x])
        await store.aadd_texts(["v2"], ids=[id_x])

        docs = await store.aget_by_ids([id_x])
        assert len(docs) == 1
        assert docs[0].page_content == "v2"

    async def test_namespace_int8_id_column_does_not_false_error(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Regression test: INT8 id column should not trigger false namespace error."""
        embeddings = FakeEmbeddings()
        table = "test_ns_int8_id"

        async with cockroachdb_engine._connect(
            begin=True, timeout=None, label="create int8 table"
        ) as conn:
            await conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS public.{table} ("
                    "id INT8 PRIMARY KEY, "
                    "content TEXT, "
                    "embedding VECTOR(3), "
                    "metadata JSONB DEFAULT '{}', "
                    "namespace TEXT NOT NULL DEFAULT '')"
                )
            )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )

        await store_a.aadd_texts(["doc"], ids=["007"])

        docs = await store_a.aget_by_ids(["007"])
        assert len(docs) == 1
        assert docs[0].page_content == "doc"

        async with cockroachdb_engine._connect(
            begin=True, timeout=None, label="drop int8 table"
        ) as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS public.{table}"))

    async def test_namespace_multi_chunk_check_rejects_before_any_write(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that multi-chunk pre-check rejects before any write."""
        embeddings = FakeEmbeddings()
        table = "test_ns_multi_chunk"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
            batch_size=1,
        )

        id_x = str(uuid.uuid4())
        await store_a.aadd_texts(["A doc"], ids=[id_x])

        id_y = str(uuid.uuid4())
        with pytest.raises(ValueError, match=str(id_x)):
            await store_b.aadd_texts(["B doc Y", "B doc X"], ids=[id_y, id_x])

        assert len(await store_b.aget_by_ids([id_y])) == 0
        assert len(await store_b.aget_by_ids([id_x])) == 0

    async def test_guarded_upsert_error_does_not_name_successful_rows(
        self,
        cockroachdb_engine: CockroachDBEngine,
    ) -> None:
        """Test that the guarded upsert error does not name successful rows."""
        embeddings = FakeEmbeddings()
        table = "test_ns_error_no_success"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        store_a = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-a",
        )
        store_b = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
            namespace="tenant-b",
        )

        id_x = str(uuid.uuid4())
        await store_a.aadd_texts(["A doc"], ids=[id_x])

        id_z = str(uuid.uuid4())
        emb_x = await embeddings.aembed_documents(["B attempt on X"])
        emb_z = await embeddings.aembed_documents(["B doc Z"])

        try:
            await store_b._insert_batch(
                ["B attempt on X", "B doc Z"],
                emb_x + emb_z,
                ["{}", "{}"],
                [id_x, id_z],
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert str(id_x) in str(e)
            assert str(id_z) not in str(e)

        docs_b = await store_b.aget_by_ids([id_z])
        assert len(docs_b) == 1
        assert docs_b[0].page_content == "B doc Z"

    async def test_store_without_namespace_sends_no_namespace_sql(
        self,
        cockroachdb_engine: CockroachDBEngine,
        unsafe_internals: bool,
    ) -> None:
        """Test that stores without a namespace send no namespace-specific SQL."""
        from tests.integration.faults import FaultInjector

        embeddings = FakeEmbeddings()
        table = "test_no_ns_sql"

        await cockroachdb_engine.ainit_vectorstore_table(
            table_name=table,
            vector_dimension=3,
            namespace_column="namespace",
            drop_if_exists=True,
        )

        injector = FaultInjector(cockroachdb_engine.engine, unsafe_internals=unsafe_internals)

        store = AsyncCockroachDBVectorStore(
            engine=cockroachdb_engine,
            embeddings=embeddings,
            collection_name=table,
        )

        id_x = str(uuid.uuid4())
        await store.aadd_texts(["v1"], ids=[id_x])
        await store.aadd_texts(["v2"], ids=[id_x])

        docs = await store.aget_by_ids([id_x])
        assert len(docs) == 1
        assert docs[0].page_content == "v2"

        assert injector.count("is distinct from") == 0
        assert injector.count("returning") == 0

        injector.close()

    async def test_amax_marginal_relevance_search_by_vector(
        self, vectorstore, sample_texts
    ) -> None:
        await vectorstore.aadd_texts(sample_texts)
        embedding = await vectorstore.embeddings.aembed_query("database")
        results = await vectorstore.amax_marginal_relevance_search_by_vector(
            embedding, k=2, fetch_k=5
        )
        assert len(results) == 2
