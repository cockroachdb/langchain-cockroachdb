"""LangChain standard vectorstore integration tests for CockroachDB.

These tests verify that CockroachDBVectorStore conforms to the
LangChain VectorStore interface contract.
"""

import asyncio
import uuid
from collections.abc import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_cockroachdb import CockroachDBEngine, CockroachDBVectorStore


class TestCockroachDBStandardVectorStore(VectorStoreIntegrationTests):
    """Standard LangChain VectorStore integration tests for CockroachDB."""

    @pytest.fixture()
    def vectorstore(self, connection_string: str) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for standard tests."""
        table_name = f"test_standard_{uuid.uuid4().hex[:8]}"

        async def _setup():
            engine = CockroachDBEngine.from_connection_string(connection_string)
            await engine.ainit_vectorstore_table(
                table_name=table_name,
                vector_dimension=6,
                id_type="TEXT",
                drop_if_exists=True,
            )
            return engine

        engine = asyncio.run(_setup())

        store = CockroachDBVectorStore(
            engine=engine,
            embeddings=self.get_embeddings(),
            collection_name=table_name,
        )

        try:
            yield store
        finally:
            asyncio.run(engine.adrop_table(table_name))
            asyncio.run(engine.aclose())
