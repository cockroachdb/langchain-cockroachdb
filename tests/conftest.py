"""Pytest configuration and shared fixtures."""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Optional

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from testcontainers.core.container import DockerContainer

from langchain_cockroachdb.engine import CockroachDBEngine


class CockroachDBContainer(DockerContainer):
    """CockroachDB testcontainer."""

    def __init__(self, image: str = "cockroachdb/cockroach:latest"):
        super().__init__(image)
        self.with_command("start-single-node --insecure")
        self.with_exposed_ports(26257)

        # Use modern wait strategy with longer timeout (future-proof API)
        try:
            from testcontainers.core.waiting_utils import LogMessageWaitStrategy

            self.waiting_for(LogMessageWaitStrategy("nodeID").with_timeout(60))
        except (ImportError, AttributeError):
            # Fallback for older testcontainers versions
            pass

    def get_connection_url(self) -> str:
        """Get connection URL."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(26257)
        return f"cockroachdb+psycopg://root@{host}:{port}/defaultdb?sslmode=disable"

    def start(self) -> "CockroachDBContainer":
        """Start container and wait for readiness."""
        import time

        super().start()

        # Give CockroachDB extra time to fully initialize after log message appears
        time.sleep(5)

        return self


@pytest.fixture(scope="session")
def cockroachdb_container() -> Generator[Optional[CockroachDBContainer], None, None]:
    """Start CockroachDB container for test session."""
    use_testcontainer = os.getenv("USE_TESTCONTAINER", "true").lower() == "true"

    if not use_testcontainer:
        # When testcontainer is disabled, return None
        # connection_string fixture will use COCKROACHDB_URL env var instead
        yield None
        return

    container = CockroachDBContainer()
    container.start()

    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def connection_string(cockroachdb_container: Optional[CockroachDBContainer]) -> str:
    """Get connection string for tests.

    Uses testcontainer if USE_TESTCONTAINER=true (default).
    Uses COCKROACHDB_URL environment variable if USE_TESTCONTAINER=false.
    """
    if cockroachdb_container is not None:
        # Using testcontainer
        return cockroachdb_container.get_connection_url()

    # Using external CockroachDB instance
    connection_string_env = os.getenv("COCKROACHDB_URL")
    if not connection_string_env:
        pytest.skip(
            "No CockroachDB connection available. "
            "Either enable USE_TESTCONTAINER=true or set COCKROACHDB_URL"
        )

    return connection_string_env


@pytest_asyncio.fixture
async def async_engine(connection_string: str) -> AsyncGenerator[AsyncEngine, None]:
    """Create async engine for tests."""
    # Ensure we use the async driver (psycopg, not psycopg2)
    # This is needed when using external CockroachDB with cockroachdb:// URLs
    if connection_string.startswith("cockroachdb://"):
        connection_string = connection_string.replace("cockroachdb://", "cockroachdb+psycopg://", 1)

    engine = create_async_engine(connection_string, pool_pre_ping=True)

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def cockroachdb_engine(
    async_engine: AsyncEngine,
) -> AsyncGenerator[CockroachDBEngine, None]:
    """Create CockroachDBEngine for tests."""
    engine = CockroachDBEngine.from_engine(async_engine)

    try:
        yield engine
    finally:
        await engine.aclose()


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return [
        "CockroachDB is a distributed SQL database",
        "LangChain is a framework for building LLM applications",
        "Vector databases store embeddings for similarity search",
        "Python is a popular programming language",
        "SQL queries can be used to retrieve data",
    ]


@pytest.fixture
def sample_metadatas() -> list[dict]:
    """Sample metadata for testing."""
    return [
        {"source": "docs", "page": 1, "category": "database"},
        {"source": "docs", "page": 2, "category": "framework"},
        {"source": "docs", "page": 3, "category": "database"},
        {"source": "docs", "page": 4, "category": "language"},
        {"source": "docs", "page": 5, "category": "database"},
    ]
