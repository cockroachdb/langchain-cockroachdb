"""Retry and configuration example for langchain-cockroachdb.

This example demonstrates how to configure retry logic and connection
pooling for different workload patterns and production requirements.
"""

import asyncio
import os

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

# Replace with your connection string
CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)


async def example_default_config() -> None:
    """Example with default configuration (suitable for development)."""
    print("1. DEFAULT CONFIGURATION (Development)")
    print("=" * 60)

    # Default settings:
    # - pool_size: 10
    # - max_overflow: 20
    # - retry_max_attempts: 5
    # - retry_initial_backoff: 0.1s
    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)

    print("   Pool size: 10 (default)")
    print("   Max overflow: 20 (default)")
    print(f"   Retry max attempts: {engine.retry_max_attempts}")
    print(f"   Retry initial backoff: {engine.retry_initial_backoff}s")

    await engine.aclose()
    print()


async def example_high_performance_config() -> None:
    """Example with high-performance configuration (production web app)."""
    print("2. HIGH PERFORMANCE CONFIGURATION (Production Web App)")
    print("=" * 60)

    # Optimized for:
    # - High concurrent connections
    # - Fast failover
    # - Distributed CockroachDB cluster
    engine = CockroachDBEngine.from_connection_string(
        CONNECTION_STRING,
        # Connection pool settings
        pool_size=20,  # More base connections
        max_overflow=40,  # Higher burst capacity
        pool_pre_ping=True,  # Health check before use
        pool_recycle=1800,  # Recycle after 30 min
        pool_timeout=30.0,  # Wait up to 30s for connection
        # Retry settings
        retry_max_attempts=10,  # More aggressive retries
        retry_initial_backoff=0.05,  # Start with quick retry
        retry_max_backoff=30.0,  # Longer max backoff
        retry_backoff_multiplier=2.0,
        retry_jitter=True,  # Prevent thundering herd
    )

    print("   Pool size: 20 (high concurrency)")
    print("   Max overflow: 40 (burst capacity)")
    print("   Pool recycle: 1800s (30 min)")
    print(f"   Retry max attempts: {engine.retry_max_attempts}")
    print(f"   Retry max backoff: {engine.retry_max_backoff}s")

    await engine.ainit_vectorstore_table(
        table_name="high_perf_test",
        vector_dimension=384,
        drop_if_exists=True,
    )

    embeddings = DeterministicFakeEmbedding(size=384)
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="high_perf_test",
        batch_size=100,  # Larger batches
        retry_max_attempts=5,  # Vectorstore retry config
    )

    # Simulate concurrent operations
    texts = [f"High performance doc {i}" for i in range(50)]
    ids = await vectorstore.aadd_texts(texts)
    print(f"   Added {len(ids)} documents successfully")

    await engine.aclose()
    print()


async def example_low_latency_config() -> None:
    """Example with low-latency configuration (single-region, fail-fast)."""
    print("3. LOW LATENCY CONFIGURATION (Single Region, Fail Fast)")
    print("=" * 60)

    # Optimized for:
    # - Low latency operations
    # - Single-region deployment
    # - Fail fast on errors
    engine = CockroachDBEngine.from_connection_string(
        CONNECTION_STRING,
        # Smaller pool for low-latency
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_timeout=5.0,  # Fail fast
        # Minimal retries
        retry_max_attempts=2,  # Fail faster
        retry_initial_backoff=0.05,
        retry_max_backoff=1.0,  # Short max backoff
        retry_jitter=False,  # Deterministic timing
    )

    print("   Pool size: 5 (low latency)")
    print("   Pool timeout: 5.0s (fail fast)")
    print(f"   Retry max attempts: {engine.retry_max_attempts}")
    print(f"   Retry jitter: {engine.retry_jitter}")

    await engine.aclose()
    print()


async def example_batch_job_config() -> None:
    """Example with batch job configuration (long-running, resilient)."""
    print("4. BATCH JOB CONFIGURATION (Long Running, Resilient)")
    print("=" * 60)

    # Optimized for:
    # - Long-running batch operations
    # - Maximum resilience to transient errors
    # - Not latency-sensitive
    engine = CockroachDBEngine.from_connection_string(
        CONNECTION_STRING,
        # Smaller pool (sequential processing)
        pool_size=3,
        max_overflow=5,
        pool_recycle=3600,  # Recycle after 1 hour
        # Very aggressive retries
        retry_max_attempts=20,  # Don't give up easily
        retry_initial_backoff=0.5,  # Start conservative
        retry_max_backoff=60.0,  # Long max backoff
        retry_backoff_multiplier=2.0,
        retry_jitter=True,
    )

    print("   Pool size: 3 (sequential)")
    print(f"   Retry max attempts: {engine.retry_max_attempts}")
    print(f"   Retry max backoff: {engine.retry_max_backoff}s")

    await engine.ainit_vectorstore_table(
        table_name="batch_test",
        vector_dimension=384,
        drop_if_exists=True,
    )

    embeddings = DeterministicFakeEmbedding(size=384)
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="batch_test",
        batch_size=50,  # Moderate batch size
        retry_max_attempts=10,  # Aggressive retries per batch
    )

    # Simulate batch processing
    texts = [f"Batch job doc {i}" for i in range(100)]
    ids = await vectorstore.aadd_texts(texts)
    print(f"   Processed {len(ids)} documents in batch")

    await engine.aclose()
    print()


async def example_multi_region_config() -> None:
    """Example with multi-region configuration (high latency, resilient)."""
    print("5. MULTI-REGION CONFIGURATION (High Latency Tolerance)")
    print("=" * 60)

    # Optimized for:
    # - Multi-region CockroachDB deployment
    # - High network latency (50-200ms)
    # - Resilient to regional failures
    engine = CockroachDBEngine.from_connection_string(
        CONNECTION_STRING,
        pool_size=15,
        max_overflow=30,
        pool_pre_ping=True,
        pool_timeout=60.0,  # Tolerate high latency
        # Tolerant retry settings
        retry_max_attempts=15,
        retry_initial_backoff=0.2,  # Account for latency
        retry_max_backoff=60.0,  # Long backoff for regional issues
        retry_backoff_multiplier=2.0,
        retry_jitter=True,
    )

    print("   Pool timeout: 60.0s (high latency tolerance)")
    print(f"   Retry max attempts: {engine.retry_max_attempts}")
    print(f"   Retry initial backoff: {engine.retry_initial_backoff}s")

    await engine.aclose()
    print()


async def example_configuration_override() -> None:
    """Example showing runtime configuration override."""
    print("6. RUNTIME CONFIGURATION OVERRIDE")
    print("=" * 60)

    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING)

    await engine.ainit_vectorstore_table(
        table_name="override_test",
        vector_dimension=384,
        drop_if_exists=True,
    )

    embeddings = DeterministicFakeEmbedding(size=384)

    # Default batch size
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="override_test",
        batch_size=100,  # Default
    )

    print(f"   Default batch size: {vectorstore.batch_size}")

    # Override batch size at runtime
    texts = [f"Doc {i}" for i in range(20)]
    ids = await vectorstore.aadd_texts(texts, batch_size=5)  # Override
    print(f"   Added {len(ids)} documents with batch_size=5 (override)")

    await engine.aclose()
    print()


async def main() -> None:
    """Run all configuration examples."""
    print("\n🪳 LangChain CockroachDB - Retry & Configuration Examples\n")

    await example_default_config()
    await example_high_performance_config()
    await example_low_latency_config()
    await example_batch_job_config()
    await example_multi_region_config()
    await example_configuration_override()

    print("=" * 60)
    print("CONFIGURATION GUIDELINES:")
    print("=" * 60)
    print("📊 Pool Size:")
    print("   - Development: 5-10")
    print("   - Production web app: 20-50")
    print("   - Batch jobs: 3-5")
    print()
    print("🔄 Retry Attempts:")
    print("   - Fail fast (dev): 2-3")
    print("   - Standard (prod): 5-10")
    print("   - Resilient (batch): 15-20")
    print()
    print("⏱️  Backoff Settings:")
    print("   - Low latency: initial=0.05s, max=1.0s")
    print("   - Standard: initial=0.1s, max=10.0s")
    print("   - High latency: initial=0.2s, max=60.0s")
    print()
    print("📦 Batch Size:")
    print("   - Small docs: 100-500")
    print("   - Large embeddings: 50-100")
    print("   - High contention: 10-50")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
