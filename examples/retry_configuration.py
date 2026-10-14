"""Retry and pool configuration presets for langchain-cockroachdb.

Every database call retries transient CockroachDB errors (40001, 40003, 08xxx,
57P01, 58C00, 58C01, 53300) under a RetryPolicy. Set it once on the engine;
the vector store and chat history inherit it, and each can override it.
"""

import asyncio
import os

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cockroachdb import (
    RETRYABLE_SQLSTATES,
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    RetryPolicy,
)

CONNECTION_STRING = os.getenv(
    "COCKROACHDB_URL",
    "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable",
)

PRESETS = {
    # Defaults: 5 attempts, 0.1s first backoff, doubling, capped at 10s, 50% jitter
    "default": RetryPolicy(),
    # Interactive web app: fail fast, stay responsive
    "web app": RetryPolicy(max_attempts=3, max_backoff=1.0),
    # Batch job: patient, rides out longer blips
    "batch job": RetryPolicy(max_attempts=15, initial_backoff=0.5, max_backoff=30.0),
    # Multi-region: also retry statement timeouts raised while leases move,
    # and cut off attempts that hang (for example a COMMIT during quorum loss)
    "multi-region failover": RetryPolicy(
        max_attempts=8,
        max_backoff=5.0,
        retryable_sqlstates=RETRYABLE_SQLSTATES | {"57014"},
        attempt_timeout=10,
    ),
}


async def show_preset(name: str, policy: RetryPolicy) -> None:
    print(f"{name}")
    print(
        f"   attempts={policy.max_attempts} backoff={policy.initial_backoff}s..{policy.max_backoff}s "
        f"jitter={policy.jitter} attempt_timeout={policy.attempt_timeout}"
    )
    engine = CockroachDBEngine.from_connection_string(CONNECTION_STRING, retry_policy=policy)
    table = "retry_config_" + name.replace(" ", "_").replace("-", "_")
    await engine.ainit_vectorstore_table(table, 384, drop_if_exists=True)
    store = AsyncCockroachDBVectorStore(
        engine, DeterministicFakeEmbedding(size=384), table, batch_size=50
    )
    ids = await store.aadd_texts([f"{name} doc {i}" for i in range(100)])
    print(f"   added {len(ids)} documents in batches of 50\n")
    await engine.adrop_table(table)
    await engine.aclose()


async def show_overrides() -> None:
    print("Per-component overrides")
    engine = CockroachDBEngine.from_connection_string(
        CONNECTION_STRING,
        pool_size=20,
        max_overflow=40,
        connect_timeout=5,  # seconds; a URL or connect_args value wins
        retry_policy=PRESETS["web app"],
    )
    store = AsyncCockroachDBVectorStore(
        engine, DeterministicFakeEmbedding(size=384), "unused", retry_max_attempts=6
    )
    print(
        f"   engine attempts={engine.retry_policy.max_attempts}, store attempts={store.retry_max_attempts}"
    )
    # Policies are plain data, so they can come from YAML or environment variables
    from_config = RetryPolicy(**{"max_attempts": 4, "retryable_sqlstates": ["40001", "40003"]})
    print(f"   policy from config retries {sorted(from_config.retryable_sqlstates)}\n")
    await engine.aclose()


async def main() -> None:
    print("\nlangchain-cockroachdb retry configuration\n")
    for name, policy in PRESETS.items():
        await show_preset(name, policy)
    await show_overrides()


if __name__ == "__main__":
    asyncio.run(main())
