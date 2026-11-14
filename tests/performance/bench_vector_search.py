"""Reproducible vector retrieval benchmark against a separate exact baseline.

Run with COCKROACHDB_URL pointing to a disposable database. The script creates
and drops two uniquely named tables; it does not modify existing tables.
"""

import argparse
import asyncio
import json
import os
import platform
import time
import uuid
from collections import Counter
from dataclasses import asdict

import numpy as np
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_cockroachdb import (
    AsyncCockroachDBVectorStore,
    CockroachDBEngine,
    CSPANNIndex,
    CSPANNQueryOptions,
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


async def measure_searches(stores, queries, k, selectivity, beam) -> dict:
    """Compare identical queries and filters after warming both search paths."""
    predicate = {"bucket": {"$lt": selectivity}}
    events = []
    options = CSPANNQueryOptions(beam_size=beam, on_search=events.append)
    # Warm both paths once; exclude these observations from the report.
    for store in stores:
        await store.asimilarity_search_by_vector(
            queries[0].tolist(), k=k, filter=predicate, query_options=options
        )
    events.clear()
    exact_ms, indexed_ms, recall = [], [], []
    for query in queries:
        started = time.perf_counter()
        exact = await stores[0].asimilarity_search_by_vector(query.tolist(), k=k, filter=predicate)
        exact_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        indexed = await stores[1].asimilarity_search_by_vector(
            query.tolist(), k=k, filter=predicate, query_options=options
        )
        indexed_ms.append((time.perf_counter() - started) * 1000)
        expected = {doc.id for doc in exact}
        recall.append(len(expected & {doc.id for doc in indexed}) / len(expected))
    return {
        "filter_percent": selectivity,
        "beam_size": beam,
        "recall_at_k": float(np.mean(recall)),
        "exact_p50_ms": float(np.percentile(exact_ms, 50)),
        "exact_p95_ms": float(np.percentile(exact_ms, 95)),
        "indexed_p50_ms": float(np.percentile(indexed_ms, 50)),
        "indexed_p95_ms": float(np.percentile(indexed_ms, 95)),
        "paths": dict(Counter(event.path for event in events)),
        "sample_diagnostics": asdict(events[-1]),
    }


async def benchmark(args: argparse.Namespace) -> dict:
    """Measure database retrieval only; synthetic vectors require no model API."""
    engine = CockroachDBEngine.from_connection_string(os.environ["COCKROACHDB_URL"])
    suffix = uuid.uuid4().hex[:12]
    tables = [f"bench_exact_{suffix}", f"bench_index_{suffix}"]
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.rows, args.dims))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(args.queries, args.dims))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, str(i))) for i in range(args.rows)]
    metadata = [{"bucket": i % 100} for i in range(args.rows)]
    stores = []
    created = []
    report = {"configuration": vars(args), "python": platform.python_version(), "runs": []}
    try:
        async with engine.engine.connect() as conn:
            report["database"] = (await conn.execute(text("SELECT version()"))).scalar_one()
        ingestion = []
        for table in tables:
            await engine.ainit_vectorstore_table(table, args.dims, namespace_column="namespace")
            created.append(table)
            store = AsyncCockroachDBVectorStore(
                engine, DeterministicFakeEmbedding(size=args.dims), table, namespace="benchmark"
            )
            stores.append(store)
            started = time.perf_counter()
            await store.aadd_embeddings(
                [f"document {i}" for i in range(args.rows)],
                vectors.tolist(),
                metadatas=metadata,
                ids=ids,
            )
            elapsed = time.perf_counter() - started
            ingestion.append({"rows_per_second": args.rows / elapsed, "seconds": elapsed})
        report["ingestion_before_index"] = ingestion
        started = time.perf_counter()
        await stores[1].aapply_vector_index(CSPANNIndex(), prefix_columns=["namespace"])
        report["index_build_seconds"] = time.perf_counter() - started
        # Populate statistics on both tables before comparing their search plans.
        for table in tables:
            await engine._run_ddl([f"ANALYZE {table}"], label="benchmark statistics")
        for selectivity in (100, 10, 1):
            for beam in args.beam:
                report["runs"].append(
                    await measure_searches(stores, queries, args.k, selectivity, beam)
                )
        return report
    finally:
        try:
            for table in reversed(created):
                await engine.adrop_table(table)
        finally:
            await engine.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, default=1000)
    parser.add_argument("--dims", type=positive_int, default=64)
    parser.add_argument("--queries", type=positive_int, default=20)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--beam", type=positive_int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(asyncio.run(benchmark(parser.parse_args())), indent=2))
