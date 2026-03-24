# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-03-24

### Changed
- Checkpointer query performance optimization: replaced correlated subqueries
  with separate lightweight queries and Python-side aggregation. Eliminates
  `jsonb_agg`/`jsonb_build_object`/`encode(blob, 'hex')` overhead in SQL.
- Blob deserialization now uses raw BYTEA via binary cursor instead of
  hex-encoding in SQL and decoding in Python.
- `list()` now batch-fetches blobs and writes for all checkpoints in 2 queries
  (instead of 2 queries per checkpoint), reducing round trips from 2N+1 to 3.
- Enabled prepared statement caching (`prepare_threshold=5`) for both sync and
  async savers, allowing query plan reuse after 5 executions.

### Added
- CockroachDB row-level TTL support via `enable_ttl()` / `disable_ttl()`
  (sync) and `aenable_ttl()` / `adisable_ttl()` (async) methods. Uses
  `ttl_expiration_expression` (recommended) to avoid full table rewrites.
  Adds `created_at` column to all checkpoint tables via migration.
- Checkpointer performance benchmark script (`tests/performance/bench_checkpointer.py`)

## [0.2.0] - 2026-02-19

### Added
- LangGraph checkpointer: `CockroachDBSaver` and `AsyncCockroachDBSaver` for
  persisting LangGraph workflow state (short-term memory, human-in-the-loop,
  fault tolerance). Uses JSONB aggregation instead of multidimensional arrays
  for CockroachDB compatibility.
- Multi-tenancy: opt-in namespace column on vectorstore for tenant isolation.
  All CRUD and search operations are scoped when `namespace` is set.
- Vectorstore standard tests compliance (25/25 LangChain standard tests passing)
- Documentation: checkpointer guide, multi-tenancy guide, API reference,
  runnable examples for both features
- Updated quickstart example to demonstrate all vectorstore feature table columns
  (delete by ID, search by vector, search with score, IDs in add documents)

### Changed
- Clarified isolation level support: works with both SERIALIZABLE (default,
  recommended) and READ COMMITTED
- Added LangChain official integration doc links to README and docsite
- Test suite expanded from 92 to 177 tests

## [0.1.0] - 2026-02-01

### Added
- Initial implementation of langchain-cockroachdb
- CockroachDBEngine for connection management
- AsyncCockroachDBVectorStore and CockroachDBVectorStore for vector operations
- C-SPANN vector index support with configurable partitions
- Multiple distance strategies (cosine, L2, inner product)
- Metadata filtering with complex operators ($and, $or, $gt, $lt, $in, etc.)
- Hybrid search combining FTS and vector similarity
- Chat message history persistence
- Comprehensive unit and integration tests
- Development and contributing guidelines
