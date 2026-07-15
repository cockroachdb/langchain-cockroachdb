# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Hybrid search now actually executes. Setting `hybrid_search_config` on the
  vectorstore previously had no effect: searches ran pure vector similarity and
  the FTS half never fired (#7). `asimilarity_search_with_score` now runs the
  vector search and a `ts_rank` full-text search in parallel and fuses the
  results with the configured fusion method. Uses `plainto_tsquery` since
  CockroachDB does not support `websearch_to_tsquery`. The query text is passed
  as a bound parameter.

### Added
- `fts_language` parameter on `ainit_vectorstore_table` so the generated
  tsvector column can use a text search configuration other than english. It
  should match `HybridSearchConfig.fts_query_language`.
- `fetch_k` keyword argument on hybrid searches to control the candidate pool
  size fetched from each leg before fusion (default: `max(k * 4, 20)`).
- Score normalization for weighted sum fusion: vector distances and ts_rank
  scores are min-max normalized to [0, 1] before weighting, so the two scales
  are comparable regardless of distance strategy.

### Changed
- `HybridSearchConfig` now validates `fts_query_language` against a strict
  identifier pattern since the value is used in SQL.

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
