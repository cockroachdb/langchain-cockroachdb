# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - Unreleased

### Added
- `RetryPolicy`: one configurable object for which errors are retried and how, set on the engine and inherited by the vector store and chat history; checkpointers take one too. Exported defaults (`RETRYABLE_SQLSTATES`, `TRANSIENT_MESSAGES`, ...) and `get_sqlstate()`.
- Retries for 40001, 40003, every 08xxx code, 57P01, 58C00, 58C01 and 53300, plus connection loss reported without a code. Failed logins and SSL or configuration errors are never retried.
- Optional hang guard (`attempt_timeout`, `ddl_attempt_timeout`) that abandons and retries an attempt the server never answers.
- `aadd_embeddings`/`add_embeddings`, `amax_marginal_relevance_search_by_vector`, the full langchain-postgres filter grammar (`$between`, `$exists`, `$like`, `$ilike`, `$not`).
- `CockroachDBChatMessageHistory` accepts a shared `CockroachDBEngine` and has a public `acreate_table_if_not_exists()`.
- Checkpointer `from_conn_string` options: `retry_policy`, `connect_timeout`, `prepare_threshold`, `connect_kwargs`.
- `connect_timeout` on `CockroachDBEngine.from_connection_string` (default 10 seconds).
- `examples/resilience.py`, end-to-end tests for every example, LangGraph checkpointer conformance tests, and integration tests that inject real CockroachDB faults.

### Fixed
- Errors were classified by psycopg2's `pgcode`, which psycopg 3 does not set, and by words like "connection" and "timeout" in any exception. Missing tables and statement timeouts were retried.
- The vector store never applied its retry settings; chat history and both checkpointers had no retries.
- Sync methods failed under concurrency ("bound to a different event loop") because every call ran its own event loop against a shared pool. They now share one background loop.
- Metadata filters broke on apostrophes, could carry injected SQL, and failed on boolean and null values. All values are bound parameters now.
- Vector indexes ignored the distance strategy and were always `vector_l2_ops`, so cosine and inner product searches never used them.
- `vector_search_beam_size` stayed set on pooled connections after a search.
- Chat history `add_messages` could leave partial writes and could duplicate messages on retry.
- MMR re-embedded every candidate through the embeddings API.
- `from_texts` failed with `ids` or vector store arguments.
- Concurrent checkpointer `setup()` could fail on a duplicate migration version, and calling the saver while iterating `list()` deadlocked.
- Running setup a second time failed with SQLSTATE 57000 when `autocommit_before_ddl` was off and the table was schema_locked, which is the default for new tables on v26. This hit `init_vectorstore_table` with `namespace_column` or `create_tsvector`, and chat history `create_table_if_not_exists`. Schema setup now runs one statement per implicit transaction.
- In pipeline mode, checkpointer `setup()` and `enable_ttl()`/`disable_ttl()` failed with `autocommit_before_ddl` off. That covered a fresh database and an upgrade that adds migrations. Every schema statement now runs in its own implicit transaction.
- After a session is canceled (`CANCEL SESSION`, or a connection dropped by a load balancer or drain), CockroachDB can answer the next query with SQLSTATE XXUUU "read tcp ...: i/o timeout". v25.4 always does; v26.3 sometimes does. The checkpointers raised it. It is now retried on a fresh connection.
- In a vector store with a namespace, adding a document whose id already belonged to another namespace overwrote that row and moved it into the writing namespace. Such ids now raise `ValueError` before anything is written, and the upsert never updates a row in another namespace.

### Changed
- psycopg 3 only: `psycopg2-binary` is no longer a dependency. Requires `langchain-core` 1.x and caps every dependency at its next major version. Python 3.10 to 3.14.
- Supports CockroachDB v25.4 or later; tested on v25.4 LTS and v26.3.
- The vector store's retry settings default to the engine's (5 attempts) instead of 3.
- `max_retries < 1` raises `ValueError` instead of never calling the function.
- Table, schema, column and index names must be plain SQL identifiers.
- Indexes created by 0.3.x for COSINE or INNER_PRODUCT should be dropped and applied again.
- `add_texts`/`add_embeddings` reject duplicate ids within one call (`ValueError`), and metadata that is not JSON serializable, including NaN and infinity (`TypeError`). Both checks run before anything is written. They used to fail partway, after earlier batches had committed.

## [0.3.0] - 2026-07-15

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
- `fts_rank_normalization` option on `HybridSearchConfig` exposing the ts_rank
  normalization bitmask, for example 1 or 32 to damp the rank bias toward
  long documents.

### Changed
- Default fusion method is now reciprocal rank fusion instead of weighted sum.
  RRF fuses by rank, so it does not depend on score scales and is the safer
  general purpose choice. Pass `fusion_type="weighted_sum"` to keep the old
  behavior. Since hybrid search never executed before this release, no
  working setup is affected.
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
