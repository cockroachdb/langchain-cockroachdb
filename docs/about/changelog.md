# Changelog

All notable changes to langchain-cockroachdb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-03-24

### Changed
- Checkpointer query performance optimization: replaced correlated subqueries
  with separate lightweight queries and raw BYTEA deserialization.
- Batch fetching in `list()`: 3 total queries instead of 2N+1.
- Prepared statement caching (`prepare_threshold=5`) for query plan reuse.

### Added
- CockroachDB row-level TTL: `enable_ttl()` / `disable_ttl()` (sync and async)
- Checkpointer performance benchmark script

## [0.2.0] - 2026-02-19

### Added
- LangGraph checkpointer: `CockroachDBSaver` and `AsyncCockroachDBSaver` for
  persisting LangGraph workflow state (short-term memory, human-in-the-loop,
  fault tolerance)
- Multi-tenancy: opt-in namespace column on vectorstore for tenant isolation
- Vectorstore standard tests compliance (25/25 LangChain standard tests passing)
- Checkpointer guide, multi-tenancy guide, API reference, runnable examples

### Changed
- Clarified isolation level support: works with both SERIALIZABLE (default,
  recommended) and READ COMMITTED
- Test suite expanded from 92 to 177 tests

## [0.1.0] - 2024-01-31

### Added
- Initial release of langchain-cockroachdb
- `CockroachDBEngine` for connection management
- `AsyncCockroachDBVectorStore` for async vector operations
- `CockroachDBVectorStore` sync wrapper
- `CSPANNIndex` for C-SPANN vector indexes
- `HybridSearchConfig` for FTS + vector search
- `CockroachDBChatMessageHistory` for chat persistence
- Automatic retry logic with exponential backoff
- Comprehensive test suite (92 tests)
- Full documentation with mkdocs
- 7 working examples
- Support for Python 3.10-3.12

### Features
- Native CockroachDB VECTOR type support
- C-SPANN distributed vector indexes
- Advanced metadata filtering
- Connection pooling with health checks
- Configurable retry parameters
- Both async and sync APIs
- Multi-tenant index support

[Unreleased]: https://github.com/cockroachdb/langchain-cockroachdb/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/cockroachdb/langchain-cockroachdb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cockroachdb/langchain-cockroachdb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cockroachdb/langchain-cockroachdb/releases/tag/v0.1.0
