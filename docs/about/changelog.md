# Changelog

All notable changes to langchain-cockroachdb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/cockroachdb/langchain-cockroachdb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cockroachdb/langchain-cockroachdb/releases/tag/v0.1.0
