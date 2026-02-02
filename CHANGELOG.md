# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

## [0.1.0] - 2026-02-01

Initial alpha release.
