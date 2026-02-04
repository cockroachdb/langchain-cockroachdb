# langchain-cockroachdb

**LangChain integration for CockroachDB with native vector support**

[![Tests](https://github.com/cockroachdb/langchain-cockroachdb/actions/workflows/test.yml/badge.svg)](https://github.com/cockroachdb/langchain-cockroachdb/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/cockroachdb/langchain-cockroachdb/branch/main/graph/badge.svg)](https://codecov.io/gh/cockroachdb/langchain-cockroachdb)
[![PyPI version](https://badge.fury.io/py/langchain-cockroachdb.svg)](https://badge.fury.io/py/langchain-cockroachdb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Overview

This package provides LangChain abstractions backed by CockroachDB, leveraging CockroachDB's native `VECTOR` type and C-SPANN (CockroachDB SPANN) indexes for vector search at scale in a distributed, horizontally scalable database.

## Key Features

### 🎯 Vector Store
- **Native Vector Support**: Uses CockroachDB's native `VECTOR` type (not pgvector)
- **C-SPANN Indexes**: CockroachDB's vector index algorithm optimized for distributed systems
- **Advanced Filtering**: Rich metadata filtering with `$and/$or/$gt/$in` operators
- **Hybrid Search**: Combine full-text search (TSVECTOR) with vector similarity
- **Query-Time Tuning**: Adjust beam size for accuracy/speed tradeoff

### 🏗️ Reliability Features
- **Automatic Retry Logic**: Handles 40001 serialization errors with exponential backoff
- **SERIALIZABLE Isolation**: Built for CockroachDB's default isolation level
- **Multi-Tenancy**: Index prefix columns for efficient tenant isolation
- **Connection Pooling**: Configurable connection pools with health checks
- **Horizontal Scalability**: Designed for distributed deployments

### 💬 Chat History
- **Persistent Storage**: Store conversation history in CockroachDB
- **Session Management**: Organize by session/thread ID
- **LangChain Integration**: Drop-in replacement for other chat history implementations

### 🔄 Async & Sync APIs
- **Async-First**: High-performance async operations for I/O concurrency
- **Sync Wrapper**: Simple synchronous API for scripts and batch jobs
- **Connection Pooling**: Efficient connection reuse across async operations

## Quick Example

```python
import asyncio
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine
from langchain_openai import OpenAIEmbeddings

async def main():
    # Initialize engine
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://user:pass@host:26257/db?sslmode=verify-full"
    )
    
    # Create table
    await engine.ainit_vectorstore_table(
        table_name="documents",
        vector_dimension=1536,
    )
    
    # Initialize vector store
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=OpenAIEmbeddings(),
        collection_name="documents",
    )
    
    # Add documents
    await vectorstore.aadd_texts([
        "CockroachDB is a distributed SQL database",
        "LangChain makes it easy to build LLM applications",
    ])
    
    # Search
    results = await vectorstore.asimilarity_search(
        "Tell me about databases",
        k=2
    )
    
    print(results)
    await engine.aclose()

asyncio.run(main())
```

## Why CockroachDB?

- **Distributed by Design**: Scale horizontally across regions
- **Native Vector Support**: First-class `VECTOR` type and C-SPANN indexes
- **SERIALIZABLE**: Strong consistency without sacrificing performance
- **Cloud Native**: Deploy anywhere (AWS, GCP, Azure, on-prem)
- **PostgreSQL Compatible**: Familiar SQL with distributed superpowers

## Getting Started

Choose your path:

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Quick Start__

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-variant:{ .lg .middle } __Guides__

    ---

    Learn key concepts and patterns

    [:octicons-arrow-right-24: Guides](guides/vector-store.md)

-   :material-code-braces:{ .lg .middle } __API Reference__

    ---

    Detailed API documentation

    [:octicons-arrow-right-24: API Docs](api/engine.md)

-   :material-github:{ .lg .middle } __Examples__

    ---

    Working code examples

    [:octicons-arrow-right-24: Examples](examples/index.md)

</div>

## Community & Support

- **GitHub**: [cockroachdb/langchain-cockroachdb](https://github.com/cockroachdb/langchain-cockroachdb)
- **Issues**: [Report bugs or request features](https://github.com/cockroachdb/langchain-cockroachdb/issues)
- **Discussions**: [Ask questions](https://github.com/cockroachdb/langchain-cockroachdb/discussions)

## Contributing

Contributions welcome! This is an open-source project built for the community.

[:octicons-arrow-right-24: Contributing Guide](development/contributing.md)

## License

Apache License 2.0 - see [LICENSE](about/license.md) for details.

---

**Built with ❤️ for the CockroachDB and LangChain communities**
