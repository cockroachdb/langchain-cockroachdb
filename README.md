# <img src="https://raw.githubusercontent.com/viragtripathi/langchain-cockroachdb/main/assets/cockroachdb_logo.svg" alt="🪳" width="25" height="25" style="vertical-align: middle;"/> langchain-cockroachdb

[![Tests](https://github.com/viragtripathi/langchain-cockroachdb/actions/workflows/test.yml/badge.svg)](https://github.com/viragtripathi/langchain-cockroachdb/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/viragtripathi/langchain-cockroachdb/branch/main/graph/badge.svg)](https://codecov.io/gh/viragtripathi/langchain-cockroachdb)
[![PyPI version](https://badge.fury.io/py/langchain-cockroachdb.svg)](https://badge.fury.io/py/langchain-cockroachdb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/badge/langchain-cockroachdb/month)](https://pepy.tech/project/langchain-cockroachdb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <strong>LangChain integration for CockroachDB with native vector support</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="https://viragtripathi.github.io/langchain-cockroachdb/">Documentation</a> •
  <a href="https://github.com/viragtripathi/langchain-cockroachdb/tree/main/examples">Examples</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Overview

Build LLM applications with **CockroachDB's distributed SQL database** and **native vector search** capabilities. This integration provides:

- 🎯 **Native Vector Support** - CockroachDB's `VECTOR` type
- 🚀 **C-SPANN Indexes** - Distributed vector indexes optimized for scale
- 🔄 **Automatic Retries** - Handles serialization errors transparently
- ⚡ **Async & Sync APIs** - Choose based on your use case
- 🏗️ **Distributed by Design** - Built for CockroachDB's architecture

## Quick Start

### Installation

```bash
pip install langchain-cockroachdb
```

### Basic Usage

```python
import asyncio
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine
from langchain_openai import OpenAIEmbeddings

async def main():
    # Initialize
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://user:pass@host:26257/db"
    )
    
    await engine.ainit_vectorstore_table(
        table_name="documents",
        vector_dimension=1536,
    )
    
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=OpenAIEmbeddings(),
        collection_name="documents",
    )
    
    # Add documents
    await vectorstore.aadd_texts([
        "CockroachDB is a distributed SQL database",
        "LangChain makes building LLM apps easy",
    ])
    
    # Search
    results = await vectorstore.asimilarity_search(
        "Tell me about databases",
        k=2
    )
    
    for doc in results:
        print(doc.page_content)
    
    await engine.aclose()

asyncio.run(main())
```

## Features

### Vector Store
- Native `VECTOR` type support with C-SPANN indexes
- Advanced metadata filtering (`$and`, `$or`, `$gt`, `$in`, etc.)
- Hybrid search (full-text + vector similarity)
- Multi-tenant index support with prefix columns

### Reliability
- Automatic retry logic with exponential backoff
- Connection pooling with health checks
- Configurable for different workloads
- Built for SERIALIZABLE isolation

### Developer Experience
- Async-first design for high concurrency
- Sync wrapper for simple scripts
- Type-safe with full type hints
- Comprehensive test suite (92 tests)

## Documentation

**📚 [Complete Documentation](https://viragtripathi.github.io/langchain-cockroachdb/)**

**Getting Started:**
- [Installation](https://viragtripathi.github.io/langchain-cockroachdb/getting-started/installation/)
- [Quick Start](https://viragtripathi.github.io/langchain-cockroachdb/getting-started/quick-start/)
- [Configuration](https://viragtripathi.github.io/langchain-cockroachdb/getting-started/configuration/)

**Guides:**
- [Vector Store](https://viragtripathi.github.io/langchain-cockroachdb/guides/vector-store/)
- [Vector Indexes](https://viragtripathi.github.io/langchain-cockroachdb/guides/vector-indexes/)
- [Hybrid Search](https://viragtripathi.github.io/langchain-cockroachdb/guides/hybrid-search/)
- [Chat History](https://viragtripathi.github.io/langchain-cockroachdb/guides/chat-history/)
- [Async vs Sync](https://viragtripathi.github.io/langchain-cockroachdb/guides/async-vs-sync/)

## Examples

**🔧 [Working Examples](https://github.com/viragtripathi/langchain-cockroachdb/tree/main/examples)**

- [`quickstart.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/quickstart.py) - Get started in 5 minutes
- [`sync_usage.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/sync_usage.py) - Synchronous API
- [`vector_indexes.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/vector_indexes.py) - Index optimization
- [`hybrid_search.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/hybrid_search.py) - FTS + vector search
- [`metadata_filtering.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/metadata_filtering.py) - Advanced queries
- [`chat_history.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/chat_history.py) - Persistent conversations
- [`retry_configuration.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/retry_configuration.py) - Configuration patterns

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/viragtripathi/langchain-cockroachdb.git
cd langchain-cockroachdb

# Install dependencies
pip install -e ".[dev]"

# Start CockroachDB
docker-compose up -d

# Run tests
make test
```

### Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Open http://127.0.0.1:8000
```

### Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/CONTRIBUTING.md) for guidelines.

## Why CockroachDB?

- **Distributed SQL** - Scale horizontally across regions
- **Native Vector Support** - First-class `VECTOR` type and C-SPANN indexes  
- **Strong Consistency** - SERIALIZABLE isolation by default
- **Cloud Native** - Deploy anywhere (AWS, GCP, Azure, on-prem)
- **PostgreSQL Compatible** - Familiar SQL with distributed superpowers

## License

Apache License 2.0 - see [LICENSE](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/LICENSE) for details.

## Acknowledgments

Built for the CockroachDB and LangChain communities.

- [CockroachDB](https://www.cockroachlabs.com/) - Distributed SQL database
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework

## Links

- [GitHub Repository](https://github.com/viragtripathi/langchain-cockroachdb)
- [PyPI Package](https://pypi.org/project/langchain-cockroachdb/)
- [CockroachDB Documentation](https://www.cockroachlabs.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)
- [Report Issues](https://github.com/viragtripathi/langchain-cockroachdb/issues)
