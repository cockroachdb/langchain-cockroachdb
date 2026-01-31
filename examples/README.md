# Examples

Comprehensive examples demonstrating langchain-cockroachdb features.

## Running Examples

### Prerequisites

Start CockroachDB locally:

```bash
docker-compose up -d
```

Set connection string (optional):

```bash
export COCKROACHDB_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
```

### Run Examples

```bash
# Quickstart - basic vector store usage
python examples/quickstart.py

# Vector indexes - C-SPANN index creation and tuning
python examples/vector_indexes.py

# Hybrid search - combining FTS and vector search
python examples/hybrid_search.py

# Metadata filtering - advanced filter queries
python examples/metadata_filtering.py

# Chat history - conversation persistence
python examples/chat_history.py
```

## Examples Overview

### 1. quickstart.py

Basic vector store operations:
- Connecting to CockroachDB
- Creating tables
- Adding documents
- Similarity search
- Filtering by metadata

**Good for**: First-time users, getting started

### 2. vector_indexes.py

Vector index management:
- Creating C-SPANN indexes
- Configuring partition sizes
- Query-time beam size tuning
- Multiple distance strategies
- Multi-tenant prefix columns

**Good for**: Performance optimization, production deployments

### 3. hybrid_search.py

Combining FTS and vector search:
- TSVECTOR column creation
- Weighted score fusion
- Reciprocal Rank Fusion (RRF)
- Fusion strategy comparison

**Good for**: Best search quality, combining keyword + semantic

### 4. metadata_filtering.py

Advanced metadata queries:
- Equality and comparison operators
- IN/NOT IN operators
- Logical AND/OR combinations
- Nested filter expressions
- Complex filter patterns

**Good for**: Multi-tenant applications, filtered search

### 5. chat_history.py

Persistent conversation storage:
- Session management
- Message persistence
- Session isolation
- Integration with LangChain chains

**Good for**: Chatbots, conversational AI

## Next Steps

After running examples, explore:

- [DEVELOPMENT.md](../DEVELOPMENT.md) - Architecture and patterns
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [README.md](../README.md) - Full documentation
