# Vector Store API

Complete API reference for AsyncCockroachDBVectorStore and CockroachDBVectorStore.

::: langchain_cockroachdb.async_vectorstore.AsyncCockroachDBVectorStore
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.vectorstores.CockroachDBVectorStore
    options:
      show_root_heading: true
      show_source: false

## Key Methods

### Adding Documents

| Method | Async | Description |
|--------|-------|-------------|
| `aadd_texts()` | ✓ | Add text documents |
| `add_texts()` | - | Sync wrapper |
| `aadd_documents()` | ✓ | Add Document objects |
| `add_documents()` | - | Sync wrapper |

### Searching

| Method | Async | Description |
|--------|-------|-------------|
| `asimilarity_search()` | ✓ | Search by text |
| `similarity_search()` | - | Sync wrapper |
| `asimilarity_search_with_score()` | ✓ | Search with scores |
| `asimilarity_search_by_vector()` | ✓ | Search by vector |
| `amax_marginal_relevance_search()` | ✓ | MMR search |

### Index Management

| Method | Async | Description |
|--------|-------|-------------|
| `aapply_vector_index()` | ✓ | Create C-SPANN index |
| `apply_vector_index()` | - | Sync wrapper |
| `adrop_vector_index()` | ✓ | Drop index |
| `drop_vector_index()` | - | Sync wrapper |

### Deleting

| Method | Async | Description |
|--------|-------|-------------|
| `adelete()` | ✓ | Delete documents |
| `delete()` | - | Sync wrapper |

## Examples

See [Vector Store Guide](../guides/vector-store.md) for comprehensive examples.
