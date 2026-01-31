# Indexes API

C-SPANN index configuration.

::: langchain_cockroachdb.indexes.CSPANNIndex
    options:
      show_root_heading: true
      show_source: false

::: langchain_cockroachdb.indexes.DistanceStrategy
    options:
      show_root_heading: true
      show_source: false

## Distance Strategies

| Strategy | Operator | Best For |
|----------|----------|----------|
| `COSINE` | `<=>` | Normalized vectors (most embeddings) |
| `EUCLIDEAN` | `<->` | Spatial data, L2 distance |
| `INNER_PRODUCT` | `<#>` | Pre-normalized vectors |

## Examples

See [Vector Indexes Guide](../guides/vector-indexes.md) for complete examples.
