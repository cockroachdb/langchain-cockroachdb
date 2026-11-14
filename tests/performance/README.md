# Vector retrieval benchmark

Install the development dependencies, then run against a disposable CockroachDB
database:

```bash
export COCKROACHDB_URL='cockroachdb://root@localhost:26257/defaultdb?sslmode=disable'
python tests/performance/bench_vector_search.py --rows 10000 --dims 128 --queries 100 --beam 16 32 64 > benchmark.json
```

The script creates two tables with identical seeded, normalized random vectors.
One has no vector index and provides the exact baseline; the other has a
namespace-prefixed C-SPANN index. Both searches select the same namespace and
metadata predicate. The script drops only its own tables when it exits normally.
A killed process can leave tables prefixed `bench_exact_` or `bench_index_`.

The JSON includes database and Python versions, seed, dimensions, row/query counts,
beam size, ingestion rate before index creation, index-build time, recall@k,
p50/p95 retrieval latency, and counts of the library's search paths. Filters match
approximately 100%, 10%, and 1% of the data (the exact fraction varies for small
row counts). Recall uses the baseline result count when fewer than k matches
exist. Warm-up searches are excluded. Search timing excludes embeddings; ingestion
uses precomputed vectors. The script holds the synthetic dataset in memory.

These are single-client measurements on synthetic data. Record the node count,
CPU/RAM, regions, network latency, and concurrent workload alongside results before
comparing runs. Replace the dataset and queries with representative embeddings to
measure useful recall for an application. An exact fallback can yield recall 1
while being slower; inspect the path counts as well as the score. The `direct`
path does not prove the optimizer chose an index. This script exercises filtered
search only and does not measure hybrid answer quality or end-to-end model latency.
