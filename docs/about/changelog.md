# Changelog

All notable changes to langchain-cockroachdb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-10-26

### Added

- `CockroachDBStore` and `AsyncCockroachDBStore`, LangGraph long-term memory stores on CockroachDB. They implement LangGraph's `BaseStore` with the API of langgraph-checkpoint-postgres's `PostgresStore` (`from_conn_string` with `pipeline` and `pool_config`, `setup()`, `index` and `ttl`), so they work with `create_agent(store=...)`, `StateGraph.compile(store=...)` and Deep Agents' `StoreBackend`. `from_conn_string` also takes `retry_policy`, `connect_timeout`, `prepare_threshold` and `connect_kwargs`, as the checkpointers do. The two stores, `CockroachDBIndexConfig`, `CockroachDBTTLConfig` and `PoolConfig` are exported from `langchain_cockroachdb.store`, and all of them but `PoolConfig` from `langchain_cockroachdb` too.
- Semantic search on CockroachDB's `VECTOR` type, with a C-SPANN index that has the namespace as a prefix column. `index` takes LangGraph's `dims`, `embed` and `fields`, plus `distance_type` (`cosine`, `l2` or `inner_product`), `cspann` partition sizes, `flat` to skip the index, and `beam_size`, which is set for each search. A search under a namespace first lists the namespaces at or under it, then ranks with the index within them, and runs a second, exact query when a filter or expired items leave its page short, unless a count shows that the index returned every vector of those namespaces. A search of the whole store, every search of a `flat` store, and a search over more than 1000 namespaces (`MAX_SEARCH_NAMESPACES`) rank every match exactly instead. So a filter or expired items never leave a page short while more matches exist.
- Expiry through CockroachDB row-level TTL. With a `ttl` configuration, `setup()` turns on the TTL job for the items and vectors tables, on a schedule from `sweep_interval_minutes` or `ttl_job_cron` (`@hourly` by default), so no sweeper thread is needed. Vectors carry their item's expiry. Expired items are left out of reads, searches and listings by default, even by a store built without `ttl`; `ttl={"omit_expired": False}` turns that off. With `refresh_on_read`, a read pushes back the expiry of the items it returns, in the same statement. `sweep_ttl()`, `start_ttl_sweeper()` and `stop_ttl_sweeper()` work as in PostgresStore. A store built without `ttl` applies no default expiry, never refreshes one, and its `setup()` leaves an existing TTL setting alone. On CockroachDB v25.4, `disable_ttl()`, `adisable_ttl()` and the TTL step of `setup()` retry while another process changes the same table.
- `delete_namespace(namespace_prefix)` and `adelete_namespace`, which delete every item and vector in a namespace and its descendants in one transaction and return the number of items deleted.
- The sync methods of `AsyncCockroachDBStore` (`batch`, `get`, `put`, `search`, `delete` and `list_namespaces`) raise `RuntimeError` instead of hanging when the store's event loop is closed or not running, or when called from a forked child, as those of `AsyncCockroachDBSaver` do. From the loop's own thread they raise `asyncio.InvalidStateError`.
- Differences from PostgresStore: the puts and deletes of one batch commit in one transaction; every call retries transient errors under the store's `RetryPolicy`; a `CockroachDBStore` on a pool runs batches concurrently; putting an item again or deleting it removes its old vectors; a filter that compares with a number matches numeric values only (PostgresStore compares text), nested dictionaries match nested keys, and `$ne` matches items without the key; results with the same score or `updated_at` are ordered by namespace and key, so paging is stable; `table_name` and `schema` configure the tables (defaults `store`, `store_vectors` and `store_migrations` in `public`); a `query` on a store without `index` logs a WARNING and runs a plain search; after `setup()`, a changed `dims` or `distance_type` raises, and other partition sizes, or `flat` with an index present, log a WARNING; `disable_ttl()` and `adisable_ttl()` remove the table TTL setting, and `drop_vector_index()` and `adrop_vector_index()` drop the vector index so that `setup()` can build it again, for example with another distance; and there is no `deserializer=` argument, because psycopg decodes JSONB before the store sees it.
- `examples/long_term_memory.py`, a LangGraph agent that saves a memory in one thread and recalls it in another, and a guide, an example page and an API page for the store in the documentation.

### Fixed

- `enable_ttl` and `disable_ttl` (and `aenable_ttl` and `adisable_ttl`) on the checkpointers no longer fail when several processes run them at once on CockroachDB v25.4. That version refuses to change a table's TTL settings while another schema change on it runs, with SQLSTATE 0A000, which was not retried. These TTL changes now retry that error under the saver's retry policy, even if its `retry_on` would refuse it; `max_attempts=1` turns this off. The code and phrase are exported as `SCHEMA_CHANGE_BUSY_SQLSTATE` and `SCHEMA_CHANGE_BUSY_MESSAGES`, with `is_schema_change_busy()` and `schema_change_policy()`.
- `CockroachDBSaver`, `AsyncCockroachDBSaver`, `CockroachDBStore` and `AsyncCockroachDBStore` could hang on a slow or TLS link when they ran on one connection in pipeline mode (`pipe=`, or `from_conn_string(..., pipeline=True)`). psycopg's pipeline sync can leave the last bytes of a write unsent, so the server waited for them while the client waited for the server. The library now sends them before it waits. Pools, and single connections without a pipeline, were not affected.

## [0.4.1] - 2026-10-16

### Performance

- Checkpointers on a psycopg_pool pool now run operations concurrently, each on its own pooled connection, so one saver can serve many agents, threads or tasks. Before, every operation in the process waited on one lock, which on a multi-region cluster capped a saver at a few operations per second. A saver on a single connection (including `pipeline=True`) still runs one operation at a time. Concurrency is bounded by the pool's `max_size`; when every connection is busy, operations wait for the pool (30 s by default) and a `PoolTimeout` is not retried. On a 9-node, 3-region CockroachDB cluster (Docker, with simulated cross-region latency and a stubbed model) with 16 LangGraph agents sharing one pooled saver, throughput went from 0.27 turns per second (median turn 55 s) in 0.4.0 to about 7.5 turns per second (median turn about 1.2 s) in 0.4.1.

### Fixed

- `put` with channel blobs, `put_writes` with several writes, and `delete_thread` now send BEGIN, their statements and COMMIT as one batch, so each write is saved completely or not at all, even when the caller is cancelled or fails partway. Before, an interrupted write could leave blob rows without their checkpoint, part of a task's pending writes (which LangGraph treats as a finished task), or a half-deleted thread. A write still takes one round trip, and CockroachDB can still retry it on the server. Single-statement writes are unchanged. Connections not in autocommit mode use psycopg's `conn.transaction()` (the docs already require `autocommit=True`).
- On a single connection that the caller also uses, a write made inside a transaction the caller opened joins it as a savepoint, so the caller's commit or rollback decides. The saver never rolls back a transaction it did not open, and an error inside the caller's transaction is raised at once, without retries, so the caller's own retry loop can restart the whole transaction.
- Pooled checkpointers retry when psycopg reports the connection broken, even when the error text is unfamiliar. This fixed checkpointer retries over TLS: on CockroachDB Cloud a dropped connection reads "SSL error: unexpected eof while reading" (OpenSSL 3), which was not recognized, so a pooled saver gave up instead of retrying. That wording is now also in `TRANSIENT_MESSAGES`. A definite SQLSTATE still decides first, as on the SQLAlchemy path.
- `AsyncCockroachDBSaver`'s sync methods raise a clear `RuntimeError` instead of hanging when the saver's loop is closed, stopped, or the process forked, and they cancel the abandoned work. `put` and `put_writes` called from the saver's own loop thread now raise `InvalidStateError` instead of deadlocking (the other sync methods already did).
- Engines built with `loop=` (`CockroachDBEngine(engine, loop=...)` or `CockroachDBEngine.from_engine(engine, loop=...)`) fail fast with a clear `RuntimeError` instead of hanging when that loop is closed, never started, stops or closes during a call, or the process forked (including an engine rebuilt in a child around the parent's loop). A loop that has not started yet gets a short grace period (`LOOP_START_TIMEOUT`, 1 s).
- `RetryPolicy` rejects NaN and infinity in its numeric fields, and `delay()` can no longer overflow for large attempt numbers or multipliers.
- The flat `jitter` / `retry_jitter` argument now raises `TypeError` for anything other than `True`, `False`, `0` or `1`. It used to be read by truthiness, so 0.3 became a jitter of 0.5. Use `RetryPolicy(jitter=...)` for a ratio.
- Retry warnings name the underlying driver error when an application wraps it, and never include SQL text or parameter values.
- `aadd_texts`, `add_texts`, `afrom_texts` and `from_texts` validate input (lengths, ids, duplicates, metadata JSON and `batch_size`) before calling the embedding model, so invalid input no longer pays for embedding calls. The namespace conflict check is a database query and still runs after embedding.
- Applying a vector index whose name already exists with different prefix columns (order or count) or a different partition size you set now logs a WARNING naming each difference, like the existing opclass warning, instead of doing nothing silently. Applying the same index again logs nothing. A partition size you leave unset accepts whatever the index has.
- Checkpointer setup logs its bootstrap step as "checkpointer setup" instead of a second "checkpointer migration 0".

## [0.4.0] - 2026-10-14

### Added
- `RetryPolicy`: one configurable object for which errors are retried and how, set on the engine and inherited by the vector store and chat history; checkpointers take one too. Exported defaults (`RETRYABLE_SQLSTATES`, `TRANSIENT_MESSAGES`, ...) and `get_sqlstate()`.
- Retries for 40001, 40003, every 08xxx code, 57P01, 58C00, 58C01 and 53300, plus connection loss reported without a code. Failed logins and SSL or configuration errors are never retried.
- Optional hang guard (`attempt_timeout`, `ddl_attempt_timeout`) that abandons and retries an attempt the server never answers.
- `aadd_embeddings`/`add_embeddings`, `amax_marginal_relevance_search_by_vector`, the full langchain-postgres filter grammar (`$between`, `$exists`, `$like`, `$ilike`, `$not`).
- `CockroachDBChatMessageHistory` accepts a shared `CockroachDBEngine` and has a public `acreate_table_if_not_exists()`.
- Checkpointer `from_conn_string` options: `retry_policy`, `connect_timeout`, `prepare_threshold`, `connect_kwargs`.
- `connect_timeout` on `CockroachDBEngine.from_connection_string` (default 10 seconds).
- `CockroachDBEngine.from_engine(engine, loop=...)`.
- `postgresql://` and `cockroachdb://` URL spellings are normalized to the psycopg driver.
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
- New connections set `gssencmode=disable` unless the URL or connect args choose otherwise. CockroachDB has no GSS encryption, and this avoids a Kerberos lookup that can crash forked workers on macOS.
- `adrop_vector_index()` and `drop_vector_index()` can be called without an index to drop the default-named index, as the upgrade note says.
- `from_texts(namespace=...)` now creates the namespace column.
- Two spellings of the same UUID in one call are rejected as duplicates.

### Changed
- psycopg 3 only: `psycopg2-binary` is no longer a dependency. Requires `langchain-core` 1.x and caps every dependency at its next major version. Python 3.10 to 3.14.
- Supports CockroachDB v25.4 or later; tested on v25.4 LTS and v26.3.
- The vector store's retry settings default to the engine's (5 attempts) instead of 3.
- `max_retries < 1` raises `ValueError` instead of never calling the function.
- Table, schema, column and index names must be plain SQL identifiers.
- Indexes created by 0.3.x for COSINE or INNER_PRODUCT should be dropped and applied again.
- `add_texts`/`add_embeddings` reject duplicate ids within one call (`ValueError`), and metadata that is not JSON serializable, including NaN and infinity (`TypeError`). Both checks run before anything is written. They used to fail partway, after earlier batches had committed.
- Range filters (`$gt`, `$gte`, `$lt`, `$lte`, `$between`) with a string value now compare as text, and only match string fields, like langchain-postgres. In 0.3.0, `{"year": {"$gte": "2020"}}` matched numeric fields; pass a number to compare numerically.
- Top-level keys next to `$and`/`$or` used to be ignored. They are now combined with AND, so results can narrow.
- The `retry_*` attributes on the engine and vector store are now read-only properties. Set them through `retry_policy` or the constructor.
- Chat message ids are generated by the client before the first attempt.
- `batch_size` is capped at 16,383. Long id lists are split automatically.

## [0.3.0] - 2026-07-15

### Fixed
- Hybrid search now actually executes. Setting `hybrid_search_config`
  previously had no effect: only vector similarity ran and the FTS half never
  fired (#7). The regular search methods now run both searches in parallel
  and fuse the results.

### Added
- `fts_language` parameter on `ainit_vectorstore_table` for non-English
  tsvector columns
- `fetch_k` keyword argument to control the hybrid candidate pool size
- `fts_rank_normalization` option exposing the ts_rank bitmask
- Score normalization for weighted sum fusion

### Changed
- Default fusion method is now reciprocal rank fusion. Pass
  `fusion_type="weighted_sum"` for the previous default.
- `HybridSearchConfig` validates `fts_query_language`

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
