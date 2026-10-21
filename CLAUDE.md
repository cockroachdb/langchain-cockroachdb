# CLAUDE.md

Notes for anyone working on langchain-cockroachdb, whether you are a person
or an AI coding assistant such as Claude Code, which reads this file at the
start of a session. Contributor setup lives in [CONTRIBUTING.md](CONTRIBUTING.md).
Deeper design notes live in [docs/development](docs/development).

## What this project is

A LangChain and LangGraph integration for CockroachDB:

- a vector store, async (`AsyncCockroachDBVectorStore`) and sync
  (`CockroachDBVectorStore`), on CockroachDB's native `VECTOR` type
- C-SPANN vector indexes
- hybrid search (full-text plus vector)
- chat message history
- the LangGraph checkpointers `CockroachDBSaver` and `AsyncCockroachDBSaver`
- the LangGraph long-term memory stores `CockroachDBStore` and
  `AsyncCockroachDBStore`, with semantic search and row-level TTL

The vector store, chat history and engine use SQLAlchemy's async engine with
psycopg 3. The checkpointers and the store use psycopg directly, like
LangGraph's PostgresSaver and PostgresStore.

CockroachDB runs every transaction at SERIALIZABLE. The library does not ask
users to switch to READ COMMITTED. Instead it retries transient errors, and
every write it makes is safe to run twice.

## Code map

| Path | What it holds |
|---|---|
| `langchain_cockroachdb/engine.py` | `CockroachDBEngine`: connection pool, retry policy, table setup (`ainit_vectorstore_table`) |
| `langchain_cockroachdb/async_vectorstore.py`, `vectorstores.py` | Vector store (async, and the sync subclass) |
| `langchain_cockroachdb/indexes.py` | `CSPANNIndex`, `CSPANNQueryOptions` |
| `langchain_cockroachdb/hybrid_search_config.py` | Hybrid search settings and score fusion |
| `langchain_cockroachdb/chat_message_histories.py` | `CockroachDBChatMessageHistory` |
| `langchain_cockroachdb/checkpointer/` | `base.py` (SQL, migrations), `saver.py`, `async_saver.py` |
| `langchain_cockroachdb/store/` | `store.py` (`CockroachDBStore`), `aio.py` (`AsyncCockroachDBStore`), `base.py` (batch planning both share), `sql.py`, `config.py`, `filters.py`, `namespaces.py` |
| `langchain_cockroachdb/retry.py` | `RetryPolicy` and error classification by SQLSTATE |
| `langchain_cockroachdb/_psycopg_ops.py` | Retries, hang guard and one-batch writes shared by the checkpointers and the store |
| `langchain_cockroachdb/_psycopg_conn.py` | Connection checkout for a psycopg connection or pool, and the errors shared by the checkpointers and the store |
| `langchain_cockroachdb/_guard.py` | Hang guard that abandons an attempt the server never answers |
| `langchain_cockroachdb/_loop.py` | Background event loop for the sync API, reset after `fork()` |
| `langchain_cockroachdb/_connection.py` | URL normalization and connection defaults (`connect_timeout`, `gssencmode`) |
| `langchain_cockroachdb/_sql.py`, `_filters.py` | Identifier validation, vector literals, metadata filters as bound parameters |
| `tests/unit/` | Fast tests with no database |
| `tests/integration/` | Tests against a real CockroachDB; `faults.py` injects real CockroachDB errors |
| `tests/e2e/` | Runs every script in `examples/` the way a user would |
| `docs/`, `mkdocs.yml` | The documentation site |

## Rules that keep the library correct

These come from real bugs. Please keep them.

1. **Every SQL value is a bound parameter.** Only identifiers are formatted into SQL text. Validate them first with `validate_identifier` in `_sql.py`. Write casts as `CAST(:x AS type)`, never `:x::type`. Metadata filters go through `FilterBuilder` in `_filters.py`, which binds keys and values.
2. **Every write is idempotent.** Use UPSERT, `INSERT ... ON CONFLICT DO NOTHING`, DELETE, or `IF [NOT] EXISTS`. Operations retry after ambiguous results (SQLSTATE 40003, a dropped connection), so a write may have committed even though the client saw an error. Choose ids before the first attempt, never inside it.
3. **Database work goes through the retry machinery.** That means `_execute` in the vector store, `_run` in chat history, `_retry` with `_cursor` in the checkpointers, and `_run_ddl` in the engine. Check out the connection inside each attempt. Never hold a lock or a connection while backing off.
4. **Schema changes run one statement per implicit transaction,** and each statement must be idempotent. With `autocommit_before_ddl` off, an explicit transaction cannot change a schema_locked table and fails with 57000. Tables are schema_locked by default on CockroachDB v26. In psycopg pipeline mode, sync after each schema statement for the same reason.
5. **Nothing is hardcoded.** Retry behavior lives in `RetryPolicy` fields with exported defaults (`RETRYABLE_SQLSTATES`, `TRANSIENT_MESSAGES` and others in `retry.py`). Connection defaults give way to a value in the URL or connect options.
6. **Async methods run on the caller's event loop. Sync methods run on the shared background loop** through `engine._run_as_sync`. Do not call `asyncio.run()` in library code.
7. **Keep VECTOR insert batches small.** The default `batch_size` is 100. One statement can carry at most 65,535 bound parameters, so long id lists are split into chunks.
8. **Namespaces isolate tenants.** Reads and deletes in a namespaced store are scoped to its namespace, and inserts never write over another namespace's ids.

## Running the tests

```bash
uv pip install -e ".[dev]"            # or: make install
.venv/bin/pytest tests/unit           # no database needed
```

Integration and end-to-end tests need CockroachDB. By default the test suite
starts one with testcontainers. To use your own instead, for example from
`make start-db` (docker compose, port 26257):

```bash
USE_TESTCONTAINER=false \
COCKROACHDB_URL='cockroachdb://root@localhost:26257/defaultdb?sslmode=disable' \
.venv/bin/pytest tests/integration tests/e2e
```

- The supported versions are CockroachDB v25.4 LTS and later. Test changes that touch SQL on v25.4 and on the latest v26.x, because their defaults differ (for example, `create_table_with_schema_locked`).
- Two TTL expiration tests wait a few minutes for CockroachDB's TTL job. While iterating, skip them with `-k "not ttl_expiration"`.
- Fault tests make CockroachDB raise real errors through `crdb_internal.force_error`, `CANCEL SESSIONS` and held row locks. Look at `tests/integration/faults.py` before mocking anything.
- Do not run two test processes against the same database at the same time. The suites share table names.

## Checks

`make check` runs `ruff check`, `ruff format --check` and mypy. The versions
are pinned in `.github/workflows/_lint.yml`, and mypy is blocking. CI also
runs:
- the tests on Python 3.10 to 3.14
- the tests at the lowest declared dependency versions
- the tests against CockroachDB v25.4
- `mkdocs build --strict` for the docs

## Writing style

Write plain, human text in code comments, docstrings, docs, the CHANGELOG and
commit messages. Use short sentences, no em dashes and no marketing language.
Commits use conventional prefixes (`fix:`, `feat:`, `test:`, `docs:`, `ci:`,
`build:`, `chore:`). Commits made with an AI assistant end with a
`Co-Authored-By:` trailer naming it.

## Pull requests and releases

- Pull requests are squash merged. Include tests for behavior changes, and a CHANGELOG entry for anything a user would notice. Keep `CHANGELOG.md` and `docs/about/changelog.md` the same.
- To release:
  1. In the release PR itself, bump `version` in `pyproject.toml` and `__version__` in `langchain_cockroachdb/__init__.py`, and date the CHANGELOG entry in both files. The merge commit is then ready to release.
  2. Run the **Release** workflow by hand with the version. It checks that the version and a dated CHANGELOG entry are in place, runs the tests, builds, publishes to PyPI through trusted publishing, and creates the tag and a GitHub release whose notes start with the CHANGELOG entry.
  3. Keep `release.yml` inlined. Trusted publishing broke when it was split into a reusable workflow.
- The documentation site deploys from `main` on its own.
- When user-facing behavior changes, also update the CockroachDB pages in the LangChain docs (the langchain-ai/docs repository). Links in those pages stay within docs.langchain.com.
