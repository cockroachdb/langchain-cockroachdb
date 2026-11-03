"""LangChain integration for CockroachDB with native vector support."""

from langchain_cockroachdb.async_vectorstore import (
    VECTOR_INDEX_REFRESH_SECONDS,
    AsyncCockroachDBVectorStore,
)
from langchain_cockroachdb.chat_message_histories import CockroachDBChatMessageHistory
from langchain_cockroachdb.checkpointer import AsyncCockroachDBSaver, CockroachDBSaver
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.hybrid_search_config import FusionType, HybridSearchConfig
from langchain_cockroachdb.indexes import (
    CSPANNIndex,
    CSPANNQueryOptions,
    DistanceStrategy,
)
from langchain_cockroachdb.retry import (
    NON_TRANSIENT_MESSAGES,
    RETRYABLE_SQLSTATE_CLASSES,
    RETRYABLE_SQLSTATES,
    SCHEMA_CHANGE_BUSY_MAX_BACKOFF,
    SCHEMA_CHANGE_BUSY_MESSAGES,
    SCHEMA_CHANGE_BUSY_SECONDS,
    SCHEMA_CHANGE_BUSY_SQLSTATE,
    SCHEMA_CHANGE_PROGRESS_SECONDS,
    SCHEMA_CHANGE_QUIET_SECONDS,
    SCHEMA_CHANGE_SETTLE_SECONDS,
    TRANSIENT_MESSAGES,
    UNCATEGORIZED_SQLSTATES,
    RetryPolicy,
    async_retry_with_backoff,
    get_sqlstate,
    is_retryable_error,
    is_schema_change_busy,
    schema_change_policy,
    sync_retry_with_backoff,
)
from langchain_cockroachdb.store import (
    AsyncCockroachDBStore,
    CockroachDBIndexConfig,
    CockroachDBStore,
    CockroachDBTTLConfig,
)
from langchain_cockroachdb.vectorstores import CockroachDBVectorStore

__all__ = [
    "CockroachDBEngine",
    "AsyncCockroachDBVectorStore",
    "CockroachDBVectorStore",
    "CSPANNIndex",
    "CSPANNQueryOptions",
    "DistanceStrategy",
    "VECTOR_INDEX_REFRESH_SECONDS",
    "FusionType",
    "HybridSearchConfig",
    "CockroachDBChatMessageHistory",
    "CockroachDBSaver",
    "AsyncCockroachDBSaver",
    "CockroachDBStore",
    "AsyncCockroachDBStore",
    "CockroachDBIndexConfig",
    "CockroachDBTTLConfig",
    "async_retry_with_backoff",
    "sync_retry_with_backoff",
    "is_retryable_error",
    "RetryPolicy",
    "get_sqlstate",
    "RETRYABLE_SQLSTATES",
    "RETRYABLE_SQLSTATE_CLASSES",
    "UNCATEGORIZED_SQLSTATES",
    "TRANSIENT_MESSAGES",
    "NON_TRANSIENT_MESSAGES",
    "SCHEMA_CHANGE_BUSY_SQLSTATE",
    "SCHEMA_CHANGE_BUSY_MESSAGES",
    "SCHEMA_CHANGE_BUSY_MAX_BACKOFF",
    "SCHEMA_CHANGE_BUSY_SECONDS",
    "SCHEMA_CHANGE_SETTLE_SECONDS",
    "SCHEMA_CHANGE_QUIET_SECONDS",
    "SCHEMA_CHANGE_PROGRESS_SECONDS",
    "is_schema_change_busy",
    "schema_change_policy",
]

__version__ = "0.5.1"
