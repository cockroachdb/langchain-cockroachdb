"""LangChain integration for CockroachDB with native vector support."""

from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore
from langchain_cockroachdb.chat_message_histories import CockroachDBChatMessageHistory
from langchain_cockroachdb.engine import CockroachDBEngine
from langchain_cockroachdb.hybrid_search_config import HybridSearchConfig
from langchain_cockroachdb.indexes import (
    CSPANNIndex,
    CSPANNQueryOptions,
    DistanceStrategy,
)
from langchain_cockroachdb.retry import (
    async_retry_with_backoff,
    is_retryable_error,
    sync_retry_with_backoff,
)
from langchain_cockroachdb.vectorstores import CockroachDBVectorStore

__all__ = [
    "CockroachDBEngine",
    "AsyncCockroachDBVectorStore",
    "CockroachDBVectorStore",
    "CSPANNIndex",
    "CSPANNQueryOptions",
    "DistanceStrategy",
    "HybridSearchConfig",
    "CockroachDBChatMessageHistory",
    "async_retry_with_backoff",
    "sync_retry_with_backoff",
    "is_retryable_error",
]

__version__ = "0.1.0"
