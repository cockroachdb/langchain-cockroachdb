# Chat History API

Persistent conversation storage.

::: langchain_cockroachdb.chat_message_histories.CockroachDBChatMessageHistory
    options:
      show_root_heading: true
      show_source: false

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | str | Required | Session or thread identifier |
| `connection_string` | str | None | Database connection string (if engine not provided) |
| `engine` | CockroachDBEngine | None | Shared engine instance |
| `table_name` | str | "message_store" | Table name for messages |
| `retry_policy` | RetryPolicy | None | Retry policy (inherits from engine if not set) |

## Key Methods

| Method | Description |
|--------|-------------|
| `add_user_message()` | Add user message |
| `add_ai_message()` | Add AI response |
| `add_message()` | Add any message type |
| `add_messages()` | Add multiple messages atomically |
| `clear()` | Clear session history |
| `messages` | Get all messages |
| `acreate_table_if_not_exists()` | Create table if needed (async) |
| `create_table_if_not_exists()` | Create table if needed (sync) |

## Examples

See [Chat History Guide](../guides/chat-history.md) for comprehensive examples.
