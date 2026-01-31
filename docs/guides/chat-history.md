# Chat History Guide

Persist conversation history in CockroachDB for chatbots and conversational AI applications.

## Overview

`CockroachDBChatMessageHistory` provides persistent storage for conversation messages with:
- Session-based organization
- Automatic table creation
- LangChain integration
- Thread isolation

## Basic Usage

### Initialization

```python
from langchain_cockroachdb import CockroachDBChatMessageHistory

# Create message history for a session
history = CockroachDBChatMessageHistory(
    session_id="user-123",
    connection_string="cockroachdb://user:pass@host:26257/db",
)
```

### Adding Messages

```python
# Add user message
history.add_user_message("Hello! How are you?")

# Add AI response
history.add_ai_message("I'm doing well, thank you! How can I help you today?")
```

### Retrieving Messages

```python
# Get all messages in session
messages = history.messages

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

### Clearing History

```python
# Clear specific session
history.clear()

# Or create new history object
history = CockroachDBChatMessageHistory(
    session_id="user-123",
    connection_string=connection_string,
)
history.clear()
```

## Table Management

### Create Table

```python
# Table is created automatically on first use
# Or create explicitly:
await CockroachDBChatMessageHistory.acreate_tables(connection_string)
```

Default table schema:
```sql
CREATE TABLE IF NOT EXISTS message_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    message JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON message_store (session_id);
```

### Custom Table Name

```python
history = CockroachDBChatMessageHistory(
    session_id="user-123",
    connection_string=connection_string,
    table_name="my_chat_history",
)
```

### Drop Table

```python
await CockroachDBChatMessageHistory.adrop_tables(
    connection_string,
    table_name="message_store"
)
```

## Integration with LangChain

### With Conversation Chains

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_cockroachdb import CockroachDBChatMessageHistory

# Create message history
history = CockroachDBChatMessageHistory(
    session_id="user-123",
    connection_string=connection_string,
)

# Create memory
memory = ConversationBufferMemory(
    chat_memory=history,
    return_messages=True,
)

# Create chain
llm = ChatOpenAI()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# Chat
response = conversation.predict(input="Hello!")
print(response)

# History is automatically saved to CockroachDB
```

### With LCEL Chains

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Create chain
chain = prompt | ChatOpenAI()

# Wrap with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: CockroachDBChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string,
    ),
    input_messages_key="input",
    history_messages_key="history",
)

# Use with session
response = chain_with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-123"}},
)
print(response.content)
```

### With ConversationBufferWindowMemory

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last 5 messages in memory
memory = ConversationBufferWindowMemory(
    k=5,  # Last 5 exchanges
    chat_memory=CockroachDBChatMessageHistory(
        session_id="user-123",
        connection_string=connection_string,
    ),
    return_messages=True,
)
```

## Multi-User Applications

### Per-User Sessions

```python
def get_session_history(user_id: str) -> CockroachDBChatMessageHistory:
    """Get chat history for a specific user."""
    return CockroachDBChatMessageHistory(
        session_id=f"user-{user_id}",
        connection_string=connection_string,
    )

# Use for different users
user_1_history = get_session_history("user-123")
user_2_history = get_session_history("user-456")

# Each user has isolated conversation
user_1_history.add_user_message("What's the weather?")
user_2_history.add_user_message("Tell me a joke")
```

### Multiple Conversations Per User

```python
def get_conversation_history(user_id: str, conversation_id: str):
    """Get specific conversation for a user."""
    return CockroachDBChatMessageHistory(
        session_id=f"{user_id}:{conversation_id}",
        connection_string=connection_string,
    )

# User can have multiple conversations
main_chat = get_conversation_history("user-123", "main")
support_chat = get_conversation_history("user-123", "support")
```

## Advanced Features

### Message Metadata

```python
from langchain_core.messages import HumanMessage, AIMessage

# Add messages with metadata
history.add_message(
    HumanMessage(
        content="What's the capital of France?",
        additional_kwargs={"timestamp": "2024-01-15T10:30:00"}
    )
)

history.add_message(
    AIMessage(
        content="The capital of France is Paris.",
        additional_kwargs={"model": "gpt-4", "tokens": 12}
    )
)
```

### Async Operations

```python
# For async applications
async def handle_chat(user_id: str, message: str):
    history = CockroachDBChatMessageHistory(
        session_id=f"user-{user_id}",
        connection_string=connection_string,
    )
    
    # Add user message
    history.add_user_message(message)
    
    # Process with LLM (async)
    response = await llm.ainvoke(message)
    
    # Add AI response
    history.add_ai_message(response.content)
    
    return response.content
```

### Query History

```python
# Get recent messages from database
async with engine.engine.connect() as conn:
    from sqlalchemy import text
    
    result = await conn.execute(
        text("""
            SELECT session_id, message, created_at
            FROM message_store
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT 10
        """),
        {"session_id": "user-123"}
    )
    
    for row in result:
        print(f"{row.created_at}: {row.message}")
```

## Message Types

### Supported Types

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)

# Human message
history.add_message(HumanMessage(content="Hello"))

# AI message
history.add_message(AIMessage(content="Hi there!"))

# System message
history.add_message(SystemMessage(content="You are a helpful assistant"))

# Function/Tool messages
history.add_message(FunctionMessage(
    content="Weather: 72°F, Sunny",
    name="get_weather"
))
```

## Session Management

### List All Sessions

```python
async def list_sessions():
    """Get all active session IDs."""
    async with engine.engine.connect() as conn:
        from sqlalchemy import text
        
        result = await conn.execute(
            text("""
                SELECT DISTINCT session_id 
                FROM message_store
                ORDER BY MAX(created_at) DESC
            """)
        )
        
        return [row[0] for row in result]

sessions = await list_sessions()
print(f"Active sessions: {sessions}")
```

### Delete Old Sessions

```python
async def cleanup_old_sessions(days: int = 30):
    """Delete sessions older than N days."""
    async with engine.engine.connect() as conn:
        from sqlalchemy import text
        
        await conn.execute(
            text("""
                DELETE FROM message_store
                WHERE created_at < NOW() - INTERVAL ':days days'
            """),
            {"days": days}
        )
        await conn.commit()

await cleanup_old_sessions(30)
```

## Best Practices

### 1. Use Descriptive Session IDs

```python
# Good: Clear, hierarchical
session_id = f"{user_id}:{conversation_type}:{conversation_id}"
session_id = "user-123:support:ticket-456"

# Bad: Opaque
session_id = "abc123xyz"
```

### 2. Clean Up Old Sessions

```python
# Periodic cleanup
import asyncio

async def periodic_cleanup():
    while True:
        await cleanup_old_sessions(days=30)
        await asyncio.sleep(86400)  # Daily
```

### 3. Handle Errors Gracefully

```python
try:
    history.add_user_message(message)
except Exception as e:
    print(f"Failed to save message: {e}")
    # Continue without history
```

### 4. Use Connection Pooling

```python
# Reuse engine across sessions
from langchain_cockroachdb import CockroachDBEngine

engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=10,
)

# Pass engine connection string to history
history = CockroachDBChatMessageHistory(
    session_id="user-123",
    connection_string=connection_string,
)
```

## Troubleshooting

### Table Not Created

```python
# Explicitly create tables
await CockroachDBChatMessageHistory.acreate_tables(connection_string)
```

### Messages Not Persisting

Check connection:
```python
# Test connection
async with engine.engine.connect() as conn:
    from sqlalchemy import text
    result = await conn.execute(text("SELECT 1"))
    print(result.scalar())
```

### Slow Queries

Create index on session_id:
```sql
CREATE INDEX IF NOT EXISTS idx_session_created 
ON message_store (session_id, created_at DESC);
```

## Examples

### Complete Chatbot

```python
from langchain_cockroachdb import CockroachDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

def create_chatbot(user_id: str):
    """Create chatbot with persistent history."""
    history = CockroachDBChatMessageHistory(
        session_id=f"user-{user_id}",
        connection_string=connection_string,
    )
    
    memory = ConversationBufferMemory(
        chat_memory=history,
        return_messages=True,
    )
    
    return ConversationChain(
        llm=ChatOpenAI(model="gpt-4"),
        memory=memory,
    )

# Use chatbot
bot = create_chatbot("user-123")

# First conversation
response1 = bot.predict(input="My name is Alice")
# AI: "Nice to meet you, Alice!"

# Later conversation (history is loaded)
response2 = bot.predict(input="What's my name?")
# AI: "Your name is Alice."
```

### Multi-Turn Dialog

```python
async def handle_conversation(user_id: str, messages: list[str]):
    """Process multi-turn conversation."""
    history = CockroachDBChatMessageHistory(
        session_id=f"user-{user_id}",
        connection_string=connection_string,
    )
    
    llm = ChatOpenAI()
    
    for user_msg in messages:
        # Add user message
        history.add_user_message(user_msg)
        
        # Get context
        context = history.messages
        
        # Generate response
        response = await llm.ainvoke(context)
        
        # Save AI response
        history.add_ai_message(response.content)
        
        print(f"User: {user_msg}")
        print(f"AI: {response.content}\n")

# Use
await handle_conversation("user-123", [
    "Hello!",
    "What can you help me with?",
    "Tell me about CockroachDB",
])
```

## Next Steps

- [Vector Store](vector-store.md) - For RAG applications
- [Configuration](../getting-started/configuration.md) - Optimize performance
- [API Reference](../api/chat-history.md) - Full API documentation
