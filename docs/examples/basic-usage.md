# Basic Usage Examples

Core examples for getting started with langchain-cockroachdb.

## Quickstart

Complete example from zero to searching in 5 minutes.

**File:** [`examples/quickstart.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/quickstart.py)

```python
import asyncio
import os
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

async def main():
    # 1. Initialize engine
    engine = CockroachDBEngine.from_connection_string(
        os.getenv("COCKROACHDB_URL", 
                  "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable")
    )
    
    # 2. Create table
    await engine.ainit_vectorstore_table(
        table_name="quickstart",
        vector_dimension=384,
        drop_if_exists=True,
    )
    
    # 3. Initialize vector store
    embeddings = DeterministicFakeEmbedding(size=384)
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="quickstart",
    )
    
    # 4. Add documents
    texts = [
        "CockroachDB is a distributed SQL database",
        "LangChain makes building LLM apps easy",
        "Vector search enables semantic similarity",
    ]
    ids = await vectorstore.aadd_texts(texts)
    print(f"✅ Added {len(ids)} documents")
    
    # 5. Search
    results = await vectorstore.asimilarity_search(
        "Tell me about databases",
        k=2
    )
    
    print("\n🔍 Search Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
    
    # 6. Cleanup
    await engine.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**
```
✅ Added 3 documents

🔍 Search Results:
1. CockroachDB is a distributed SQL database
2. Vector search enables semantic similarity
```

## Sync Wrapper

Using the synchronous API for simple scripts.

**File:** [`examples/sync_usage.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/sync_usage.py)

```python
import os
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_cockroachdb import CockroachDBVectorStore, CockroachDBEngine

def main():
    # Initialize (no async/await needed)
    engine = CockroachDBEngine.from_connection_string(
        os.getenv("COCKROACHDB_URL")
    )
    
    engine.init_vectorstore_table(
        table_name="sync_demo",
        vector_dimension=384,
        drop_if_exists=True,
    )
    
    embeddings = DeterministicFakeEmbedding(size=384)
    vectorstore = CockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="sync_demo",
    )
    
    # Add documents (sync)
    texts = ["Document 1", "Document 2", "Document 3"]
    ids = vectorstore.add_texts(texts)
    print(f"Added {len(ids)} documents")
    
    # Search (sync)
    results = vectorstore.similarity_search("document", k=2)
    for doc in results:
        print(f"- {doc.page_content}")
    
    # Cleanup
    engine.close()

if __name__ == "__main__":
    main()
```

## Chat History

Persistent conversation storage.

**File:** [`examples/chat_history.py`](https://github.com/viragtripathi/langchain-cockroachdb/blob/main/examples/chat_history.py)

```python
import os
from langchain_cockroachdb import CockroachDBChatMessageHistory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

def main():
    # Initialize chat history
    history = CockroachDBChatMessageHistory(
        session_id="user-123",
        connection_string=os.getenv("COCKROACHDB_URL"),
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        chat_memory=history,
        return_messages=True,
    )
    
    # Create chatbot
    chatbot = ConversationChain(
        llm=ChatOpenAI(model="gpt-4"),
        memory=memory,
        verbose=True,
    )
    
    # First interaction
    response1 = chatbot.predict(input="My name is Alice")
    print(f"Bot: {response1}")
    
    # Later interaction (history is remembered)
    response2 = chatbot.predict(input="What's my name?")
    print(f"Bot: {response2}")
    
    # View history
    messages = history.messages
    print(f"\n📝 Chat history: {len(messages)} messages")

if __name__ == "__main__":
    main()
```

## With Real Embeddings

Using OpenAI embeddings instead of fake embeddings.

```python
import asyncio
import os
from langchain_openai import OpenAIEmbeddings
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine

async def main():
    engine = CockroachDBEngine.from_connection_string(
        os.getenv("COCKROACHDB_URL")
    )
    
    await engine.ainit_vectorstore_table(
        table_name="real_embeddings",
        vector_dimension=1536,  # OpenAI ada-002
        drop_if_exists=True,
    )
    
    # Use real embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="real_embeddings",
    )
    
    # Add documents with semantic meaning
    texts = [
        "The capital of France is Paris",
        "Paris is known for the Eiffel Tower",
        "Tokyo is the capital of Japan",
    ]
    await vectorstore.aadd_texts(texts)
    
    # Semantic search
    results = await vectorstore.asimilarity_search(
        "What is the capital city of France?",
        k=2
    )
    
    for doc in results:
        print(doc.page_content)
    
    await engine.aclose()

asyncio.run(main())
```

## With Metadata

Adding and filtering documents by metadata.

```python
async def metadata_example():
    # Add documents with metadata
    texts = [
        "CockroachDB documentation",
        "LangChain tutorial",
        "Vector search guide",
    ]
    metadatas = [
        {"category": "database", "source": "docs", "year": 2024},
        {"category": "ai", "source": "tutorial", "year": 2024},
        {"category": "ai", "source": "guide", "year": 2023},
    ]
    
    await vectorstore.aadd_texts(texts, metadatas=metadatas)
    
    # Filter by metadata
    results = await vectorstore.asimilarity_search(
        "AI content",
        k=5,
        filter={"category": "ai"}
    )
    
    print("AI documents:")
    for doc in results:
        print(f"- {doc.page_content} ({doc.metadata['year']})")
```

## Next Steps

- [Advanced Filtering](advanced-filtering.md) - Complex queries
- [Index Optimization](index-optimization.md) - Performance tuning
- [Guides](../guides/vector-store.md) - Detailed guides
