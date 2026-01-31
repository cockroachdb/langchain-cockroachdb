# Quick Start

Get started with langchain-cockroachdb in 5 minutes!

## Prerequisites

- Python 3.9+
- CockroachDB running (Cloud, Docker, or local)
- langchain-cockroachdb installed

## Basic Vector Store Example

```python
import asyncio
from langchain_cockroachdb import AsyncCockroachDBVectorStore, CockroachDBEngine
from langchain_core.embeddings import DeterministicFakeEmbedding

async def main():
    # 1. Create engine
    engine = CockroachDBEngine.from_connection_string(
        "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
    )
    
    # 2. Initialize table
    await engine.ainit_vectorstore_table(
        table_name="my_documents",
        vector_dimension=384,
        drop_if_exists=True,
    )
    
    # 3. Create vector store
    embeddings = DeterministicFakeEmbedding(size=384)
    vectorstore = AsyncCockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name="my_documents",
    )
    
    # 4. Add documents
    texts = [
        "CockroachDB is a distributed SQL database",
        "LangChain simplifies building LLM applications",
        "Vector search enables semantic similarity",
    ]
    ids = await vectorstore.aadd_texts(texts)
    print(f"Added {len(ids)} documents")
    
    # 5. Search
    results = await vectorstore.asimilarity_search(
        "Tell me about databases",
        k=2
    )
    
    for doc in results:
        print(f"- {doc.page_content}")
    
    # 6. Cleanup
    await engine.aclose()

asyncio.run(main())
```

## With Real Embeddings (OpenAI)

```python
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "your-key-here"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = AsyncCockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
)
```

## Synchronous Usage

For simple scripts:

```python
from langchain_cockroachdb import CockroachDBVectorStore

# Same API, but synchronous
vectorstore = CockroachDBVectorStore(
    engine=engine,
    embeddings=embeddings,
    collection_name="documents",
)

# Add documents (no await)
ids = vectorstore.add_texts(texts)

# Search (no await)
results = vectorstore.similarity_search("query", k=5)
```

## With Metadata

```python
texts = ["Doc 1", "Doc 2", "Doc 3"]
metadatas = [
    {"source": "web", "category": "tech"},
    {"source": "pdf", "category": "science"},
    {"source": "web", "category": "tech"},
]

await vectorstore.aadd_texts(texts, metadatas=metadatas)

# Filter by metadata
results = await vectorstore.asimilarity_search(
    "tech content",
    k=5,
    filter={"category": "tech"}
)
```

## Common Patterns

### 1. Document Loading

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

# Add to vector store
await vectorstore.aadd_documents(chunks)
```

### 2. Retrieval Chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
)

# Ask questions
answer = await qa.ainvoke("What is CockroachDB?")
print(answer)
```

### 3. Chat History

```python
from langchain_cockroachdb import CockroachDBChatMessageHistory

history = CockroachDBChatMessageHistory(
    session_id="user-123",
    connection_string="cockroachdb://...",
)

# Add messages
history.add_user_message("Hello!")
history.add_ai_message("Hi! How can I help?")

# Get history
messages = history.messages
```

## Next Steps

- [Configuration Guide](configuration.md) - Tune for your workload
- [Vector Indexes Guide](../guides/vector-indexes.md) - Optimize query performance
- [API Reference](../api/vectorstore.md) - Full API documentation
