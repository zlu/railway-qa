# Wiseman QA System: User Guide

This guide addresses common questions and issues when using the Wiseman QA system.

---

## FAQ

### 1. How does the system know which ChromaDB to use and how does the LLM interact with it?

#### How does the code know whether to use ChromaDB and which ChromaDB to use?

The system determines which ChromaDB to use based on how you create the `vector_store` object:

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    persist_directory="sample_chroma_db",  # This specifies the DB location
    collection_name="sample_document_embeddings"
)
```

- The `persist_directory` argument tells Chroma where to store and later retrieve the database files.
- When you call `vector_store.as_retriever(...)`, it uses the ChromaDB instance you created with the specified directory and collection.

**To use a different ChromaDB, change the `persist_directory` path.**

**To load an existing DB:**
```python
vector_store = Chroma(
    persist_directory="sample_chroma_db",
    embedding_function=embed_model,
    collection_name="sample_document_embeddings"
)
```

#### How does the LLM interact with information stored in ChromaDB?

- **ChromaDB** is used for **retrieval**: it stores document embeddings and metadata.
- When you ask a question, the retriever finds the most relevant document chunks using vector similarity search.
- These retrieved chunks are then **passed as context** to the LLM via the `RetrievalQA` chain.
- The LLM does **not** directly access ChromaDB; it only sees the text chunks retrieved by the retriever and uses them to generate an answer.

**Summary:**
- ChromaDB = search engine for relevant text chunks
- LLM = language model that reads those chunks and answers your question

#### What are the files in the ChromaDB directory?

- `chroma.sqlite3`: Stores metadata, document info, and collection structure
- Files with long names (hashes/UUIDs): Store the actual embedding vectors and binary data for fast retrieval

**Do not delete or modify these files manually**—they are required for Chroma to function correctly.

---

### 2. Memory Issues: Retrieval never finishes on low-RAM machines

#### Is this a memory constraint?

**Yes, very likely.** On machines with limited RAM (like 8GB MacBooks), retrieval may not finish due to memory constraints:

- **Embeddings in RAM**: Each chunk gets embedded into a high-dimensional vector (768, 1024, or more floats per chunk)
- **All embeddings are loaded into RAM** for fast similarity search
- **Memory calculation**: Total chunks × embedding size × 4 bytes (float32) can easily exceed available memory
- **When RAM is exhausted**: The process may slow to a crawl (swapping) or never finish

#### Would changing chunk_size help?

**Yes, changing `chunk_size` can significantly reduce memory usage:**

- **Larger `chunk_size`** → Fewer chunks (each chunk is bigger, so fewer are created)
- **Fewer chunks** → Fewer embeddings to store in memory
- **Smaller `chunk_size`** → More chunks, more embeddings, higher memory usage

**Example:**
```python
# For low-memory systems, use larger chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

If you double the `chunk_size`, you roughly halve the number of chunks (and thus embeddings), which can make a big difference for memory usage.

#### Other tips for low-memory machines:

1. **Increase `chunk_size`** (e.g., from 1000 to 2000 or 3000)
2. **Reduce `chunk_overlap`** (less overlap = fewer total chunks)
3. **Index fewer documents at a time** (split your KB into smaller batches)
4. **Use a smaller embedding model** (if possible)
5. **Try FAISS** (sometimes has lower memory overhead than Chroma)

#### Memory Usage Summary:

| Setting | Effect on Memory Usage |
|---------|----------------------|
| Larger chunk_size | ↓ Fewer chunks, ↓ memory usage |
| Smaller chunk_size | ↑ More chunks, ↑ memory usage |
| Fewer documents | ↓ memory usage |

**Recommendation for 8GB MacBooks:** Start with `chunk_size=3000` and adjust based on your specific documents and available memory.

---

## Quick Reference

### ChromaDB Files
- `chroma.sqlite3`: Metadata and structure
- Long-named files: Embedding vectors (don't delete!)

### Memory Optimization
- Use larger chunk sizes for low-RAM systems
- Reduce chunk overlap
- Process documents in smaller batches
- Consider FAISS as an alternative to Chroma

### LLM Integration
- LLM receives retrieved chunks as context
- LLM does not directly access the vector database
- Retrieval and generation are separate steps in the pipeline 