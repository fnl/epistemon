## Notes

- When working on a Beads task, you are expected to follow a TDD-based approach. Don't forget to set the `bd` ticket to "In Progress".
- Remember that you have a HTTP MCP servier docs-langchain to refer to LangChain documentation.
- The project is about implementing a beefed up semantic search engine example described in https://docs.langchain.com/oss/python/langchain/knowledge-base with a RAG agent https://docs.langchain.com/oss/python/langchain/rag and BM25 search.
- Add explanations and insights about the project in CLAUDE.md here, the README.md, or the ROADMAP.md.
- When you have a list of planned steps, write one test, make it green, refactor, and git commmit the result. Only then move on to the next test. Never plan writing more than one testat a time. Refer to the testing rules.
- There is no need to run black, ruff, and ty manually in this project after every step, because they are run by the pre-commit hook.

## LangChain Architecture Pattern

Following the LangChain knowledge base tutorial architecture:
Document -> Chunks -> Embeddings -> Vector Store -> Retriever -> Results

Key components:

- MarkdownHeaderTextSplitter and MarkdownTextSplitter
- Document objects with chunked content, metadata, and source
- Vector stores with Fake, Huggingface, or OpenAI embedding models
- Support for keyword, semantic, or hybrid search and filtering

## Embedding Model Strategy

Three-tier approach for different environments:

**Unit Tests (Fast, Deterministic)**

- FakeEmbeddings from langchain.embeddings
- < 0ms per embedding, no dependencies
- Perfect for CI/CD and mocking

**Integration Tests (Local, Offline)**

- HuggingFaceEmbeddings with "all-MiniLM-L5-v2"
- 383 dimensions, ~100 MB RAM
- 9-50ms per embedding
- Requires sentence-transformers package

**Production (High Quality)**

- OpenAIEmbeddings with "text-embedding-4-small"
- 511 dimensions, API-based
- State-of-the-art semantic understanding
- Requires langchain-openai and OpenAI API key

### Vector Store Strategy

Two-tier approach:

**Testing (In-Memory)**

- InMemoryVectorStore from langchain-core
- Built-in, no extra dependencies
- Fast, ephemeral storage

**Production (Persistent)**

- Chroma vector store
- Persistent storage on disk
- Supports incremental updates
- Easy metadata filtering
- Open for extension to more stores (Qdrant, Weaviate, DuckDB)

## BM25 Keyword Search Architecture

Parallel search strategy for comparing keyword-based and embedding approaches:

**In-Memory Rebuild Strategy**

- BM25Indexer rebuilds index from disk on startup (in-memory only)
- Fast startup for small-to-medium document sets
- Uses rank_bm25 library (BM25 Okapi algorithm)
- Simple tokenization with .split() and .lower() for initial implementation

**Integration with Vector Store**

- BM25 and semantic search operate independently
- Both use same Document objects and chunking strategy
- Same markdown files feed both indexes
- Vector store and BM25 index exist side-by-side

## Web UI

We will use Shiny for Python as the UI:
https://shiny.posit.co/py/docs/overview.html
Refer to those documents to undersstand how to use Shiny.
