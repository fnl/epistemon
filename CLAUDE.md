## Notes

- Refer to the AGENTS.md file to understand how to track work and understand the current status of project with `bd` (Beads).
- When working on a Beads task, you are expected to follow a TDD-based approach. Don't forget to set the `bd` ticket to "In Progress".
- Remember that you have a HTTP MCP servier docs-langchain to refer to LangChain documentation.
- The project is about implementing a beefed up semantic search engine example described in https://docs.langchain.com/oss/python/langchain/knowledge-base
- Add explanations and insights about the project in CLAUDE.md here, the README.md, or the ROADMAP.md.
- When you have a list of planned steps, write one test, make it green, refactor, and git commmit the result. Only then move on to the next test. Never plan writing more than one testat a time. Refer to the testing rules.
- There is no need to run black, ruff, and mypy manually in this project after every step, because they are run by the pre-commit hook.

## LangChain Architecture Pattern

Following the LangChain knowledge base tutorial architecture:
Document -> Chunks -> Embeddings -> Vector Store -> Retriever -> Results

Key components:

- MarkdownTextSplitter
- Document objects with chunked content, metadata, and source
- Vector store retriever with as_retriever() method
- Support for similarity search and score threshold filtering

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
- Requires langchain-openai and API key

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
- Open for extension to more stores

## BM25 Keyword Search Architecture

Parallel search strategy for comparing keyword-based and semantic approaches:

**In-Memory Rebuild Strategy**

- BM25Indexer rebuilds index from disk on startup
- No persistent BM25 index storage
- Fast startup for small-to-medium document sets
- Uses rank_bm25 library (BM25Okapi algorithm)
- Simple tokenization with .split() for initial implementation

**Integration with Vector Store**

- BM25 and semantic search operate independently
- Both use same Document objects and chunking strategy
- Same markdown files feed both indexes
- BM25 indexer loads documents via load_and_chunk_markdown()
- Vector store and BM25 index exist side-by-side

**Comparison with Semantic Search**

BM25 (Keyword):
- Exact term matching with statistical ranking
- No ML model or embeddings needed
- Fast query execution (no embedding step)
- Scores based on term frequency and inverse document frequency
- Works well for queries with specific technical terms

Semantic (Embeddings):
- Understands meaning and context
- Finds conceptually similar content
- Requires embedding model (OpenAI, HuggingFace, etc.)
- Higher computational cost
- Better for natural language queries

**Implementation Patterns**

Configuration:
- Enable BM25 via config.yaml bm25.enabled flag
- Configure chunk size/overlap same as vector store
- Optional parameter in create_shiny_app()

Code structure:
- epistemon/indexing/bm25_indexer.py - BM25Indexer class
- epistemon/web/shiny_ui.py - Dual render functions (bm25_results, results)
- Side-by-side two-column layout for direct comparison
- Independent error handling per search type
- Distinct badge colors (bg-info for BM25, bg-success/primary for semantic)

## Web UI

We will use Shiny for Python as the UI:
https://shiny.posit.co/py/docs/overview.html
Refer to those documents to undersstand how to use Shiny.
