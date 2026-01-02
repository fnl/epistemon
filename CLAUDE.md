## Notes

- Refer to the AGENTS.md file to understand how to track work and understand the current status of project with `bd`.
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

## Web UI

We will use Shiny for Python as the UI:
https://shiny.posit.co/py/docs/overview.html
Refer to those documents to undersstand how to use Shiny.
