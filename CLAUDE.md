## Notes

- Refer to the TODO.md file to understand the current status of the development process.
- Remember that you have a HTTP MCP servier docs-langchain to refer to LangChain documentation.
- The project is about implementing a beefed up semantic search engine example described in https://docs.langchain.com/oss/python/langchain/knowledge-base
- Don't add more to the TODO.md file than tasks; Add explanations in CLAUDE.md here, or in other supporting markdown files.
- Once you have completed your current tasks according to the TODO.md file, and before committing changes to git, update the status of the TODO items to reflect on what has been done.
- When you have a list of planned steps, write one test, make it green, refactor, and git commmit the result. Only then move on to the next test. Never plan writing more than one testat a time. Refer to the testing rules.
- There is no need to run black, ruff, and mypy manually in this project, because they now are run by the pre-commit hook.

## LangChain Architecture Pattern

Following the LangChain knowledge base tutorial architecture:
Document -> Chunks -> Embeddings -> Vector Store -> Retriever -> Results

Key components:

- RecursiveCharacterTextSplitter (chunk_size=999, overlap=200, add_start_index=True)
- Document objects with page_content, metadata, and optional id
- Vector store retriever with as_retriever() method
- Support for similarity search, MMR, and score threshold filtering

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
