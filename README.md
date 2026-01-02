# Semantic Markdown Search

A lightweight semantic search application for indexing and retrieving information in Markdown files. The project provides a command-line indexing workflow and a web-based UI for researching the indexed content. It is designed for local-first usage, fast iteration, and easy configurability.

## Use Cases

- Visualize different retrieval strategies and their impact.
- Experiment with different embedding models and vector stores.
- Explore the effect of chunking strategies on semantic search results.
- Make local/personal content more readily available to an AI assistant.

## Overview

This project builds a semantic search engine over Markdown (`.md`) files using **Python**. It supports:

- Incremental indexing of Markdown files from a configurable input directory
- Persistent storage of embeddings and metadata in a vector store
- BM25 keyword-based indexing for traditional search
- Automatic detection of new and modified files on re-index
- A search API endpoint to query for document chunks
- A web UI to:
  - View indexed files
  - Compare search strategies: keyword (BM25), semantic (embeddings), and RAG-generated answers
  - Inspect ranked search results by relevance

## Key Features

### Incremental Indexing

- Scans a configurable input directory for Markdown files
- Splits files into semantic chunks
- Stores embeddings along with:
  - Source file name
  - Last modified timestamp
- On re-run:
  - New files are added
  - Modified files are re-chunked and re-indexed
  - Unchanged files are skipped

### Persistent Vector Store

- Embeddings are stored in a persistent vector database on the host
- Supports configurable vector store backends (Chroma, Weaviate, DuckDB, Qdrant)
- Metadata is stored alongside embeddings to enable provenance tracking and updates

### Search Strategy Comparison

The application implements three different retrieval approaches for side-by-side comparison:

**Keyword Search (BM25)**
- Traditional term-frequency ranking
- Excellent for exact term matching and known terminology
- Fast and deterministic

**Semantic Search (Embeddings)**
- Queries are embedded and matched against stored chunks
- Returns top-ranked chunks ordered by semantic similarity score
- Understands conceptual relationships beyond keyword overlap
- Each result includes source file and content snippet

**RAG Answer Generation**
- Retrieves relevant chunks using semantic search
- Generates natural language answers from retrieved context
- Two-step process: retrieve then generate
- Provides synthesized responses instead of raw chunks

The web UI displays results from all three strategies, allowing direct comparison of what each approach surfaces and how they differ in precision, recall, and usefulness.

### Web Application

- Lists all indexed files and basic metadata
- Provides a search interface for semantic queries
- Displays ranked search results in a clean, minimal UI

## Technology Stack

- **Python**
- **LangChain** for:
  - Document loading
  - Chunking
  - Embeddings (configurable: Huggingface or OpenAI)
  - Vector store (in memory, Chroma, Weaviate, Qdrant, or DuckDB)
  - BM25 retrieval
  - RAG chain composition
- **uv** for project management and dependency resolution
- **Shiny** as a reactive UI

---

## Project Structure

```text
.
├── epistemon/           # Source code
│   ├── indexing/        # Indexing logic
│   ├── web/             # Web UI and API
│   │   └── app.py       # Web UI main.
│   └── config.py        # Configuration reader
├── config.yaml          # Optional configuration
├── tests/               # Unit and a few e2e tests
│   ├── data/            # Data for tests, such as markdown files
├── pyproject.toml       # Project setup
└── *.md                 # Docs for humans and code assistants
```

## Configuration

All major components are configurable from the YAML file, including:

- Input directory for Markdown files
- Embedding model
- Vector store backend and storage path
- Chunk size and overlap settings
- Number of search results returned per query

Configuration is handled via the config.yaml.
If none is provided, the defaults are used.

### Important: Changing Chunk Size Settings

**If you change `chunk_size` or `chunk_overlap` settings**, you must delete the existing vector store database and re-index from scratch:

```bash
rm -rf ./data/chroma_db
uv run upsert-index  # or python demo.py
```

This is necessary because the incremental indexing system tracks file modification times, not configuration changes. Files that haven't been modified won't be re-chunked automatically, leaving old chunks with the previous chunk size in the database.

## Commands/Usage

The project is built and managed using **uv**. Two primary commands are exposed via `pyproject.toml`.
To set up the dependencies according to the latest lock-file:

```bash
uv sync
```

### Indexing

Indexes or updates the vector store based on the current state of the input directory.

```bash
uv run upsert-index
```

This command:

- Loads Markdown files
- Detects new or modified files
- Updates the vector store accordingly

### Start Web UI

Launches the web application for browsing and searching the index.

```bash
uv run web-ui
```

Once running, the web UI allows users to:

- View the list of indexed files
- Submit semantic search queries
- Explore ranked search result chunks

### Demo

To quickly test-run the application with boring test documents:

```bash
uv run python demo.py
```

This starts a web server at http://localhost:8000 with the sample markdown file pre-indexed. Try searching for terms like "LangChain", "embeddings", or "vector stores".

## Testing

The test suite includes unit tests, integration tests, and end-to-end tests:

- Unit and integration tests use FakeEmbeddings and InMemoryVectorStore for fast, deterministic execution
- End-to-end tests use Playwright for headless browser testing (slow, skipped by default)

To run unit and integration tests (fast):

```bash
uv run pytest tests
```

To run E2E tests install the Playwright Chromium browser:

```bash
uv run playwright install chromium
```

And run those **slow** tests like the Weaviate vector store tests, E2E tests, or performance tests:

```bash
uv run pytest tests -m slow
```

E2E tests are skipped by default to keep the test suite fast. They are only executed when explicitly requested with `-m e2e`.

## License

MIT License. See `LICENSE` for details.
