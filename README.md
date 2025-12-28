# Semantic Markdown Search

A lightweight semantic search application for indexing and querying Markdown files. The project provides a command-line indexing workflow and a web-based UI for browsing indexed files and running semantic search queries. It is designed for local-first usage, fast iteration, and easy configurability.

## Overview

This project builds a semantic search engine over Markdown (`.md`) files using **Python** and **LangChain**. It supports:

- Incremental indexing of Markdown files from a configurable input directory
- Persistent storage of embeddings and metadata in a vector store on the host
- Automatic detection of new and modified files on re-index
- A web UI to:
  - View indexed files
  - Perform semantic search queries
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
- Supports configurable vector store backends supported by LangChain
- Metadata is stored alongside embeddings to enable file-level inspection

### Semantic Search

- Queries are embedded and matched against stored chunks
- Returns a configurable number of top-ranked chunks
- Results are ordered by semantic similarity score
- Each result includes its source file and content snippet

### Web Application

- Lists all indexed files and basic metadata
- Provides a search interface for semantic queries
- Displays ranked search results in a clean, minimal UI

## Technology Stack

- **Python**
- **LangChain** for:
  - Document loading
  - Chunking
  - Embeddings (configurable)
  - Vector store (configurable)
- **uv** for project management and dependency resolution

- **In-memory embeddings and vector store** for tests (fast and deterministic)

---

## Project Structure

```text
.
├── epistemon/
│   ├── indexing/        # Indexing logic
│   ├── search/          # Query and retrieval logic
│   ├── web/             # Web UI and API
│   │   └── main.py      # Web UI main.
│   └── config.py        # Configuration reader
├── config.yaml          # Configuration
├── tests/
│   ├── data/            # Data for tests, such as markdown files
│   └── ...              # Fast tests with in-memory stores
├── pyproject.toml
└── README.md
```

## Configuration

All major components are configurable from the YAML file, including:

- Input directory for Markdown files
- Embedding model
- Vector store backend and storage path
- Number of search results returned per query

Configuration is handled via the config.yaml.

## Commands

The project is built and managed using **uv**. Two primary commands are exposed via `pyproject.toml`.

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

## Demo

To quickly test the application with sample data:

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

To run E2E tests (requires Playwright browsers):

```bash
uv run playwright install chromium
uv run pytest tests -m e2e
```

E2E tests are skipped by default to keep the test suite fast. They are only executed when explicitly requested with `-m e2e`.

## Use Cases

- Experiment with different embedding models and vector stores.

## License

MIT License. See `LICENSE` for details.
