# Epistemon Implementation TODO

This file tracks the implementation progress of the Semantic Markdown Search application.

## Project Status

- [x] Initial project setup with uv, git, and pytest
- [x] Configuration module with YAML loading and validation
- [x] Core indexing functionality with incremental updates
- [x] Search API with FastAPI endpoints
- [x] Web UI migration to Shiny
- [x] CLI commands for indexing and web server
- [ ] Production ready

## Phase 1: Project Setup and Configuration

Completed project setup with uv package manager, LangChain dependencies (langchain, langchain-community, langchain-text-splitters), embedding models (FakeEmbeddings for testing, HuggingFaceEmbeddings for integration tests, OpenAIEmbeddings for production), vector stores (InMemoryVectorStore for testing, Chroma for production), web framework (FastAPI, uvicorn), YAML configuration (pyyaml), and development tools (black, ruff, mypy, pytest, pip-audit).

Created package structure: epistemon/{indexing,search,web}/, tests/data/, and config.yaml template.

## Phase 2: Configuration Module (TDD)

Implemented immutable Configuration dataclass with YAML loading, sensible defaults for all fields (input_directory, embedding_provider, embedding_model, vector_store_type, vector_store_path, chunk_size, chunk_overlap, search_results_limit), enum validation, type validation, value constraints, and comprehensive error handling.

## Phase 3: Walking Skeleton (End-to-End Spike)

Built end-to-end proof of concept: markdown loading and chunking, embedding and indexing with FakeEmbeddings in InMemoryVectorStore, similarity search, FastAPI GET /search endpoint, and minimal HTML/JavaScript UI. Validated complete workflow from indexing to search via browser.

## Phase 4: Indexing Module (TDD)

Implemented comprehensive indexing system with recursive directory scanning, file modification tracking, incremental updates (detect new/modified/deleted files), configurable embeddings (FakeEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings), vector store persistence, MarkdownTextSplitter for semantic chunking with smart headline handling (prevents dangling headlines), and unified index() API.

Performance achieved with FakeEmbeddings: 0.23ms per file for new indexing (43x better than 10ms target), 0.07ms per file for re-indexing unchanged files (142x better than target). Created instrumentation module for performance monitoring.

## Phase 5: Search API (TDD)

Implemented FastAPI GET /search endpoint with VectorStoreRetriever, configurable result limits, score-based ranking, empty query handling, score threshold filtering with alerts, source file links with URL encoding, automatic metric type detection (distance vs similarity), and comprehensive metadata inclusion (content snippets, similarity scores, last modified timestamps). Added GET /files/{path} endpoint to serve markdown files as formatted HTML.

## Phase 6: Web UI Module (TDD) - COMPLETE

Migrated from vanilla HTML/JavaScript to Shiny for Python to enable future side-by-side RAG comparison features.

**Completed:**

- [x] Refactored to VectorStore interface (removed VectorStoreRetriever indirection)
- [x] Implemented FastAPI endpoints: GET /search, GET /files, GET /files/{path}
- [x] Added VectorStoreManager abstraction for cross-store file listing (InMemory, Chroma)
- [x] Comprehensive error handling with consistent JSON responses (400, 404, 422, 500)
- [x] Full Shiny UI implementation with reactive search, metric detection, result cards
- [x] Score threshold filtering, source links, metadata display (timestamps)
- [x] Root path (/) redirects to Shiny app at /app/
- [x] Removed legacy static HTML UI

**Features:**

- Search input with configurable result limit
- Automatic metric type detection (Distance vs Similarity)
- Score badges with metric labels (e.g., "Similarity: 0.8532")
- Clickable source links, formatted timestamps
- Empty query validation, no-results handling

**Remaining:**

- [ ] Document architecture plan for side-by-side RAG comparison in ROADMAP.md
- [ ] Manual UI testing (search functionality, display elements, API validation)

## Phase 7: CLI Commands (TDD) - COMPLETE

Implemented two CLI entry points for production use.

**7.1 upsert-index Command:**

- [x] Loads configuration, creates vector store, triggers incremental indexing
- [x] Progress logging and comprehensive error handling
- Usage: `uv run upsert-index [--config CONFIG]`

**7.2 web-ui Command:**

- [x] Starts FastAPI + Shiny web server with uvicorn
- [x] Configurable host, port, and config file
- [x] Natural shutdown handling (Ctrl+C)
- Usage: `uv run web-ui [--config CONFIG] [--host HOST] [--port PORT]`

## Phase 8: Integration and End-to-End Testing

- [ ] Write test for full indexing workflow (config -> scan -> chunk -> embed -> store)
- [ ] Write test for web API integration with search module
- [ ] Ensure all modules integrate correctly
- [ ] Document configuration options in README

## Phase 9: Documentation

- [ ] Add docstrings to all public functions and classes
- [ ] Update README with installation instructions
- [ ] Update README with usage examples
- [ ] Create example workflow documentation

## Phase 10: Polish and Release

- [ ] Test upsert-index command with real markdown files
- [ ] Test web-ui command and verify UI in browser
- [ ] Test incremental indexing by modifying files
- [ ] Test search quality with various queries
- [ ] Test error handling for missing config, invalid paths, etc.
- [ ] Profile indexing performance with large markdown collections
- [ ] Optimize chunking and search if needed
- [ ] Ensure reasonable memory usage
- [ ] Update version in pyproject.toml
- [ ] Add LICENSE file
- [ ] Final README review
- [ ] Tag release in git
- [ ] Update this TODO with final status
