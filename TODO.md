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

## Phase 6: Web UI Module (TDD)

### Core Refactoring

#### 6.1 Refactor to VectorStore Interface (TDD Cycle 1)

- [x] Write test verifying create_app() works with VectorStore instead of VectorStoreRetriever
- [x] Update create_app() signature to accept vector_store: VectorStore parameter
- [x] Update /search endpoint to use vector_store.similarity_search_with_score() directly
- [x] Remove retriever.vectorstore indirection
- [x] Update all existing tests to pass VectorStore instances

Note: VectorStoreRetriever doesn't expose similarity_search_with_score(), which is needed for score display. Using VectorStore directly is cleaner and more flexible.

### FastAPI Endpoints (Remaining Work)

#### 6.2 GET /files File Listing

- [x] Write test for the GET /files endpoint to return a list of all indexed files
- [x] Write test to check the presence of all file metadata in response
- [x] Implement file list retrieval from vector store with metadata
- [x] Write test for empty index handling (no files in store)
- [x] Implement empty index response handling
- [x] Write test for sorting files by name or date
- [x] Implement file sorting and response serialization

Note: File list retrieval implemented by iterating through InMemoryVectorStore.store dictionary.

#### 6.2.1 Vector Store Manager Abstraction (Refactoring)

- [x] Write test for VectorStoreManager.get_indexed_files() with InMemoryVectorStore
- [x] Implement VectorStoreManager class with get_indexed_files() method
- [x] Write test for VectorStoreManager.get_indexed_files() with Chroma
- [x] Implement Chroma support in get_indexed_files()
- [x] Refactor GET /files endpoint to use VectorStoreManager
- [x] Update all tests to use VectorStoreManager where appropriate

Note: VectorStoreManager abstraction now allows get_indexed_files() to work across different vector store implementations. The web API maintains backward compatibility while supporting the new manager pattern.

#### 6.3 API Error Handling

- [x] Write test for error handling in endpoints
- [x] Implement comprehensive error handling and error responses

Note: Comprehensive error handling implemented for all endpoints:

- GET /search: validates parameters (422), handles vector store errors (500)
- GET /files: validates sort_by parameter (400), handles vector store errors (500)
- GET /files/{path}: returns 404/403 for not found/access denied, handles file read errors (500)
  All error responses follow consistent JSON format with "error" and "detail" fields.

### Shiny UI Migration

Goal: Replace vanilla HTML/JavaScript UI with Shiny for Python to enable future side-by-side RAG comparison features while maintaining FastAPI endpoints for programmatic access.

#### 6.4 Setup and Dependencies (TDD Cycle 2)

- [x] Add shiny to dependencies via uv
- [x] Verify shiny installation

#### 6.5 Basic Shiny App Structure (TDD Cycle 3)

- [x] Write test for create_shiny_app() factory function
- [x] Implement epistemon/web/shiny_app.py with app structure
- [x] Create app_ui with page layout and sidebar
- [x] Create server function skeleton
- [x] Return App instance

#### 6.6 Search Input Components (TDD Cycle 4)

- [x] Write test for search input widgets
- [x] Implement ui.input_text for query
- [x] Implement ui.input_numeric for result limit
- [x] Implement ui.input_action_button for search trigger

#### 6.7 Reactive Search Execution (TDD Cycle 5)

- [x] Write test for search execution on button click
- [x] Implement @render.ui with @reactive.event(input.search)
- [x] Call vector_store.similarity_search_with_score()
- [x] Verify reactivity works

#### 6.8 Empty Query Validation (TDD Cycle 6)

- [x] Write test for empty query handling
- [x] Implement validation for empty/whitespace queries
- [x] Return warning alert for invalid input

#### 6.9 Metric Type Detection (TDD Cycle 7)

- [x] Write test for metric type detection (distance vs similarity)
- [x] Implement score ordering analysis
- [x] Set appropriate metric_type label

#### 6.10 Result Card Display (TDD Cycle 8)

- [x] Write test for rendering results as cards
- [x] Implement ui.card for each result
- [x] Include result number, score badge, content, source, timestamp
- [x] Return ui.TagList of cards

#### 6.11 Score Display (TDD Cycle 9)

- [x] Write test for score formatting and display
- [x] Implement score badge with metric label
- [x] Format to 4 decimal places

#### 6.12 Source Links (TDD Cycle 10)

- [x] Write test for source file links
- [x] Implement conditional link creation with base_url
- [x] Use quote() for URL encoding
- [x] Set target="\_blank" for new tab

#### 6.13 Score Threshold Filtering (TDD Cycle 11)

- [x] Write test for score_threshold filtering
- [x] Implement result filtering logic
- [x] Display alert for filtered results count

#### 6.14 No Results Handling (TDD Cycle 12)

- [x] Write test for zero results scenario
- [x] Return info alert when no results found

#### 6.15 Metadata Display (TDD Cycle 13)

- [x] Write test for timestamp display
- [x] Format last_modified timestamps
- [x] Handle missing metadata gracefully

#### 6.16 FastAPI Integration (TDD Cycle 14)

- [x] Write test for Shiny app mounting
- [x] Update create_app() to accept optional mount_shiny parameter
- [x] Mount Shiny app at /app when enabled
- [x] Keep /search and /files APIs at root
- [x] Verify both UI and API endpoints work

#### 6.17 Root Path Handling (TDD Cycle 15)

- [x] Replace the current HTML UI at GET / with the Shiny app (always).
- [x] Remove static/index.html and anything else that is left from the original UI.

#### 6.18 Future RAG Comparison Preparation (TDD Cycle 16)

- [ ] Document architecture plan for side-by-side comparison in ROADMAP.md

#### 6.19 Manual UI Testing

- [ ] Test search functionality via Shiny UI
- [ ] Verify all display elements work correctly
- [ ] Compare behavior with original HTML/JS UI
- [ ] Validate API endpoints remain functional
- [ ] Test /app and root redirect

## Phase 7: CLI Commands (TDD) - COMPLETE

### 7.1 upsert-index Command

- [x] Write test for upsert-index command execution
- [x] Create epistemon/cli.py module and implement index_command function
- [x] Implement configuration loading
- [x] Implement indexing trigger logic
- [x] Add progress logging
- [x] Add error handling and user messages
- [x] Configure upsert-index script in pyproject.toml

### 7.2 web-ui Command

- [x] Write test for web-ui command execution
- [x] Implement server startup code (similar to demo.py)
- [x] Add configuration loading
- [x] Add shutdown handling
- [x] Configure web-ui script in pyproject.toml

Note: Implemented `web_ui_command()` function that loads configuration, creates vector store, creates FastAPI app with Shiny UI, and runs uvicorn server. Created `web_ui_main()` entry point with argument parsing for --config, --host, and --port options. Shutdown is handled naturally by uvicorn (Ctrl+C).

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
