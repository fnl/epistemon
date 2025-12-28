# Epistemon Implementation TODO

This file tracks the implementation progress of the Semantic Markdown Search application.

## Project Status

- [x] Initial project setup with uv, git, and pytest
- [ ] Core functionality implemented
- [ ] Web UI implemented
- [ ] Production ready

## Phase 1: Project Setup and Configuration

### 1.1 Dependencies and Tooling

- [x] Add LangChain dependencies (langchain, langchain-community, langchain-text-splitters)
- [x] Add embedding model dependencies:
  - For testing: FakeEmbeddings (built into langchain, no extra deps)
  - For integration tests: sentence-transformers (HuggingFaceEmbeddings with all-MiniLM-L6-v2)
  - For production: langchain-openai (OpenAIEmbeddings with text-embedding-3-small)
- [x] Add vector store dependencies:
  - For testing: InMemoryVectorStore (built into langchain-core, no extra deps)
  - For production: chromadb (persistent vector store)
- [x] Add document loader dependencies (pypdf for PDF support, though focus is markdown)
- [x] Add web framework dependencies (fastapi, uvicorn)
- [x] Add YAML configuration dependencies (pyyaml)
- [x] Add development tools (black, ruff, mypy, pytest, pip-audit)
- [x] Configure ruff rules in pyproject.toml
- [x] Configure mypy strict mode in pyproject.toml

### 1.2 Project Structure

- [x] Create epistemon/ package directory
- [x] Create epistemon/__init__.py
- [x] Create epistemon/indexing/ module directory
- [x] Create epistemon/search/ module directory
- [x] Create epistemon/web/ module directory
- [x] Create tests/data/ directory for test markdown files
- [x] Create config.yaml template file

## Phase 2: Configuration Module (TDD)

### 2.1 Basic Configuration Loading (TDD Cycle 1)

- [x] Write test for basic YAML configuration loading
- [x] Implement Configuration data class with type hints
- [x] Implement YAML file loader to make test pass
- [x] Commit green test

### 2.2 Default Values for All Fields (TDD Cycle 2)

All fields have sensible defaults (config file is optional):
- input_directory: Default "./tests/data"
- embedding_provider: Default "huggingface"
- embedding_model: Default "all-MiniLM-L6-v2"
- vector_store_type: Default "chroma"
- vector_store_path: Default "./data/chroma_db"
- chunk_size: Default 1000 (LangChain best practice)
- chunk_overlap: Default 200 (LangChain best practice)
- search_results_limit: Default 5

Tasks:
- [ ] Write test for missing config file (uses all defaults)
- [ ] Write test for empty config file (uses all defaults)
- [ ] Write test for partial config (some fields overridden)
- [ ] Implement default value handling
- [ ] Make tests pass
- [ ] Commit green tests

### 2.3 Field Override Validation (TDD Cycle 3)

Ensure individual field overrides work correctly:
- [ ] Write test for overriding input_directory only
- [ ] Write test for overriding embedding config only
- [ ] Write test for overriding vector store config only
- [ ] Verify implementation handles overrides
- [ ] Make tests pass
- [ ] Commit green tests

### 2.4 Invalid Configuration Handling (TDD Cycle 4)

- [ ] Write test for invalid YAML syntax
- [ ] Implement error handling for invalid YAML
- [ ] Make test pass
- [ ] Commit green test

### 2.5 File Not Found Handling (TDD Cycle 5)

- [ ] Write test for missing configuration file
- [ ] Implement error handling for missing files
- [ ] Make test pass
- [ ] Commit green test

### 2.6 Enum Validation (TDD Cycle 6)

- [ ] Write test for embedding_provider validation
- [ ] Write test for vector_store_type validation
- [ ] Implement enum validation
- [ ] Make tests pass
- [ ] Commit green tests

### 2.7 Value Constraints (TDD Cycle 7)

- [ ] Write test for positive integer validation
- [ ] Implement constraints validation
- [ ] Make test pass
- [ ] Commit green test

### 2.8 Immutability (TDD Cycle 8)

- [ ] Write test for Configuration immutability
- [ ] Ensure frozen dataclass implementation
- [ ] Make test pass
- [ ] Commit green test

## Phase 3: Indexing Module (TDD)

### 3.1 File Discovery Tests

- [ ] Write test for scanning directory for markdown files
- [ ] Write test for recursive directory traversal
- [ ] Write test for filtering non-markdown files
- [ ] Write test for tracking file modification times

### 3.2 File Discovery Implementation

- [ ] Implement markdown file scanner
- [ ] Implement file modification time tracking
- [ ] Ensure all file discovery tests pass

### 3.3 Document Processing Tests

- [ ] Write test for loading markdown file content
- [ ] Write test for chunking document text
- [ ] Write test for preserving metadata (filename, timestamp)
- [ ] Write test for handling empty files
- [ ] Write test for handling malformed markdown

### 3.4 Document Processing Implementation

- [ ] Implement markdown document loader
- [ ] Implement text splitter with configurable chunk size
- [ ] Implement metadata extraction and attachment
- [ ] Ensure all document processing tests pass

### 3.5 Incremental Indexing Tests

- [ ] Write test for detecting new files
- [ ] Write test for detecting modified files
- [ ] Write test for skipping unchanged files
- [ ] Write test for removing deleted file embeddings
- [ ] Write test for full index rebuild

### 3.6 Incremental Indexing Implementation

- [ ] Implement file state tracking (hash or timestamp based)
- [ ] Implement new file detection logic
- [ ] Implement modified file detection logic
- [ ] Implement embedding update logic
- [ ] Implement deletion handling
- [ ] Ensure all incremental indexing tests pass

### 3.7 Vector Store Integration Tests

- [ ] Write test for creating embeddings from chunks
- [ ] Write test for storing embeddings in vector store
- [ ] Write test for updating existing embeddings
- [ ] Write test for vector store persistence
- [ ] Write test with in-memory vector store (for fast tests)

### 3.8 Vector Store Integration Implementation

- [ ] Implement embedding generation interface
- [ ] Implement vector store initialization
- [ ] Implement embedding storage logic
- [ ] Implement embedding update logic
- [ ] Implement persistence handling
- [ ] Ensure all vector store tests pass

## Phase 4: Search Module (TDD)

### 4.1 Query Processing Tests

- [ ] Write test for embedding query text
- [ ] Write test for similarity search with configurable limit
- [ ] Write test for result ranking by score
- [ ] Write test for handling empty queries
- [ ] Write test for handling no-match scenarios

### 4.2 Query Processing Implementation

- [ ] Implement query embedding interface
- [ ] Implement similarity search function
- [ ] Implement result ranking and limiting
- [ ] Implement edge case handling
- [ ] Ensure all query tests pass

### 4.3 Result Formatting Tests

- [ ] Write test for extracting source file from results
- [ ] Write test for extracting content snippet from results
- [ ] Write test for including similarity score in results
- [ ] Write test for including metadata in results

### 4.4 Result Formatting Implementation

- [ ] Implement result data class with type hints
- [ ] Implement result extraction from vector store response
- [ ] Implement metadata extraction
- [ ] Ensure all result formatting tests pass

## Phase 5: Web UI Module (TDD)

### 5.1 File Listing Tests

- [ ] Write test for endpoint returning all indexed files
- [ ] Write test for file metadata in response
- [ ] Write test for empty index handling
- [ ] Write test for sorting files by name or date

### 5.2 File Listing Implementation

- [ ] Implement FastAPI app initialization
- [ ] Implement GET /files endpoint
- [ ] Implement file list retrieval from vector store
- [ ] Implement response serialization
- [ ] Ensure all file listing tests pass

### 5.3 Search API Tests

- [ ] Write test for POST /search endpoint with query
- [ ] Write test for search result response format
- [ ] Write test for empty query handling
- [ ] Write test for result limit enforcement
- [ ] Write test for error handling

### 5.4 Search API Implementation

- [ ] Implement POST /search endpoint
- [ ] Implement query parameter validation
- [ ] Implement search result serialization
- [ ] Implement error handling and responses
- [ ] Ensure all search API tests pass

### 5.5 Frontend UI

- [ ] Create static HTML template for file listing
- [ ] Create static HTML template for search interface
- [ ] Add minimal CSS for clean UI
- [ ] Add JavaScript for API interaction
- [ ] Implement result display with source file links
- [ ] Test UI manually in browser

## Phase 6: CLI Commands (TDD)

### 6.1 Index Command Tests

- [ ] Write test for upsert-index command execution
- [ ] Write test for command loading configuration
- [ ] Write test for command triggering indexing
- [ ] Write test for command reporting progress
- [ ] Write test for command error handling

### 6.2 Index Command Implementation

- [ ] Create epistemon/cli.py module
- [ ] Implement index_command function
- [ ] Add progress logging
- [ ] Add error handling and user messages
- [ ] Configure upsert-index script in pyproject.toml
- [ ] Ensure all index command tests pass

### 6.3 Web UI Command Tests

- [ ] Write test for web-ui command execution
- [ ] Write test for command starting web server
- [ ] Write test for command loading configuration
- [ ] Write test for command handling shutdown

### 6.4 Web UI Command Implementation

- [ ] Implement web_ui_command function
- [ ] Add server startup logic
- [ ] Add configuration loading
- [ ] Add shutdown handling
- [ ] Configure web-ui script in pyproject.toml
- [ ] Ensure all web UI command tests pass

## Phase 7: Integration and End-to-End Testing

### 7.1 Integration Tests

- [ ] Write test for full indexing workflow (config -> scan -> chunk -> embed -> store)
- [ ] Write test for full search workflow (query -> embed -> search -> format)
- [ ] Write test for incremental re-indexing workflow
- [ ] Write test for web API integration with search module

### 7.2 Integration Implementation

- [ ] Ensure all modules integrate correctly
- [ ] Fix any integration issues
- [ ] Ensure all integration tests pass

### 7.3 Example Data

- [ ] Create sample config.yaml with sensible defaults
- [ ] Create tests/data/ with example markdown files
- [ ] Create example input directory with markdown files
- [ ] Document configuration options in README

## Phase 8: Code Quality and Documentation

### 8.1 Code Quality Checks

- [ ] Run black formatter on all code
- [ ] Run ruff linter and fix all issues
- [ ] Run mypy strict type checking and fix all issues
- [ ] Run pytest and ensure 100% passing tests
- [ ] Run pip-audit and ensure no vulnerabilities

### 8.2 Documentation

- [ ] Add docstrings to all public functions and classes
- [ ] Update README with installation instructions
- [ ] Update README with usage examples
- [ ] Add inline comments only where logic is non-obvious
- [ ] Create example workflow documentation

## Phase 9: Polish and Release

### 9.1 Final Testing

- [ ] Test upsert-index command with real markdown files
- [ ] Test web-ui command and verify UI in browser
- [ ] Test incremental indexing by modifying files
- [ ] Test search quality with various queries
- [ ] Test error handling for missing config, invalid paths, etc.

### 9.2 Performance

- [ ] Profile indexing performance with large markdown collections
- [ ] Optimize chunking if needed
- [ ] Optimize search if needed
- [ ] Ensure reasonable memory usage

### 9.3 Release Readiness

- [ ] Update version in pyproject.toml
- [ ] Add LICENSE file
- [ ] Final README review
- [ ] Tag release in git
- [ ] Update this TODO with final status
