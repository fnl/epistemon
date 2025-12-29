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
- [x] Create epistemon/**init**.py
- [x] Create epistemon/indexing/ module directory
- [x] Create epistemon/search/ module directory
- [x] Create epistemon/web/ module directory
- [x] Create tests/data/ directory for test markdown files
- [x] Create config.yaml template file

## Phase 2: Configuration Module (TDD)

### 2.1 Basic Configuration Loading (TDD Cycle 1)

- [x] Write test for basic YAML configuration loading
- [x] Implement Configuration data class with type hints
- [x] Implement YAML file loader

### 2.2 Default Values for All Fields (TDD Cycle 2)

All fields have sensible defaults (config file is optional):

- input_directory: Default "./tests/data"
- embedding_provider: Default "huggingface"
- embedding_model: Default "all-MiniLM-L6-v2"
- vector_store_type: Default "chroma"
- vector_store_path: Default "./data/chroma_db"
- chunk_size: Default 500 (optimized for better semantic granularity)
- chunk_overlap: Default 200 (LangChain best practice)
- search_results_limit: Default 5

Tasks:

- [x] Write test for missing config file (uses all defaults)
- [x] Write test for empty config file (uses all defaults)
- [x] Write test for partial config (some fields overridden)
- [x] Implement default value handling for None config path
- [x] Implement default value handling for empty YAML files

### 2.3 Field Override Validation (TDD Cycle 3) - SKIPPED

Ensure individual field overrides work correctly:

- [~] Write test for overriding input_directory only (SKIPPED - covered by partial override test)
- [~] Write test for overriding embedding config only (SKIPPED - covered by partial override test)
- [~] Write test for overriding vector store config only (SKIPPED - covered by partial override test)

### 2.4 Invalid Configuration Handling (TDD Cycle 4)

- [x] Write test for invalid YAML syntax
- [x] Implement error handling for invalid YAML

### 2.5 File Not Found Handling (TDD Cycle 5)

- [x] Write test for missing configuration file
- [x] Implement error handling for missing files

### 2.6 Enum Validation (TDD Cycle 6)

- [x] Write test for embedding_provider validation
- [x] Write test for vector_store_type validation
- [x] Implement enum validation

### 2.7 Value Constraints (TDD Cycle 7)

- [x] Write test for positive integer validation
- [x] Implement constraints validation

### 2.8 Immutability (TDD Cycle 8)

- [x] Write test for Configuration immutability
- [x] Verify frozen dataclass behavior

### 2.9 Type Validation (TDD Cycle 9)

- [x] Write test for string field type validation
- [x] Write test for integer field type validation
- [x] Implement type validation for all fields

## Phase 3: Walking Skeleton (End-to-End Spike)

### 3.1 Load and Chunk Single Markdown File (TDD Cycle 1)

- [x] Write test for loading and chunking a single markdown file
- [x] Implement load_and_chunk_markdown function

### 3.2 Embed and Index Chunks (TDD Cycle 2)

- [x] Write test for embedding and indexing chunks in InMemoryVectorStore
- [x] Implement embed_and_index function using FakeEmbeddings

### 3.3 Search Indexed Content (TDD Cycle 3)

- [x] Write test for searching indexed content
- [x] Implement search function

### 3.4 FastAPI Search Endpoint (TDD Cycle 4)

- [x] Write test for GET /search endpoint
- [x] Implement minimal FastAPI app with search endpoint

### 3.5 Minimal HTML Search UI

- [x] Create minimal HTML page with search form
- [x] Add JavaScript to call search API and display results

### 3.6 End-to-End Manual Test

- [x] Test complete workflow: index sample.md and search via UI

## Phase 4: Indexing Module (TDD)

### 4.1 Scan Directory for Markdown Files (TDD Cycle 1)

- [x] Write test for scanning directory for markdown files
- [x] Implement markdown file scanner

### 4.2 Recursive Directory Traversal (TDD Cycle 2)

- [x] Write test for recursive directory traversal
- [x] Implement recursive directory scanning
- [x] Make recursive scanning the default behavior

### 4.3 Filter Non-Markdown Files (TDD Cycle 3)

- [x] Write test for filtering non-markdown files
- [x] Implement file extension filtering (covered by \*.md glob pattern)

### 4.X Relative Source Paths

- [x] Write test for relative source paths in metadata
- [x] Implement relative path computation from base directory

### 4.4 Track File Modification Times (TDD Cycle 4)

- [x] Write test for tracking file modification times
- [x] Implement file modification time tracking

### 4.5 Load Markdown File Content (TDD Cycle 5)

- [x] Write test for loading markdown file content (covered in Phase 3.1)
- [x] Implement markdown document loader (covered in Phase 3.1)

### 4.6 Chunk Document Text (TDD Cycle 6)

- [x] Write test for chunking document text (covered in Phase 3.1)
- [x] Implement text splitter with configurable chunk size (covered in Phase 3.1)

### 4.7 Preserve Metadata (TDD Cycle 7)

- [x] Write test for preserving metadata (filename, timestamp) (covered in Phase 3.1 and 4.4)
- [x] Implement metadata extraction and attachment (covered in Phase 3.1 and 4.4)

### 4.8 Handle Empty Files (TDD Cycle 8)

- [x] Write test for handling empty files
- [x] Implement empty file handling (already works - returns empty list)

### 4.9 Handle Malformed Markdown (TDD Cycle 9)

- [x] Write test for handling malformed markdown
- [x] Implement malformed markdown handling (already works - treats as text)

### 4.10 Detect New Files (TDD Cycle 10)

- [x] Write test for detecting new files
- [x] Implement file state tracking and new file detection

### 4.11 Detect Modified Files (TDD Cycle 11)

- [x] Write test for detecting modified files
- [x] Implement modified file detection logic

### 4.12 Skip Unchanged Files (TDD Cycle 12)

- [x] Write test for skipping unchanged files
- [x] Implement unchanged file skipping logic

### 4.13 Log Warnings for Empty Files

- [x] Write test for logging warning when empty file is encountered
- [x] Implement warning log when file produces no chunks

### 4.14 Remove Deleted File Embeddings (TDD Cycle 14)

- [x] Write test for removing deleted file embeddings
- [x] Implement deletion handling

### 4.15 Create Embeddings from Chunks (TDD Cycle 15)

- [x] Write test for creating embeddings from chunks
- [x] Implement embedding generation interface

### 4.16 Store Embeddings in Vector Store (TDD Cycle 16)

- [x] Write test for storing embeddings in vector store
- [x] Implement vector store initialization and storage

### 4.17 Update Existing Embeddings (TDD Cycle 17)

- [x] Write test for updating existing embeddings
- [x] Implement embedding update logic

### 4.18 Vector Store Persistence (TDD Cycle 18)

- [x] Write test for vector store persistence
- [x] Implement persistence handling

### 4.19 Make embeddings configurable (TDD Cycle 19)

- [x] Write test to validate the vector store is using the configured embedding model
- [x] Implement configurable embeddings as documented in the README

### 4.20 Add unstructured markdown splitting (TDD Cycle 20)

- [x] Write test to validate that the chunks are created around markdown elements like titles, paragraphs, and tables
- [x] Implement proper markdown chunking with MarkdownTextSplitter to create syntactic chunks
- [x] Write test to validate that huge chunks that are too big for embeddings get split up into smaller docs
- [x] Implement sub-chunking of markdown chunks that are too big, according to the chunk size settings
- [x] Clean up the implementation and configuration, checking for refactoring options

### 4.21 Create a unified API to run an indexing process (TDD Cycle 21)

- [x] Write a test for a function index(directory: Path, vector_store: VectorStore) -> None: that updates the given vector store with the contents of directory
- [x] Implement the function so that it upserts all new files into the vector store, encapsulating the existing functions in the indexer
- [x] Review if some refactoring is necessary to make use of LangChain's ability to process multiple documents at once

### 4.22 Optimize the indexing process

- [x] Create instrumentation to measure the time to index a file
- [x] Improve the indexing process over a directory of markdown files with FakeEmbeddings to be under 0.01s per file, or get as close as possible
- [x] If new functions need to be created, add relevant tests for their core behavior

Performance achieved:

- New file indexing: 0.23ms per file (target: 10ms) - 43x better than target
- Re-indexing unchanged files: 0.07ms per file (target: 10ms) - 142x better than target
- Instrumentation module created for detailed performance breakdown
- Three performance tests added to validate and monitor performance

### 4.23 Don't let chunks end in headlines unless the headline is the only piece of the chunk.

- [ ] Write a test where markdown chunk using the configured chunk size would cover a headline, a paragraph, and another headline, but not the next paragraph. The test should ensure that second headline is NOT part of the chunk, and headlines don't get "torn" from the paragraphs they follow.
- [ ] Implement any changes needed to ensure that the chunker.py correctly prevents a "dangling" headline in a chunk.

## Phase 5: GET /search API (TDD)

### 5.1 Embed Query Text (TDD Cycle 1)

- [~] Write test for embedding query text (Skip)
- [~] Implement query interface using a VectorStoreRetriever. (Skip)

### 5.2 Similarity Search with Configurable Limit (TDD Cycle 2)

- [x] Write test for similarity search with configurable result limit
- [x] Implement similarity search with k limit

### 5.3 Result Ranking by Score (TDD Cycle 3)

- [x] Write test for result ranking by score
- [x] Implement result ranking and limiting

### 5.4 Empty Query Handling (TDD Cycle 4)

- [x] Write test for handling empty queries
- [x] Implement empty query edge case handling

### 5.5 Below cutoff score handling (TDD Cycle 5)

- [x] Write test for handling no-match scenarios if all results are below a cutoff score.
- [x] Implement no-match scenario handling: return an empty result, but with an alert that no match was found

### 5.6 Link to Source File from Results (TDD Cycle 6)

- [x] Write test to check for links to source file from each search result
- [x] Implement linking result chunks to a configurable base http url (Configuration!) with the encoded source path appended
- [x] Add a simple mechanism to serve the test data markdown files over http, to test and validate the functionality

### 5.7 Extract Content Snippet from Results (TDD Cycle 7) - SKIPPED

- [~] Write test for extracting content snippet from results (SKIPPED - content already included)
- [~] Implement content snippet extraction from vector store response (SKIPPED - content already included)

Note: Content snippets are already included in the page_content field of search results.
Additionally, markdown files are now rendered as nicely formatted HTML when accessed via links.

### 5.8 Include Similarity Score in Results (TDD Cycle 8)

- [x] Write test for displaying similarity score in web UI
- [x] Update web UI to display similarity scores for each result
- [x] Ensure results are ordered by decreasing similarity score (test exists and passes)

### 5.9 Include Metadata in Results (TDD Cycle 9)

- [ ] Write test for including metadata in results
- [ ] Implement metadata extraction and inclusion

## Phase 6: Web UI Module (TDD)

### 6.1 Endpoint Returns All Indexed Files (TDD Cycle 1)

- [ ] Write test for endpoint returning all indexed files
- [ ] Implement FastAPI app initialization and GET /files endpoint

### 6.2 File Metadata in Response (TDD Cycle 2)

- [ ] Write test for file metadata in response
- [ ] Implement file list retrieval from vector store with metadata

### 6.3 Empty Index Handling (TDD Cycle 3)

- [ ] Write test for empty index handling
- [ ] Implement empty index response handling

### 6.4 Sorting Files by Name or Date (TDD Cycle 4)

- [ ] Write test for sorting files by name or date
- [ ] Implement file sorting and response serialization

### 6.5 POST /search Endpoint with Query (TDD Cycle 5)

- [ ] Write test for POST /search endpoint with query
- [ ] Implement POST /search endpoint

### 6.6 Search Result Response Format (TDD Cycle 6)

- [ ] Write test for search result response format
- [ ] Implement search result serialization

### 6.7 Empty Query Handling (TDD Cycle 7)

- [ ] Write test for empty query handling
- [ ] Implement query parameter validation

### 6.8 Result Limit Enforcement (TDD Cycle 8)

- [ ] Write test for result limit enforcement
- [ ] Implement result limit enforcement logic

### 6.9 Error Handling (TDD Cycle 9)

- [ ] Write test for error handling
- [ ] Implement error handling and error responses

### 6.10 Frontend UI

- [ ] Create static HTML template for file listing
- [ ] Create static HTML template for search interface
- [ ] Add minimal CSS for clean UI
- [ ] Add JavaScript for API interaction
- [ ] Implement result display with source file links
- [ ] Test UI manually in browser

## Phase 7: CLI Commands (TDD)

### 7.1 Index Command Execution (TDD Cycle 1)

- [ ] Write test for upsert-index command execution
- [ ] Create epistemon/cli.py module and implement index_command function

### 7.2 Command Loads Configuration (TDD Cycle 2)

- [ ] Write test for command loading configuration
- [ ] Implement configuration loading in index_command

### 7.3 Command Triggers Indexing (TDD Cycle 3)

- [ ] Write test for command triggering indexing
- [ ] Implement indexing trigger logic

### 7.4 Command Reports Progress (TDD Cycle 4)

- [ ] Write test for command reporting progress
- [ ] Add progress logging

### 7.5 Command Error Handling (TDD Cycle 5)

- [ ] Write test for command error handling
- [ ] Add error handling and user messages

### 7.6 Configure Index Command Script (TDD Cycle 6)

- [ ] Write test for upsert-index script in pyproject.toml
- [ ] Configure upsert-index script in pyproject.toml

### 7.7 Web UI Command Execution (TDD Cycle 7)

- [ ] Write test for web-ui command execution
- [ ] Implement web_ui_command function

### 7.8 Command Starts Web Server (TDD Cycle 8)

- [ ] Write test for command starting web server
- [ ] Add server startup logic

### 7.9 Command Loads Configuration (TDD Cycle 9)

- [ ] Write test for command loading configuration
- [ ] Add configuration loading

### 7.10 Command Handles Shutdown (TDD Cycle 10)

- [ ] Write test for command handling shutdown
- [ ] Add shutdown handling

### 7.11 Configure Web UI Command Script (TDD Cycle 11)

- [ ] Write test for web-ui script in pyproject.toml
- [ ] Configure web-ui script in pyproject.toml

## Phase 8: Integration and End-to-End Testing

### 8.1 Full Indexing Workflow Integration Test

- [ ] Write test for full indexing workflow (config -> scan -> chunk -> embed -> store)
- [ ] Ensure all indexing modules integrate correctly

### 8.2 Full Search Workflow Integration Test

- [ ] Write test for full search workflow (query -> embed -> search -> format)
- [ ] Ensure search module integrates with indexing output

### 8.3 Incremental Re-indexing Integration Test

- [ ] Write test for incremental re-indexing workflow
- [ ] Fix any incremental update issues

### 8.4 Web API Integration Test

- [ ] Write test for web API integration with search module
- [ ] Ensure FastAPI endpoints work with search functionality

### 8.5 Example Configuration

- [ ] Create sample config.yaml with sensible defaults
- [ ] Document configuration options in README

### 8.6 Example Markdown Data

- [ ] Create tests/data/ with example markdown files
- [ ] Create example input directory with markdown files

## Phase 9: Documentation

### 9.1 Add Docstrings

- [ ] Add docstrings to all public functions and classes

### 9.2 Installation Instructions

- [ ] Update README with installation instructions

### 9.3 Usage Examples

- [ ] Update README with usage examples

### 9.4 Code Comments

- [ ] Add inline comments only where logic is non-obvious

### 9.5 Workflow Documentation

- [ ] Create example workflow documentation

## Phase 10: Polish and Release

### 10.1 Test Index Command

- [ ] Test upsert-index command with real markdown files

### 10.2 Test Web UI Command

- [ ] Test web-ui command and verify UI in browser

### 10.3 Test Incremental Indexing

- [ ] Test incremental indexing by modifying files

### 10.4 Test Search Quality

- [ ] Test search quality with various queries

### 10.5 Test Error Handling

- [ ] Test error handling for missing config, invalid paths, etc.

### 10.6 Profile Indexing Performance

- [ ] Profile indexing performance with large markdown collections

### 10.7 Optimize Chunking

- [ ] Optimize chunking if needed

### 10.8 Optimize Search

- [ ] Optimize search if needed

### 10.9 Memory Usage

- [ ] Ensure reasonable memory usage

### 10.10 Update Version

- [ ] Update version in pyproject.toml

### 10.11 Add License

- [ ] Add LICENSE file

### 10.12 Final README Review

- [ ] Final README review

### 10.13 Tag Release

- [ ] Tag release in git

### 10.14 Update TODO

- [ ] Update this TODO with final status
