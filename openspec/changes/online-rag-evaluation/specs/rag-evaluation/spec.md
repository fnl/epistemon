## ADDED Requirements

### Requirement: Judge scores context relevance for each retrieval strategy
The system SHALL score context relevance for BM25, semantic, and hybrid retrieval independently for every query where tracing is enabled. Each score SHALL be a float in [0.0, 1.0] where 0.0 means the retrieved documents are completely irrelevant and 1.0 means they fully support answering the query.

#### Scenario: BM25 context relevance scored
- **WHEN** a query is processed and BM25 results are available
- **THEN** the judge returns a context relevance score and a one-sentence reason for those BM25 documents

#### Scenario: Semantic context relevance scored
- **WHEN** a query is processed and semantic results are available
- **THEN** the judge returns a context relevance score and a one-sentence reason for those semantic documents

#### Scenario: Hybrid context relevance scored
- **WHEN** a query is processed and hybrid results are available
- **THEN** the judge returns a context relevance score and a one-sentence reason for the hybrid documents

### Requirement: Judge scores answer faithfulness
The system SHALL score how faithfully the generated answer is grounded in the retrieved source documents. The score SHALL be a float in [0.0, 1.0] where 0.0 means the answer contradicts or ignores the documents and 1.0 means every claim is directly supported.

#### Scenario: Answer faithfulness scored
- **WHEN** a query produces a non-empty answer and source documents
- **THEN** the judge returns an answer faithfulness score and a one-sentence reason

#### Scenario: Empty answer not scored for faithfulness
- **WHEN** no source documents were retrieved and the answer is the no-documents fallback
- **THEN** answer faithfulness scoring is skipped

### Requirement: Scores posted to LangFuse as trace-level scores
The system SHALL post all four scores to LangFuse using the scoring API with the trace ID of the current query. Each score SHALL include the numeric value and a comment containing the judge's reason. Score names SHALL be `context_relevance_bm25`, `context_relevance_semantic`, `context_relevance_hybrid`, and `answer_faithfulness`.

#### Scenario: All four scores appear on the LangFuse trace
- **WHEN** a query completes and the judge is enabled
- **THEN** four scores with the correct names are attached to the trace in LangFuse

#### Scenario: Scores visible for sorting in LangFuse
- **WHEN** multiple queries have been processed
- **THEN** the LangFuse trace list can be sorted by any of the four score names to surface low-quality responses

### Requirement: Scoring runs asynchronously without blocking the user response
The system SHALL start scoring in a background thread after returning the RAGResponse to the caller so that the user-facing latency is not increased.

#### Scenario: User response returned before scoring completes
- **WHEN** a query is invoked
- **THEN** the RAGResponse is returned to the caller before any judge LLM calls are made

### Requirement: Scoring is opt-in via judge dependency
The system SHALL only perform scoring when a judge is explicitly provided to the traced RAG chain. When no judge is provided, the system SHALL behave identically to the current behaviour with no scoring overhead.

#### Scenario: No scoring when judge is absent
- **WHEN** TracedRAGChain is constructed without a judge
- **THEN** no LLM judge calls are made and no scores are posted to LangFuse

#### Scenario: Scoring enabled when judge is present
- **WHEN** TracedRAGChain is constructed with a judge
- **THEN** all four scores are posted after each query

### Requirement: Judge failures are logged and do not propagate to the caller
The system SHALL catch all exceptions raised during scoring and log them at WARNING level. A scoring failure SHALL NOT raise an exception or affect the RAGResponse returned to the user.

#### Scenario: Judge LLM call fails
- **WHEN** the judge LLM raises an exception during scoring
- **THEN** the error is logged and the user-facing response is unaffected

### Requirement: Judge uses structured JSON responses
The judge SHALL instruct the LLM to respond with `{"score": <float>, "reason": "<string>"}` and SHALL parse that JSON. If parsing fails, the system SHALL return a score of 0.0 with reason "parse error" rather than raising an exception.

#### Scenario: Valid JSON response parsed correctly
- **WHEN** the LLM responds with valid JSON containing score and reason
- **THEN** the judge returns the parsed float score and reason string

#### Scenario: Invalid JSON response handled gracefully
- **WHEN** the LLM responds with text that is not valid JSON
- **THEN** the judge returns score 0.0 and reason "parse error"
