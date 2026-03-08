## MODIFIED Requirements

### Requirement: Traced retriever wrappers capture last result set
`TracedBM25Retriever` and `TracedSemanticRetriever` SHALL each store the document list from their most recent call in a `last_results` attribute so that `TracedRAGChain` can access all three retrieval result sets after the hybrid retrieval step completes, without re-running any retriever.

#### Scenario: BM25 last_results available after retrieve
- **WHEN** `TracedBM25Retriever.retrieve(query)` is called
- **THEN** `TracedBM25Retriever.last_results` contains the list of `(Document, float)` tuples returned by that call

#### Scenario: Semantic last_results available after similarity search
- **WHEN** `TracedSemanticRetriever.similarity_search_with_score(query)` is called
- **THEN** `TracedSemanticRetriever.last_results` contains the list of `(Document, float)` tuples returned by that call

### Requirement: TracedRAGChain accepts an optional judge dependency
`TracedRAGChain.__init__` SHALL accept an optional `judge` parameter. When provided, `TracedRAGChain` SHALL read `last_results` from the traced BM25 and semantic retriever wrappers after retrieval and pass all three document sets to the judge in a background thread. When absent, behaviour is unchanged.

#### Scenario: Judge wired in via create_traced_rag_chain
- **WHEN** `create_traced_rag_chain` is called with a judge instance
- **THEN** the returned `TracedRAGChain` holds a reference to that judge

#### Scenario: No judge means no scoring calls
- **WHEN** `TracedRAGChain` is constructed without a judge
- **THEN** no calls to any judge method are made during invoke
