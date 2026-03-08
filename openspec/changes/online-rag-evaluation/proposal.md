## Why

Epistemon generates RAG answers from hybrid retrieval, but there is no signal on whether those answers are good. Without per-query quality scores it is impossible to know which user queries are failing or whether BM25, semantic, or hybrid retrieval is the root cause.

## What Changes

- Add an LLM-as-judge module that scores every live user query on four metrics after the RAG pipeline completes.
- Extend the traced retriever wrappers to capture intermediate BM25 and semantic result sets so all three retrieval paths can be scored independently.
- Post four trace-level scores to LangFuse after each query so queries can be sorted and ranked by any score in the LangFuse dashboard.
- Scoring runs in a background thread so there is no latency impact on the user-facing response.

## Capabilities

### New Capabilities

- `rag-evaluation`: Per-query LLM-as-judge scoring of context relevance (BM25, semantic, hybrid) and answer faithfulness, posted as LangFuse trace-level scores.

### Modified Capabilities

- `tracing`: Traced retriever wrappers must capture last result sets and observation IDs so the evaluation module can access all three document sets after a query.

## Impact

- New module `epistemon/evaluation/` with `judge.py`.
- `epistemon/tracing.py`: `TracedBM25Retriever`, `TracedSemanticRetriever`, and `TracedRAGChain` extended.
- `TracedRAGChain` gains an optional `judge` dependency; when present, spawns a background scoring thread after each invocation.
- No new external dependencies: uses the existing OpenAI LLM and the existing LangFuse client.
- No breaking changes to public interfaces.
