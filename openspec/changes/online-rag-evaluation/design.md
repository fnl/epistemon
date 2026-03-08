## Context

Epistemon has a fully instrumented RAG pipeline via `TracedRAGChain` in `tracing.py`. Each query produces a LangFuse trace with spans for BM25 search, semantic search, retrieval (hybrid), and generation. The `TracedBM25Retriever` and `TracedSemanticRetriever` wrappers already record result counts and source names in their span outputs, but do not retain the full document sets after the span closes. The `TracedRAGChain` only receives the fused hybrid document set from `HybridRetriever.retrieve()`.

To score context relevance across all three retrieval strategies, all three document sets must be available in one place after retrieval completes. The LangFuse Python SDK (v3) scoring API accepts `trace_id` and optional `observation_id`, allowing scores to be attached to the top-level trace.

## Goals / Non-Goals

**Goals:**
- Score every live user query on `context_relevance_bm25`, `context_relevance_semantic`, `context_relevance_hybrid`, and `answer_faithfulness` (0.0-1.0).
- Post all four as trace-level scores with a reason comment so queries are sortable in LangFuse.
- Add zero latency to the user-facing response (background thread).
- Reuse the existing OpenAI LLM and LangFuse client; add no new dependencies.

**Non-Goals:**
- Batch/offline evaluation against a synthetic dataset.
- Scoring at the observation/span level (trace-level is sufficient for ranking).
- Modifying the Shiny UI or any public interfaces.
- Supporting LangFuse cloud-only features.

## Decisions

### D1: Trace-level scores over observation-level scores

LangFuse's UI allows sorting and filtering all traces by trace-level scores. Observation-level scores are visible only after clicking into a trace. Since the goal is to rank all user queries by quality, trace-level is the right granularity. All four scores are posted with `langfuse.score(trace_id=..., name=..., value=..., comment=...)`.

Alternative considered: scoring on the `bm25-search` and `semantic-search` spans directly. Rejected because the LangFuse dashboard does not expose observation-level scores in the trace list view.

### D2: Capture last_results on the traced retriever wrappers

`TracedBM25Retriever` and `TracedSemanticRetriever` are extended with a `last_results` attribute that stores the result list from the most recent call. After `_retrieve_with_span` completes, `TracedRAGChain` reads these attributes to get the BM25 and semantic document sets without re-running retrieval.

Alternative considered: extending `HybridRetriever.retrieve()` to return all three result sets. Rejected because it would change the `RetrieverProtocol` interface and cascade through the codebase.

Alternative considered: re-running BM25 and semantic retrieval in the background scorer. Rejected because semantic search triggers an embedding API call, making duplication expensive.

### D3: Async scoring via background thread

Scoring requires two or four LLM calls (one per score). Running synchronously would roughly double query latency. A `threading.Thread(daemon=True)` is started after `RAGResponse` is returned to the caller. The thread captures the trace ID, query, documents, and answer by value before the context closes.

Alternative considered: `asyncio` task. Rejected because the Shiny server is synchronous and mixing asyncio contexts adds complexity.

### D4: New epistemon/evaluation/ module

Judge logic (prompts, LLM calls, score parsing) lives in `epistemon/evaluation/judge.py`. It accepts an `LLMProtocol` and exposes `score_context_relevance(query, docs) -> JudgeScore` and `score_answer_faithfulness(query, docs, answer) -> JudgeScore`. `TracedRAGChain` accepts an optional `judge: RetrievalJudge` dependency.

This keeps the judge testable independently of tracing by injecting a mock LLM.

### D5: Structured JSON response from the judge

Each judge call instructs the LLM to respond with `{"score": <float 0.0-1.0>, "reason": "<one sentence>"}`. The module parses the JSON and falls back gracefully (score=0.0, reason="parse error") if the LLM responds in an unexpected format.

Alternative considered: free-text with regex extraction. Rejected as fragile.

## Risks / Trade-offs

- **Statefulness of last_results**: `TracedBM25Retriever.last_results` is overwritten on every call. Under concurrent requests this could return wrong results for a given query. For a local single-user instance this is acceptable. → Mitigation: document the limitation; a future fix would use thread-local storage.
- **LLM judge cost**: Four LLM calls per user query. Calls are short (one chunk at a time) but still incur API cost. → Mitigation: judge is opt-in via the `judge` parameter; if `None`, no scoring happens.
- **Judge accuracy**: An LLM judge is not ground truth. It can be wrong. → Mitigation: scores are a triage signal, not a definitive measure. Low scores prompt human investigation.
- **Background thread failure**: If the scoring thread raises an exception, it is silently swallowed unless logged. → Mitigation: wrap the thread body in a try/except and log errors at WARNING level.

## Migration Plan

The `judge` parameter on `TracedRAGChain` defaults to `None`, making the feature fully opt-in. Existing deployments continue to work without change. To enable, instantiate a `RetrievalJudge` with the existing LLM and pass it when constructing `TracedRAGChain` via `create_traced_rag_chain`.
