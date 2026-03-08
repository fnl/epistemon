## 1. JudgeScore and context relevance scoring

- [ ] 1.1 [epistemon-a5i] Test `score_context_relevance` with a mock LLM returning valid JSON — assert score and reason parsed correctly
- [ ] 1.2 [epistemon-ayj] Create `epistemon/evaluation/__init__.py`, implement `JudgeScore` dataclass (score: float, reason: str) in `epistemon/evaluation/judge.py`, and implement `score_context_relevance(query, docs, llm) -> JudgeScore` with structured JSON prompt
- [ ] 1.3 [epistemon-461] Test `score_context_relevance` with a mock LLM returning invalid JSON — assert score=0.0 and reason="parse error"
- [ ] 1.4 [epistemon-9do] Add graceful parse-error fallback to `score_context_relevance`

## 2. Answer faithfulness scoring

- [ ] 2.1 [epistemon-nb3] Test `score_answer_faithfulness` with a mock LLM returning valid JSON — assert score and reason parsed correctly
- [ ] 2.2 [epistemon-nu1] Implement `score_answer_faithfulness(query, docs, answer, llm) -> JudgeScore` with structured JSON prompt
- [ ] 2.3 [epistemon-xsj] Test `score_answer_faithfulness` with a mock LLM returning invalid JSON — assert score=0.0 and reason="parse error"
- [ ] 2.4 [epistemon-gly] Add graceful parse-error fallback to `score_answer_faithfulness`

## 3. RetrievalJudge class

- [ ] 3.1 [epistemon-4fi] Test that `RetrievalJudge` delegates `score_context_relevance` and `score_answer_faithfulness` to the underlying LLM and returns the correct `JudgeScore`
- [ ] 3.2 [epistemon-cdr] Implement `RetrievalJudge` class that holds an `LLMProtocol` and exposes both score methods

## 4. Tracing wrappers with last_results

- [ ] 4.1 [epistemon-3zw] Test that `TracedBM25Retriever.last_results` is populated after `retrieve()` is called
- [ ] 4.2 [epistemon-bmx] Add `last_results: list[tuple[Document, float]]` attribute to `TracedBM25Retriever` and populate it in `retrieve()`
- [ ] 4.3 [epistemon-k3d] Test that `TracedSemanticRetriever.last_results` is populated after `similarity_search_with_score()` is called
- [ ] 4.4 [epistemon-0vw] Add `last_results: list[tuple[Document, float]]` attribute to `TracedSemanticRetriever` and populate it in `similarity_search_with_score()`

## 5. TracedRAGChain scoring integration

- [ ] 5.1 [epistemon-spo] Test that `TracedRAGChain` without a judge makes no judge calls and no scoring API calls
- [ ] 5.2 [epistemon-6wp] Add optional `judge: RetrievalJudge | None` parameter to `TracedRAGChain.__init__` and thread it through `create_traced_rag_chain`; no-op when judge is None
- [ ] 5.3 [epistemon-wgc] Test that `TracedRAGChain` with a judge posts four scores to LangFuse after `invoke()`
- [ ] 5.4 [epistemon-zli] After `_retrieve_with_span` returns, read `last_results` from the traced BM25 and semantic wrappers (guarded by `isinstance` check for `HybridRetriever`); implement `_score_async` that posts all four scores via `langfuse.score(trace_id=..., name=..., value=..., comment=...)`
- [ ] 5.5 [epistemon-mvb] Test that the RAGResponse is returned before the scoring thread has run (mock thread start to verify ordering)
- [ ] 5.6 [epistemon-0eu] Start `_score_async` in a `threading.Thread(daemon=True)` after the `RAGResponse` is ready
- [ ] 5.7 [epistemon-zer] Test that a judge LLM exception is caught, logged, and does not propagate to the caller
- [ ] 5.8 [epistemon-m2m] Wrap thread body in try/except and log warnings on failure
