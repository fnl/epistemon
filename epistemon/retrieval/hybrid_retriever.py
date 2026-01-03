"""Hybrid retriever combining BM25 and vector-based search."""

from typing import Any, Optional

from langchain_core.documents import Document


class HybridRetriever:
    """Combines BM25 keyword search with vector-based semantic search."""

    def __init__(
        self,
        bm25_retriever: Optional[Any] = None,
        semantic_retriever: Optional[Any] = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever

    def retrieve(
        self, query: str, max_docs: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """Retrieve and merge results from both BM25 and semantic retrievers.

        Args:
            query: The search query
            max_docs: Maximum number of documents to return (None for no limit)

        Returns:
            List of (Document, score) tuples, deduplicated and sorted by score
        """
        results_dict: dict[str, tuple[Document, float]] = {}

        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query)
            for doc, score in bm25_results:
                source = doc.metadata.get("source", "")
                if source not in results_dict or score > results_dict[source][1]:
                    results_dict[source] = (doc, score)

        if self.semantic_retriever:
            semantic_results = self.semantic_retriever.similarity_search_with_score(
                query
            )
            for doc, score in semantic_results:
                source = doc.metadata.get("source", "")
                if source not in results_dict or score > results_dict[source][1]:
                    results_dict[source] = (doc, score)

        sorted_results = sorted(results_dict.values(), key=lambda x: x[1], reverse=True)

        if max_docs is not None:
            return sorted_results[:max_docs]

        return sorted_results
