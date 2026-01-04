"""Hybrid retriever combining BM25 and vector-based search using RRF."""

from typing import Optional, Protocol

from langchain_core.documents import Document


class BM25RetrieverProtocol(Protocol):
    """Protocol for BM25-style retrievers with a retrieve method."""

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents with BM25 scores."""
        ...


class SemanticRetrieverProtocol(Protocol):
    """Protocol for semantic retrievers with similarity search."""

    def similarity_search_with_score(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents with semantic similarity scores."""
        ...


class HybridRetriever:
    """Combines BM25 keyword search with vector-based semantic search using Reciprocal Rank Fusion.

    This retriever merges results from a keyword-based (BM25) retriever and a
    semantic (embedding-based) retriever using the Reciprocal Rank Fusion (RRF)
    algorithm. RRF computes fused scores based on rank positions rather than
    raw scores, making it effective for combining results from different retrieval
    systems with incompatible scoring scales.

    The RRF score for each document is computed as:
        RRF_score = sum over retrievers of: weight / (k + rank)

    where rank is the 1-indexed position in that retriever's results, k is a
    constant (default 60), and weight is the retriever-specific weight.
    """

    def __init__(
        self,
        bm25_retriever: BM25RetrieverProtocol,
        semantic_retriever: SemanticRetrieverProtocol,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever with BM25 and semantic retrievers.

        Args:
            bm25_retriever: A retriever implementing the retrieve(query) method
                that returns list[tuple[Document, float]]
            semantic_retriever: A retriever implementing the
                similarity_search_with_score(query) method that returns
                list[tuple[Document, float]]
            bm25_weight: Weight for BM25 results in RRF fusion (default 0.5)
            semantic_weight: Weight for semantic results in RRF fusion (default 0.5)
            rrf_k: Constant for RRF score calculation (default 60). Higher values
                reduce the impact of rank position differences.
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k

    def _accumulate_rrf_scores(
        self,
        results: list[tuple[Document, float]],
        weight: float,
        rrf_scores: dict[str, float],
        doc_map: dict[str, Document],
    ) -> None:
        """Accumulate RRF scores from retriever results.

        Args:
            results: List of (Document, score) tuples from a retriever
            weight: Weight to apply to this retriever's contributions
            rrf_scores: Dictionary to accumulate RRF scores by document source
            doc_map: Dictionary to store first occurrence of each document
        """
        for rank, (doc, _score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "")
            rrf_scores[source] = rrf_scores.get(source, 0.0) + weight / (
                self.rrf_k + rank
            )
            if source not in doc_map:
                doc_map[source] = doc

    def retrieve(
        self, query: str, max_docs: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """Retrieve and merge results using Reciprocal Rank Fusion.

        RRF score = sum over retrievers of: weight / (k + rank)
        where rank is 1-indexed position in that retriever's results.

        Args:
            query: The search query
            max_docs: Maximum number of documents to return (None for no limit)

        Returns:
            List of (Document, RRF score) tuples, sorted by RRF score descending
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        bm25_results = self.bm25_retriever.retrieve(query)
        self._accumulate_rrf_scores(bm25_results, self.bm25_weight, rrf_scores, doc_map)

        semantic_results = self.semantic_retriever.similarity_search_with_score(query)
        self._accumulate_rrf_scores(
            semantic_results, self.semantic_weight, rrf_scores, doc_map
        )

        sorted_results = sorted(
            [(doc_map[source], score) for source, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        if max_docs is not None:
            return sorted_results[:max_docs]

        return sorted_results
