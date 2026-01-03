"""Tests for the HybridRetriever class."""

from unittest.mock import Mock

from langchain_core.documents import Document

from epistemon.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_initialization() -> None:
    """Test that HybridRetriever can be instantiated."""
    retriever = HybridRetriever()

    assert retriever is not None


def test_merge_and_deduplicate_results() -> None:
    """Test that duplicate documents are merged with highest score."""
    bm25_retriever = Mock()
    semantic_retriever = Mock()

    doc1 = Document(page_content="Content 1", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Content 2", metadata={"source": "file2.md"})
    doc3 = Document(page_content="Content 1 variant", metadata={"source": "file1.md"})

    bm25_retriever.retrieve.return_value = [
        (doc1, 0.9),
        (doc2, 0.7),
    ]
    semantic_retriever.similarity_search_with_score.return_value = [
        (doc3, 0.8),
        (doc2, 0.6),
    ]

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever, semantic_retriever=semantic_retriever
    )
    results = retriever.retrieve("test query")

    assert len(results) == 2
    assert results[0][0].metadata["source"] == "file1.md"
    assert results[0][1] == 0.9
    assert results[1][0].metadata["source"] == "file2.md"
    assert results[1][1] == 0.7


def test_empty_results_handling() -> None:
    """Test that empty results from both retrievers are handled gracefully."""
    bm25_retriever = Mock()
    semantic_retriever = Mock()

    bm25_retriever.retrieve.return_value = []
    semantic_retriever.similarity_search_with_score.return_value = []

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever, semantic_retriever=semantic_retriever
    )
    results = retriever.retrieve("test query")

    assert results == []
    assert isinstance(results, list)
