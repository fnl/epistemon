"""Tests for the HybridRetriever class."""

from unittest.mock import Mock

from langchain_core.documents import Document

from epistemon.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_initialization() -> None:
    """Test that HybridRetriever can be instantiated with retrievers."""
    bm25_retriever = Mock()
    semantic_retriever = Mock()

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever, semantic_retriever=semantic_retriever
    )

    assert retriever is not None
    assert retriever.bm25_retriever == bm25_retriever
    assert retriever.semantic_retriever == semantic_retriever
    assert retriever.bm25_weight == 0.3
    assert retriever.semantic_weight == 0.7
    assert retriever.rrf_k == 60


def test_merge_and_deduplicate_results() -> None:
    """Test that duplicate documents are merged using RRF scoring."""
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
    assert results[1][0].metadata["source"] == "file2.md"
    expected_file1_score = 0.3 / (60 + 1) + 0.7 / (60 + 1)
    expected_file2_score = 0.3 / (60 + 2) + 0.7 / (60 + 2)
    assert abs(results[0][1] - expected_file1_score) < 0.0001
    assert abs(results[1][1] - expected_file2_score) < 0.0001


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


def test_max_docs_limit_enforcement() -> None:
    """Test that max_docs parameter limits the number of results returned."""
    bm25_retriever = Mock()
    semantic_retriever = Mock()

    doc1 = Document(page_content="Content 1", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Content 2", metadata={"source": "file2.md"})
    doc3 = Document(page_content="Content 3", metadata={"source": "file3.md"})
    doc4 = Document(page_content="Content 4", metadata={"source": "file4.md"})

    bm25_retriever.retrieve.return_value = [
        (doc1, 0.9),
        (doc2, 0.7),
        (doc3, 0.5),
    ]
    semantic_retriever.similarity_search_with_score.return_value = [
        (doc4, 0.6),
    ]

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever, semantic_retriever=semantic_retriever
    )
    results = retriever.retrieve("test query", max_docs=2)

    assert len(results) == 2
    result_sources = {r[0].metadata["source"] for r in results}
    assert "file1.md" in result_sources
    assert "file4.md" in result_sources


def test_rrf_with_custom_weights() -> None:
    """Test that RRF correctly applies custom weights."""
    bm25_retriever = Mock()
    semantic_retriever = Mock()

    doc1 = Document(page_content="Content 1", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Content 2", metadata={"source": "file2.md"})

    bm25_retriever.retrieve.return_value = [
        (doc1, 0.9),
    ]
    semantic_retriever.similarity_search_with_score.return_value = [
        (doc2, 0.8),
    ]

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever,
        bm25_weight=0.3,
        semantic_weight=0.7,
    )
    results = retriever.retrieve("test query")

    assert len(results) == 2
    expected_doc1_score = 0.3 / (60 + 1)
    expected_doc2_score = 0.7 / (60 + 1)
    assert results[0][0].metadata["source"] == "file2.md"
    assert abs(results[0][1] - expected_doc2_score) < 0.0001
    assert results[1][0].metadata["source"] == "file1.md"
    assert abs(results[1][1] - expected_doc1_score) < 0.0001
