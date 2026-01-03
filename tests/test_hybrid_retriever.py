"""Tests for the HybridRetriever class."""

from epistemon.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_initialization() -> None:
    """Test that HybridRetriever can be instantiated."""
    retriever = HybridRetriever()

    assert retriever is not None
