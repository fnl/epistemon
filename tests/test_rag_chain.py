"""Tests for the RAG chain module."""

from langchain_core.documents import Document

from epistemon.rag.rag_chain import RAGResponse


def test_rag_response_creation() -> None:
    """Test that RAGResponse can be created with required fields."""
    doc1 = Document(page_content="Content 1", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Content 2", metadata={"source": "file2.md"})

    response = RAGResponse(
        answer="This is the generated answer",
        source_documents=[doc1, doc2],
        query="What is this about?",
    )

    assert response.answer == "This is the generated answer"
    assert len(response.source_documents) == 2
    assert response.query == "What is this about?"
