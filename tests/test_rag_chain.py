"""Tests for the RAG chain module."""

from unittest.mock import Mock

from langchain_core.documents import Document

from epistemon.rag.rag_chain import RAGChain, RAGResponse


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


def test_rag_chain_initialization() -> None:
    """Test that RAGChain can be instantiated with retriever and LLM."""
    retriever = Mock()
    llm = Mock()

    chain = RAGChain(retriever=retriever, llm=llm)

    assert chain is not None
    assert chain.retriever == retriever
    assert chain.llm == llm


def test_format_context_documents() -> None:
    """Test that documents are formatted with clear separators and metadata."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="Content from file 1", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Content from file 2", metadata={"source": "file2.md"})

    context = chain.format_context([doc1, doc2])

    assert "Content from file 1" in context
    assert "Content from file 2" in context
    assert "file1.md" in context
    assert "file2.md" in context
    assert context.count("---") >= 1
