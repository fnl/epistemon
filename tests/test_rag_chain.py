"""Tests for the RAG chain module."""

from unittest.mock import Mock

from langchain_core.documents import Document

from epistemon.retrieval.rag_chain import RAGChain, RAGResponse


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


def test_basic_answer_generation() -> None:
    """Test that the RAG chain generates answers using the LLM."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(
        page_content="LangChain is a framework", metadata={"source": "file1.md"}
    )
    doc2 = Document(
        page_content="It helps build LLM apps", metadata={"source": "file2.md"}
    )

    retriever.retrieve.return_value = [(doc1, 0.9), (doc2, 0.8)]
    llm.invoke.return_value = Mock(
        content="LangChain is a framework for building LLM applications."
    )

    response = chain.invoke("What is LangChain?")

    assert isinstance(response, RAGResponse)
    assert response.answer == "LangChain is a framework for building LLM applications."
    assert response.query == "What is LangChain?"
    assert len(response.source_documents) == 2
    assert response.source_documents[0] == doc1
    assert response.source_documents[1] == doc2


def test_empty_context_handling() -> None:
    """Test that empty context is handled gracefully without calling the LLM."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    retriever.retrieve.return_value = []

    response = chain.invoke("What is LangChain?")

    assert isinstance(response, RAGResponse)
    assert "no relevant documents" in response.answer.lower()
    assert response.query == "What is LangChain?"
    assert len(response.source_documents) == 0
    llm.invoke.assert_not_called()


def test_source_document_preservation() -> None:
    """Test that source documents and metadata are preserved in the response."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(
        page_content="Python is a programming language",
        metadata={"source": "python.md", "last_modified": 1234567890},
    )
    doc2 = Document(
        page_content="It is widely used",
        metadata={"source": "usage.md", "last_modified": 1234567891},
    )

    retriever.retrieve.return_value = [(doc1, 0.95), (doc2, 0.85)]
    llm.invoke.return_value = Mock(
        content="Python is a widely used programming language."
    )

    response = chain.invoke("What is Python?")

    assert len(response.source_documents) == 2
    assert response.source_documents[0].metadata["source"] == "python.md"
    assert response.source_documents[0].metadata["last_modified"] == 1234567890
    assert response.source_documents[1].metadata["source"] == "usage.md"
    assert response.source_documents[1].metadata["last_modified"] == 1234567891


def test_api_error_handling() -> None:
    """Test that API errors are caught and handled gracefully."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="Some content", metadata={"source": "file.md"})

    retriever.retrieve.return_value = [(doc1, 0.9)]
    llm.invoke.side_effect = Exception("API rate limit exceeded")

    response = chain.invoke("What is this about?")

    assert isinstance(response, RAGResponse)
    assert "error" in response.answer.lower()
    assert "rate limit" in response.answer.lower()
    assert response.query == "What is this about?"
    assert len(response.source_documents) == 1


def test_custom_prompt_template() -> None:
    """Test that custom prompt templates are used correctly."""
    retriever = Mock()
    llm = Mock()
    custom_template = "Context: {context}\n\nQ: {query}\nA:"

    chain = RAGChain(retriever=retriever, llm=llm, prompt_template=custom_template)

    doc1 = Document(page_content="Python is great", metadata={"source": "python.md"})

    retriever.retrieve.return_value = [(doc1, 0.9)]
    llm.invoke.return_value = Mock(content="Python is a programming language.")

    response = chain.invoke("What is Python?")

    llm.invoke.assert_called_once()
    call_args = llm.invoke.call_args[0][0]
    assert "Q: What is Python?" in call_args
    assert "Context:" in call_args
    assert "Python is great" in call_args
    assert isinstance(response, RAGResponse)
