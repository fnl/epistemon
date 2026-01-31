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
    """Test that API errors are raised to the caller."""
    import pytest

    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="Some content", metadata={"source": "file.md"})

    retriever.retrieve.return_value = [(doc1, 0.9)]
    llm.invoke.side_effect = Exception("API rate limit exceeded")

    with pytest.raises(Exception, match="API rate limit exceeded"):
        chain.invoke("What is this about?")


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


def test_invoke_respects_k_limit() -> None:
    """Test that invoke limits the number of documents when k is specified."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="First doc", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Second doc", metadata={"source": "file2.md"})
    doc3 = Document(page_content="Third doc", metadata={"source": "file3.md"})
    doc4 = Document(page_content="Fourth doc", metadata={"source": "file4.md"})

    retriever.retrieve.return_value = [
        (doc1, 0.95),
        (doc2, 0.90),
        (doc3, 0.85),
        (doc4, 0.80),
    ]
    llm.invoke.return_value = Mock(content="Answer based on limited docs.")

    response = chain.invoke("test query", k=2)

    assert len(response.source_documents) == 2
    assert response.source_documents[0] == doc1
    assert response.source_documents[1] == doc2

    call_args = llm.invoke.call_args[0][0]
    assert "First doc" in call_args
    assert "Second doc" in call_args
    assert "Third doc" not in call_args
    assert "Fourth doc" not in call_args


def test_invoke_without_k_uses_all_documents() -> None:
    """Test that invoke uses all documents when k is not specified."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="First doc", metadata={"source": "file1.md"})
    doc2 = Document(page_content="Second doc", metadata={"source": "file2.md"})
    doc3 = Document(page_content="Third doc", metadata={"source": "file3.md"})

    retriever.retrieve.return_value = [
        (doc1, 0.95),
        (doc2, 0.90),
        (doc3, 0.85),
    ]
    llm.invoke.return_value = Mock(content="Answer based on all docs.")

    response = chain.invoke("test query")

    assert len(response.source_documents) == 3
    assert response.source_documents[0] == doc1
    assert response.source_documents[1] == doc2
    assert response.source_documents[2] == doc3


def test_default_prompt_instructs_llm_to_produce_markdown() -> None:
    """Test that the default prompt template instructs the LLM to produce markdown."""
    from epistemon.retrieval.rag_chain import load_default_prompt_template

    prompt = load_default_prompt_template()

    assert "markdown" in prompt.lower()
    assert "format" in prompt.lower() or "use" in prompt.lower()


def test_default_prompt_instructs_llm_to_be_concise() -> None:
    """Test that the default prompt template instructs the LLM to be concise."""
    from epistemon.retrieval.rag_chain import load_default_prompt_template

    prompt = load_default_prompt_template()

    assert "concise" in prompt.lower() or "brief" in prompt.lower()
    assert "relevant" in prompt.lower() or "necessary" in prompt.lower()


def test_format_context_includes_urls_when_base_url_provided() -> None:
    """Test that context includes full URLs when base_url is provided."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="Content 1", metadata={"source": "docs/file1.md"})
    doc2 = Document(page_content="Content 2", metadata={"source": "docs/file2.md"})

    context = chain.format_context([doc1, doc2], base_url="http://example.com/files")

    assert "http://example.com/files/docs/file1.md" in context
    assert "http://example.com/files/docs/file2.md" in context
    assert "Content 1" in context
    assert "Content 2" in context


def test_retrieve_documents_returns_flat_document_list() -> None:
    """Test that retrieve_documents strips scores and returns plain Documents."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(page_content="First", metadata={"source": "a.md"})
    doc2 = Document(page_content="Second", metadata={"source": "b.md"})
    retriever.retrieve.return_value = [(doc1, 0.95), (doc2, 0.80)]

    result = chain.retrieve_documents("some query")

    assert result == [doc1, doc2]


def test_generate_answer_formats_context_and_calls_llm() -> None:
    """Test that generate_answer fills the prompt template and returns a RAGResponse."""
    retriever = Mock()
    llm = Mock()
    custom_template = "Context: {context}\n\nQ: {query}\nA:"
    chain = RAGChain(retriever=retriever, llm=llm, prompt_template=custom_template)

    doc = Document(page_content="Python is great", metadata={"source": "py.md"})
    llm.invoke.return_value = Mock(content="Python is indeed great.")

    response = chain.generate_answer("What is Python?", [doc])

    assert isinstance(response, RAGResponse)
    assert response.answer == "Python is indeed great."
    assert response.query == "What is Python?"
    assert response.source_documents == [doc]
    llm.invoke.assert_called_once()
    prompt_sent = llm.invoke.call_args[0][0]
    assert "Python is great" in prompt_sent
    assert "Q: What is Python?" in prompt_sent


def test_invoke_passes_base_url_to_context() -> None:
    """Test that invoke passes base_url to format_context."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(retriever=retriever, llm=llm)

    doc1 = Document(
        page_content="Test content",
        metadata={"source": "test.md"},
    )

    retriever.retrieve.return_value = [(doc1, 0.9)]
    llm.invoke.return_value = Mock(
        content="Answer with [source](http://example.com/files/test.md)"
    )

    chain.invoke("test query", base_url="http://example.com/files")

    call_args = llm.invoke.call_args[0][0]
    assert "http://example.com/files/test.md" in call_args
