"""Tests for the tracing module."""

import logging
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from epistemon.retrieval.rag_chain import RAGChain
from epistemon.tracing import TracedRAGChain, create_traced_rag_chain


def test_traced_rag_chain_creates_retrieval_span() -> None:
    """Test that TracedRAGChain wraps retrieval in a span with document count and sources."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_template="Context: {context}\n\nQ: {query}\nA:",
    )

    doc = Document(page_content="content", metadata={"source": "a.md"})
    retriever.retrieve.return_value = [(doc, 0.9)]
    llm.invoke.return_value = Mock(content="answer")

    span = Mock()
    langfuse_client = Mock()
    langfuse_client.start_as_current_observation.return_value.__enter__ = Mock(
        return_value=span
    )
    langfuse_client.start_as_current_observation.return_value.__exit__ = Mock(
        return_value=False
    )
    handler = Mock()

    traced = TracedRAGChain(chain, langfuse_client, handler)
    traced.invoke("test query")

    langfuse_client.start_as_current_observation.assert_called_once()
    call_kwargs = langfuse_client.start_as_current_observation.call_args[1]
    assert call_kwargs["name"] == "retrieval"
    span.update.assert_called_once()
    update_kwargs = span.update.call_args[1]
    assert update_kwargs["output"]["document_count"] == 1
    assert update_kwargs["output"]["sources"] == ["a.md"]


def test_traced_rag_chain_forwards_callback_handler_to_llm() -> None:
    """Test that the LLM receives config with callbacks containing the LangFuse handler."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_template="Context: {context}\n\nQ: {query}\nA:",
    )

    doc = Document(page_content="content", metadata={"source": "a.md"})
    retriever.retrieve.return_value = [(doc, 0.9)]
    llm.invoke.return_value = Mock(content="answer")

    langfuse_client = Mock()
    span = Mock()
    langfuse_client.start_as_current_observation.return_value.__enter__ = Mock(
        return_value=span
    )
    langfuse_client.start_as_current_observation.return_value.__exit__ = Mock(
        return_value=False
    )
    handler = Mock()

    traced = TracedRAGChain(chain, langfuse_client, handler)
    traced.invoke("test query")

    llm.invoke.assert_called_once()
    call_kwargs = llm.invoke.call_args[1]
    assert call_kwargs["config"]["callbacks"] == [handler]


def test_create_traced_rag_chain_returns_plain_chain_when_tracing_disabled() -> None:
    """Test that the factory returns the original RAGChain when tracing is disabled."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_template="Context: {context}\n\nQ: {query}\nA:",
    )

    result = create_traced_rag_chain(chain, tracing_enabled=False)

    assert result is chain


@patch("langfuse.get_client")
@patch("langfuse.langchain.CallbackHandler")
def test_create_traced_rag_chain_logs_info_when_tracing_enabled(
    _mock_handler: Mock, _mock_get_client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    """Enabling tracing logs an INFO message indicating LangFuse tracing is active."""
    chain = RAGChain(retriever=Mock(), llm=Mock(), prompt_template="{context}{query}")

    with caplog.at_level(logging.DEBUG, logger="epistemon.tracing"):
        create_traced_rag_chain(chain, tracing_enabled=True)

    info_messages = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("tracing" in m.message.lower() for m in info_messages)
