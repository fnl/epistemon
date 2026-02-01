"""Tests for the tracing module."""

import logging
import typing
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from epistemon.retrieval.rag_chain import RAGChain
from epistemon.tracing import TracedRAGChain, create_traced_rag_chain


def test_traced_rag_chain_constructor_uses_concrete_types() -> None:
    """TracedRAGChain constructor parameters use concrete types instead of Any."""
    hints = typing.get_type_hints(TracedRAGChain.__init__)
    assert hints["langfuse_client"] is not typing.Any
    assert hints["callback_handler"] is not typing.Any


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

    retrieval_span = Mock()
    retrieval_cm = Mock()
    retrieval_cm.__enter__ = Mock(return_value=retrieval_span)
    retrieval_cm.__exit__ = Mock(return_value=False)
    generation_cm = Mock()
    generation_cm.__enter__ = Mock(return_value=Mock())
    generation_cm.__exit__ = Mock(return_value=False)

    langfuse_client = Mock()
    langfuse_client.start_as_current_observation.side_effect = [
        retrieval_cm,
        generation_cm,
    ]
    handler = Mock()

    traced = TracedRAGChain(chain, langfuse_client, handler)
    traced.invoke("test query")

    retrieval_call = langfuse_client.start_as_current_observation.call_args_list[0][1]
    assert retrieval_call["name"] == "retrieval"
    retrieval_span.update.assert_called_once()
    update_kwargs = retrieval_span.update.call_args[1]
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
    cm = Mock()
    cm.__enter__ = Mock(return_value=Mock())
    cm.__exit__ = Mock(return_value=False)
    langfuse_client.start_as_current_observation.return_value = cm
    handler = Mock()

    traced = TracedRAGChain(chain, langfuse_client, handler)
    traced.invoke("test query")

    llm.invoke.assert_called_once()
    call_kwargs = llm.invoke.call_args[1]
    assert call_kwargs["config"]["callbacks"] == [handler]


def test_traced_rag_chain_creates_generation_span() -> None:
    """Answer generation is wrapped in a LangFuse span with query and document count."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_template="Context: {context}\n\nQ: {query}\nA:",
    )

    docs = [
        Document(page_content="a", metadata={"source": "a.md"}),
        Document(page_content="b", metadata={"source": "b.md"}),
    ]
    retriever.retrieve.return_value = [(d, 0.9) for d in docs]
    llm.invoke.return_value = Mock(content="short answer")

    retrieval_span = Mock()
    generation_span = Mock()
    retrieval_cm = Mock()
    retrieval_cm.__enter__ = Mock(return_value=retrieval_span)
    retrieval_cm.__exit__ = Mock(return_value=False)
    generation_cm = Mock()
    generation_cm.__enter__ = Mock(return_value=generation_span)
    generation_cm.__exit__ = Mock(return_value=False)

    langfuse_client = Mock()
    langfuse_client.start_as_current_observation.side_effect = [
        retrieval_cm,
        generation_cm,
    ]

    traced = TracedRAGChain(chain, langfuse_client, Mock())
    traced.invoke("test query")

    calls = langfuse_client.start_as_current_observation.call_args_list
    generation_call = calls[1][1]
    assert generation_call["name"] == "generation"
    assert generation_call["input"]["query"] == "test query"
    assert generation_call["input"]["document_count"] == 2

    gen_output = generation_span.update.call_args[1]["output"]
    assert gen_output["answer_length"] == len("short answer")


def test_langfuse_imports_are_at_module_level() -> None:
    """Verify langfuse symbols are importable directly from epistemon.tracing."""
    from epistemon import tracing

    assert hasattr(tracing, "get_client")
    assert hasattr(tracing, "CallbackHandler")


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


@patch("epistemon.tracing.get_client")
@patch("epistemon.tracing.CallbackHandler")
def test_create_traced_rag_chain_logs_info_when_tracing_enabled(
    _mock_handler: Mock, _mock_get_client: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    """Enabling tracing logs an INFO message indicating LangFuse tracing is active."""
    chain = RAGChain(retriever=Mock(), llm=Mock(), prompt_template="{context}{query}")

    with caplog.at_level(logging.DEBUG, logger="epistemon.tracing"):
        create_traced_rag_chain(chain, tracing_enabled=True)

    info_messages = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("tracing" in m.message.lower() for m in info_messages)


def test_create_traced_rag_chain_logs_debug_when_tracing_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Disabling tracing logs a DEBUG message indicating tracing is disabled."""
    chain = RAGChain(retriever=Mock(), llm=Mock(), prompt_template="{context}{query}")

    with caplog.at_level(logging.DEBUG, logger="epistemon.tracing"):
        create_traced_rag_chain(chain, tracing_enabled=False)

    debug_messages = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("tracing" in m.message.lower() for m in debug_messages)


def test_traced_rag_chain_invoke_logs_query_and_document_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each traced invocation logs a DEBUG message with the query and document count."""
    retriever = Mock()
    llm = Mock()
    chain = RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_template="Context: {context}\n\nQ: {query}\nA:",
    )

    docs = [
        Document(page_content="a", metadata={"source": "a.md"}),
        Document(page_content="b", metadata={"source": "b.md"}),
    ]
    retriever.retrieve.return_value = [(d, 0.9) for d in docs]
    llm.invoke.return_value = Mock(content="answer")

    langfuse_client = Mock()
    cm = Mock()
    cm.__enter__ = Mock(return_value=Mock())
    cm.__exit__ = Mock(return_value=False)
    langfuse_client.start_as_current_observation.return_value = cm

    traced = TracedRAGChain(chain, langfuse_client, Mock())

    with caplog.at_level(logging.DEBUG, logger="epistemon.tracing"):
        traced.invoke("what is Python?")

    debug_messages = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(
        "what is python?" in m.message.lower() and "2" in m.message
        for m in debug_messages
    )
