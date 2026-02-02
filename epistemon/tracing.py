"""LangFuse tracing module for the RAG pipeline."""

import logging
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from epistemon.retrieval.hybrid_retriever import (
    BM25RetrieverProtocol,
    HybridRetriever,
)
from epistemon.retrieval.rag_chain import RAGChain, RAGChainProtocol, RAGResponse

logger = logging.getLogger(__name__)


class TracedBM25Retriever:
    """Wraps a BM25 retriever to record a LangFuse span for each search."""

    def __init__(
        self,
        retriever: BM25RetrieverProtocol,
        langfuse_client: Langfuse,
    ) -> None:
        self.retriever = retriever
        self.langfuse_client = langfuse_client

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        with self.langfuse_client.start_as_current_observation(
            as_type="span",
            name="bm25-search",
            input={"query": query},
        ) as span:
            results = self.retriever.retrieve(query)
            span.update(
                output={
                    "document_count": len(results),
                    "sources": [
                        doc.metadata.get("source", "Unknown") for doc, _ in results
                    ],
                }
            )
        return results


class TracedRAGChain:
    """RAG chain wrapper that adds LangFuse tracing to retrieval and generation."""

    def __init__(
        self,
        chain: RAGChain,
        langfuse_client: Langfuse,
        callback_handler: BaseCallbackHandler,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.chain = chain
        self.langfuse_client = langfuse_client
        self.callback_handler = callback_handler
        self.embedding_model = embedding_model

    def invoke(
        self, query: str, k: Optional[int] = None, base_url: str = ""
    ) -> RAGResponse:
        """Invoke the RAG chain with LangFuse tracing."""
        with self.langfuse_client.start_as_current_observation(
            as_type="span",
            name="rag-pipeline",
            input={"query": query},
        ) as trace:
            if self.embedding_model is not None:
                self._record_embedding(query)
            source_documents = self._retrieve_with_span(query, k)
            logger.debug(
                "Traced query: '%s', retrieved %d document(s)",
                query,
                len(source_documents),
            )

            if not source_documents:
                trace.update(
                    output={"answer_length": 0, "document_count": 0},
                )
                return RAGResponse(
                    answer="No relevant documents were found to answer your question.",
                    source_documents=[],
                    query=query,
                )

            response = self._generate_with_span(query, source_documents, base_url)
            trace.update(
                output={
                    "answer_length": len(response.answer),
                    "document_count": len(source_documents),
                },
            )
            return response

    def _record_embedding(self, query: str) -> None:
        with self.langfuse_client.start_as_current_observation(
            as_type="embedding",
            name="query-embedding",
            model=self.embedding_model,
            input={"query": query},
        ):
            pass

    def _generate_with_span(
        self, query: str, source_documents: list[Document], base_url: str
    ) -> RAGResponse:
        with self.langfuse_client.start_as_current_observation(
            as_type="span",
            name="generation",
            input={"query": query, "document_count": len(source_documents)},
        ) as span:
            response = self.chain.generate_answer(
                query,
                source_documents,
                base_url=base_url,
                config={"callbacks": [self.callback_handler]},
            )
            span.update(output={"answer_length": len(response.answer)})
        return response

    def _retrieve_with_span(self, query: str, k: Optional[int]) -> list[Document]:
        with self.langfuse_client.start_as_current_observation(
            as_type="span",
            name="retrieval",
            input={"query": query},
        ) as span:
            source_documents = self.chain.retrieve_documents(query)
            if k is not None and k > 0:
                source_documents = source_documents[:k]
            span.update(
                output={
                    "document_count": len(source_documents),
                    "sources": [
                        doc.metadata.get("source", "Unknown")
                        for doc in source_documents
                    ],
                }
            )
        return source_documents


def create_traced_rag_chain(
    chain: RAGChain,
    *,
    tracing_enabled: bool,
    embedding_model: Optional[str] = None,
) -> RAGChainProtocol:
    """Create a RAG chain optionally wrapped with LangFuse tracing.

    Args:
        chain: The base RAGChain to optionally wrap
        tracing_enabled: Whether tracing is enabled
        embedding_model: Name of the embedding model for trace metadata

    Returns:
        The original chain if tracing is disabled, or a TracedRAGChain wrapper
    """
    if not tracing_enabled:
        logger.debug("Tracing is disabled")
        return chain

    langfuse_client = get_client()
    handler = CallbackHandler()
    _install_retriever_tracing(chain, langfuse_client)
    logger.info("LangFuse tracing is active")
    return TracedRAGChain(
        chain, langfuse_client, handler, embedding_model=embedding_model
    )


def _install_retriever_tracing(chain: RAGChain, langfuse_client: Langfuse) -> None:
    """Replace sub-retrievers with traced wrappers when using HybridRetriever."""
    retriever = chain.retriever
    if not isinstance(retriever, HybridRetriever):
        return
    retriever.bm25_retriever = TracedBM25Retriever(
        retriever.bm25_retriever, langfuse_client
    )
