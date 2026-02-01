"""LangFuse tracing module for the RAG pipeline."""

import logging
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from epistemon.retrieval.rag_chain import RAGChain, RAGChainProtocol, RAGResponse

logger = logging.getLogger(__name__)


class TracedRAGChain:
    """RAG chain wrapper that adds LangFuse tracing to retrieval and generation."""

    def __init__(
        self,
        chain: RAGChain,
        langfuse_client: Langfuse,
        callback_handler: BaseCallbackHandler,
    ) -> None:
        self.chain = chain
        self.langfuse_client = langfuse_client
        self.callback_handler = callback_handler

    def invoke(
        self, query: str, k: Optional[int] = None, base_url: str = ""
    ) -> RAGResponse:
        """Invoke the RAG chain with LangFuse tracing."""
        source_documents = self._retrieve_with_span(query, k)
        logger.debug(
            "Traced query: '%s', retrieved %d document(s)",
            query,
            len(source_documents),
        )

        if not source_documents:
            return RAGResponse(
                answer="No relevant documents were found to answer your question.",
                source_documents=[],
                query=query,
            )

        context = self.chain.format_context(source_documents, base_url=base_url)
        prompt = self.chain.prompt_template.format(context=context, query=query)
        response = self.chain.llm.invoke(
            prompt, config={"callbacks": [self.callback_handler]}
        )

        return RAGResponse(
            answer=response.content,
            source_documents=source_documents,
            query=query,
        )

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
    chain: RAGChain, *, tracing_enabled: bool
) -> RAGChainProtocol:
    """Create a RAG chain optionally wrapped with LangFuse tracing.

    Args:
        chain: The base RAGChain to optionally wrap
        tracing_enabled: Whether tracing is enabled

    Returns:
        The original chain if tracing is disabled, or a TracedRAGChain wrapper
    """
    if not tracing_enabled:
        logger.debug("Tracing is disabled")
        return chain

    langfuse_client = get_client()
    handler = CallbackHandler()
    logger.info("LangFuse tracing is active")
    return TracedRAGChain(chain, langfuse_client, handler)
