"""RAG chain for question answering over documents."""

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class RAGResponse:
    """Response from the RAG chain."""

    answer: str
    source_documents: list[Document]
    query: str


class RAGChain:
    """RAG chain that retrieves documents and generates answers using an LLM."""

    def __init__(self, retriever: Any, llm: Any) -> None:
        self.retriever = retriever
        self.llm = llm

    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into context for the LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string with document content and metadata
        """
        if not documents:
            return ""

        formatted_docs = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            formatted_docs.append(f"Source: {source}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_docs)
