"""RAG chain for question answering over documents."""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RAGResponse:
    """Response from the RAG chain."""

    answer: str
    source_documents: list[Document]
    query: str
