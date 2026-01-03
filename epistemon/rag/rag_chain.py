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
