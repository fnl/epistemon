"""Semantic search functionality."""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def search(vector_store: VectorStore, query: str, limit: int) -> list[Document]:
    return vector_store.similarity_search(query, k=limit)
