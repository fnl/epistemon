"""Embedding and indexing functionality."""

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def embed_and_index(chunks: list[Document]) -> InMemoryVectorStore:
    embeddings = FakeEmbeddings(size=384)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    return vector_store
