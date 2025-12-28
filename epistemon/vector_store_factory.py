"""Vector store factory for creating the appropriate vector store based on configuration."""

from typing import cast

from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from epistemon.config import Configuration


def create_vector_store(config: Configuration) -> VectorStore:
    embeddings = FakeEmbeddings(size=384)

    if config.vector_store_type == "chroma":
        chroma_store: VectorStore = cast(
            VectorStore,
            Chroma(
                collection_name="epistemon",
                embedding_function=embeddings,
                persist_directory=config.vector_store_path,
            ),
        )
        return chroma_store
    else:
        return InMemoryVectorStore(embeddings)
