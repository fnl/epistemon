"""Vector store factory for creating the appropriate vector store based on configuration."""

from typing import cast

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from epistemon.config import Configuration


def create_embeddings(config: Configuration) -> Embeddings:
    if config.embedding_provider == "fake":
        return FakeEmbeddings(size=384)
    elif config.embedding_provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=config.embedding_model)  # type: ignore[no-any-return]
    elif config.embedding_provider == "openai":
        return OpenAIEmbeddings(model=config.embedding_model)  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


def create_vector_store(config: Configuration) -> VectorStore:
    embeddings = create_embeddings(config)

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
