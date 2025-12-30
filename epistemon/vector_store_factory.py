"""Vector store factory for creating the appropriate vector store based on configuration."""

from pathlib import Path
from typing import cast

import duckdb
import weaviate
from langchain_chroma import Chroma
from langchain_community.vectorstores import DuckDB
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_weaviate import WeaviateVectorStore

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
    elif config.vector_store_type == "weaviate":
        Path(config.vector_store_path).mkdir(parents=True, exist_ok=True)
        client = weaviate.connect_to_embedded(
            persistence_data_path=config.vector_store_path,
            version="1.28.8",
        )
        weaviate_store: VectorStore = WeaviateVectorStore(
            client=client,
            index_name="Epistemon",
            text_key="text",
            embedding=embeddings,
        )
        return weaviate_store
    elif config.vector_store_type == "qdrant":
        qdrant_store: VectorStore = QdrantVectorStore.from_documents(
            [],
            embeddings,
            path=config.vector_store_path,
            collection_name="epistemon",
        )
        return qdrant_store
    elif config.vector_store_type == "duckdb":
        Path(config.vector_store_path).mkdir(parents=True, exist_ok=True)
        db_file = f"{config.vector_store_path}/epistemon.duckdb"
        conn = duckdb.connect(database=db_file, read_only=False)

        duckdb_store: VectorStore = DuckDB(
            embedding=embeddings,
            connection=conn,
        )
        return duckdb_store
    else:
        return InMemoryVectorStore(embeddings)
