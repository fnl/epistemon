"""Vector store manager abstraction for different vector store implementations."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.vectorstores import DuckDB
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_qdrant import QdrantVectorStore
from langchain_weaviate import WeaviateVectorStore


class VectorStoreManager(ABC):
    """Abstract base class for managing vector store operations."""

    def __init__(self, vector_store: VectorStore, base_directory: Path) -> None:
        self.vector_store = vector_store
        self.base_directory = base_directory

    @abstractmethod
    def get_indexed_files(self) -> dict[str, float]:
        """Get all indexed files with their modification times."""
        pass

    @abstractmethod
    def remove_documents_by_source(self, sources: set[str]) -> None:
        """Remove documents matching the given source paths."""
        pass

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        self.vector_store.add_documents(documents)

    def update_documents_for_file(
        self, file_path: Path, new_chunks: list[Document]
    ) -> None:
        """Update documents for a specific file."""
        relative_source = str(file_path.relative_to(self.base_directory))
        self.remove_documents_by_source({relative_source})
        self.add_documents(new_chunks)


class ChromaVectorStoreManager(VectorStoreManager):
    """Manager for Chroma vector stores."""

    def __init__(self, vector_store: Chroma, base_directory: Path) -> None:
        super().__init__(vector_store, base_directory)
        self.chroma_store = vector_store

    def get_indexed_files(self) -> dict[str, float]:
        indexed_files: dict[str, float] = {}
        result = self.chroma_store.get()

        if result and "metadatas" in result:
            for metadata in result["metadatas"]:
                if metadata and "source" in metadata and "last_modified" in metadata:
                    source = metadata["source"]
                    mtime = metadata["last_modified"]
                    indexed_files[str(self.base_directory / source)] = mtime

        return indexed_files

    def remove_documents_by_source(self, sources: set[str]) -> None:
        result = self.chroma_store.get()

        if not result or "ids" not in result or "metadatas" not in result:
            return

        doc_ids_to_remove = [
            doc_id
            for doc_id, metadata in zip(
                result["ids"], result["metadatas"], strict=False
            )
            if metadata and metadata.get("source") in sources
        ]

        if doc_ids_to_remove:
            self.chroma_store.delete(doc_ids_to_remove)


class InMemoryVectorStoreManager(VectorStoreManager):
    """Manager for in-memory vector stores."""

    def __init__(self, vector_store: InMemoryVectorStore, base_directory: Path) -> None:
        super().__init__(vector_store, base_directory)
        self.inmemory_store = vector_store

    def get_indexed_files(self) -> dict[str, float]:
        indexed_files: dict[str, float] = {}

        for _doc_id, doc_dict in self.inmemory_store.store.items():
            source = doc_dict["metadata"]["source"]
            mtime = doc_dict["metadata"]["last_modified"]
            indexed_files[str(self.base_directory / source)] = mtime

        return indexed_files

    def remove_documents_by_source(self, sources: set[str]) -> None:
        doc_ids_to_remove = [
            doc_id
            for doc_id, doc_dict in self.inmemory_store.store.items()
            if doc_dict["metadata"]["source"] in sources
        ]

        for doc_id in doc_ids_to_remove:
            del self.inmemory_store.store[doc_id]


class WeaviateVectorStoreManager(VectorStoreManager):
    """Manager for Weaviate vector stores."""

    def __init__(self, vector_store: WeaviateVectorStore, base_directory: Path) -> None:
        super().__init__(vector_store, base_directory)
        self.weaviate_store = vector_store

    def get_indexed_files(self) -> dict[str, float]:
        indexed_files: dict[str, float] = {}
        collection = self.weaviate_store._client.collections.get(
            self.weaviate_store._index_name
        )

        for item in collection.iterator():
            if item.properties and "source" in item.properties:
                source = str(item.properties["source"])
                mtime_val = item.properties.get("last_modified", 0.0)
                if isinstance(mtime_val, (int, float)):
                    mtime = float(mtime_val)
                else:
                    mtime = 0.0
                indexed_files[str(self.base_directory / source)] = mtime

        return indexed_files

    def remove_documents_by_source(self, sources: set[str]) -> None:
        collection = self.weaviate_store._client.collections.get(
            self.weaviate_store._index_name
        )

        for item in collection.iterator():
            if (
                item.properties
                and "source" in item.properties
                and item.properties["source"] in sources
            ):
                collection.data.delete_by_id(item.uuid)


class QdrantVectorStoreManager(VectorStoreManager):
    """Manager for Qdrant vector stores."""

    def __init__(self, vector_store: QdrantVectorStore, base_directory: Path) -> None:
        super().__init__(vector_store, base_directory)
        self.qdrant_store = vector_store

    def get_indexed_files(self) -> dict[str, float]:
        indexed_files: dict[str, float] = {}

        points, _next_offset = self.qdrant_store.client.scroll(
            collection_name=self.qdrant_store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            if point.payload and "metadata" in point.payload:
                metadata = point.payload["metadata"]
                if "source" in metadata:
                    source = metadata["source"]
                    mtime = float(metadata.get("last_modified", 0.0))
                    indexed_files[str(self.base_directory / source)] = mtime

        return indexed_files

    def remove_documents_by_source(self, sources: set[str]) -> None:
        points, _next_offset = self.qdrant_store.client.scroll(
            collection_name=self.qdrant_store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        ids_to_remove = []
        for point in points:
            if (
                point.payload
                and "metadata" in point.payload
                and point.payload["metadata"].get("source") in sources
            ):
                ids_to_remove.append(point.id)

        if ids_to_remove:
            self.qdrant_store.client.delete(
                collection_name=self.qdrant_store.collection_name,
                points_selector=ids_to_remove,
            )


class DuckDBVectorStoreManager(VectorStoreManager):
    """Manager for DuckDB vector stores."""

    def __init__(
        self, vector_store: "VectorStore", base_directory: Path  # noqa: F821
    ) -> None:
        super().__init__(vector_store, base_directory)
        self.duckdb_store = vector_store

    def get_indexed_files(self) -> dict[str, float]:
        indexed_files: dict[str, float] = {}

        if isinstance(self.duckdb_store, DuckDB):
            result = self.duckdb_store._connection.execute(
                f"SELECT {self.duckdb_store._id_key}, metadata FROM {self.duckdb_store._table_name}"  # noqa: S608
            ).fetchall()

            for _doc_id, metadata_json in result:
                metadata = json.loads(metadata_json)
                if "source" in metadata:
                    source = metadata["source"]
                    mtime = float(metadata.get("last_modified", 0.0))
                    indexed_files[str(self.base_directory / source)] = mtime

        return indexed_files

    def remove_documents_by_source(self, sources: set[str]) -> None:
        if isinstance(self.duckdb_store, DuckDB):
            result = self.duckdb_store._connection.execute(
                f"SELECT {self.duckdb_store._id_key}, metadata FROM {self.duckdb_store._table_name}"  # noqa: S608
            ).fetchall()

            ids_to_remove = []
            for doc_id, metadata_json in result:
                metadata = json.loads(metadata_json)
                if metadata.get("source") in sources:
                    ids_to_remove.append(doc_id)

            if ids_to_remove:
                placeholders = ",".join(["?" for _ in ids_to_remove])
                self.duckdb_store._connection.execute(
                    f"DELETE FROM {self.duckdb_store._table_name} WHERE {self.duckdb_store._id_key} IN ({placeholders})",  # noqa: S608
                    ids_to_remove,
                )


def create_vector_store_manager(
    vector_store: VectorStore, base_directory: Path
) -> VectorStoreManager:
    """Factory function to create the appropriate vector store manager."""
    if isinstance(vector_store, Chroma):
        return ChromaVectorStoreManager(vector_store, base_directory)
    elif isinstance(vector_store, InMemoryVectorStore):
        return InMemoryVectorStoreManager(vector_store, base_directory)
    elif isinstance(vector_store, WeaviateVectorStore):
        return WeaviateVectorStoreManager(vector_store, base_directory)
    elif isinstance(vector_store, QdrantVectorStore):
        return QdrantVectorStoreManager(vector_store, base_directory)
    elif isinstance(vector_store, DuckDB):
        return DuckDBVectorStoreManager(vector_store, base_directory)
    else:
        raise ValueError(f"Unsupported vector store type: {type(vector_store)}")
