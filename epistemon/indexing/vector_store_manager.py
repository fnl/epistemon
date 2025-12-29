"""Vector store manager abstraction for different vector store implementations."""

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore


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


def create_vector_store_manager(
    vector_store: VectorStore, base_directory: Path
) -> VectorStoreManager:
    """Factory function to create the appropriate vector store manager."""
    if isinstance(vector_store, Chroma):
        return ChromaVectorStoreManager(vector_store, base_directory)
    elif isinstance(vector_store, InMemoryVectorStore):
        return InMemoryVectorStoreManager(vector_store, base_directory)
    else:
        raise ValueError(f"Unsupported vector store type: {type(vector_store)}")
