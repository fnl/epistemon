from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from epistemon.indexing.vector_store_manager import (
    InMemoryVectorStoreManager,
    create_vector_store_manager,
)


@pytest.fixture
def base_directory() -> Path:
    return Path("tests/data")


@pytest.fixture
def inmemory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(FakeEmbeddings(size=384))


def test_inmemory_manager_get_indexed_files_with_multiple_files(
    inmemory_store: InMemoryVectorStore, base_directory: Path
) -> None:
    docs = [
        Document(
            page_content="Content A",
            metadata={"source": "file_a.md", "last_modified": 1234567890.0},
        ),
        Document(
            page_content="Content B",
            metadata={"source": "file_b.md", "last_modified": 1234567900.0},
        ),
        Document(
            page_content="More A",
            metadata={"source": "file_a.md", "last_modified": 1234567890.0},
        ),
    ]
    inmemory_store.add_documents(docs)
    manager = InMemoryVectorStoreManager(inmemory_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert len(indexed_files) == 2
    assert str(base_directory / "file_a.md") in indexed_files
    assert str(base_directory / "file_b.md") in indexed_files
    assert indexed_files[str(base_directory / "file_a.md")] == 1234567890.0
    assert indexed_files[str(base_directory / "file_b.md")] == 1234567900.0


def test_inmemory_manager_get_indexed_files_empty_store(
    inmemory_store: InMemoryVectorStore, base_directory: Path
) -> None:
    manager = InMemoryVectorStoreManager(inmemory_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert indexed_files == {}


def test_create_vector_store_manager_returns_inmemory_manager(
    inmemory_store: InMemoryVectorStore, base_directory: Path
) -> None:
    manager = create_vector_store_manager(inmemory_store, base_directory)

    assert isinstance(manager, InMemoryVectorStoreManager)
