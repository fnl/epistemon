from pathlib import Path

import duckdb
import pytest
import weaviate
from langchain_chroma import Chroma
from langchain_community.vectorstores import DuckDB
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore
from langchain_weaviate import WeaviateVectorStore

from epistemon.indexing.vector_store_manager import (
    ChromaVectorStoreManager,
    DuckDBVectorStoreManager,
    InMemoryVectorStoreManager,
    QdrantVectorStoreManager,
    WeaviateVectorStoreManager,
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


@pytest.fixture
def chroma_store(tmp_path: Path) -> Chroma:
    return Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=384),
        persist_directory=str(tmp_path / "chroma_db"),
    )


def test_chroma_manager_get_indexed_files_with_multiple_files(
    chroma_store: Chroma, base_directory: Path
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
    chroma_store.add_documents(docs)
    manager = ChromaVectorStoreManager(chroma_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert len(indexed_files) == 2
    assert str(base_directory / "file_a.md") in indexed_files
    assert str(base_directory / "file_b.md") in indexed_files
    assert indexed_files[str(base_directory / "file_a.md")] == 1234567890.0
    assert indexed_files[str(base_directory / "file_b.md")] == 1234567900.0


def test_chroma_manager_get_indexed_files_empty_store(
    chroma_store: Chroma, base_directory: Path
) -> None:
    manager = ChromaVectorStoreManager(chroma_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert indexed_files == {}


def test_create_vector_store_manager_returns_chroma_manager(
    chroma_store: Chroma, base_directory: Path
) -> None:
    manager = create_vector_store_manager(chroma_store, base_directory)

    assert isinstance(manager, ChromaVectorStoreManager)


@pytest.fixture
def weaviate_store(tmp_path: Path) -> WeaviateVectorStore:
    client = weaviate.connect_to_embedded(
        persistence_data_path=str(tmp_path / "weaviate_db"),
        version="1.28.8",
    )
    store = WeaviateVectorStore(
        client=client,
        index_name="TestCollection",
        text_key="text",
        embedding=FakeEmbeddings(size=384),
    )
    yield store
    client.close()


@pytest.mark.slow
def test_weaviate_manager_get_indexed_files_with_multiple_files(
    weaviate_store: WeaviateVectorStore, base_directory: Path
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
    weaviate_store.add_documents(docs)
    manager = WeaviateVectorStoreManager(weaviate_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert len(indexed_files) == 2
    assert str(base_directory / "file_a.md") in indexed_files
    assert str(base_directory / "file_b.md") in indexed_files


@pytest.mark.slow
def test_weaviate_manager_get_indexed_files_empty_store(
    weaviate_store: WeaviateVectorStore, base_directory: Path
) -> None:
    manager = WeaviateVectorStoreManager(weaviate_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert indexed_files == {}


@pytest.mark.slow
def test_create_vector_store_manager_returns_weaviate_manager(
    weaviate_store: WeaviateVectorStore, base_directory: Path
) -> None:
    manager = create_vector_store_manager(weaviate_store, base_directory)

    assert isinstance(manager, WeaviateVectorStoreManager)


@pytest.fixture
def qdrant_store(tmp_path: Path) -> QdrantVectorStore:
    return QdrantVectorStore.from_documents(
        [],
        FakeEmbeddings(size=384),
        path=str(tmp_path / "qdrant_db"),
        collection_name="test_collection",
    )


def test_qdrant_manager_get_indexed_files_with_multiple_files(
    qdrant_store: QdrantVectorStore, base_directory: Path
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
    qdrant_store.add_documents(docs)
    manager = QdrantVectorStoreManager(qdrant_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert len(indexed_files) == 2
    assert str(base_directory / "file_a.md") in indexed_files
    assert str(base_directory / "file_b.md") in indexed_files


def test_qdrant_manager_get_indexed_files_empty_store(
    qdrant_store: QdrantVectorStore, base_directory: Path
) -> None:
    manager = QdrantVectorStoreManager(qdrant_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert indexed_files == {}


def test_create_vector_store_manager_returns_qdrant_manager(
    qdrant_store: QdrantVectorStore, base_directory: Path
) -> None:
    manager = create_vector_store_manager(qdrant_store, base_directory)

    assert isinstance(manager, QdrantVectorStoreManager)


@pytest.fixture
def duckdb_store(tmp_path: Path) -> DuckDB:
    db_file = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(database=db_file, read_only=False)
    return DuckDB(embedding=FakeEmbeddings(size=384), connection=conn)


def test_duckdb_manager_get_indexed_files_with_multiple_files(
    duckdb_store: DuckDB, base_directory: Path
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
    duckdb_store.add_documents(docs)
    manager = DuckDBVectorStoreManager(duckdb_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert len(indexed_files) == 2
    assert str(base_directory / "file_a.md") in indexed_files
    assert str(base_directory / "file_b.md") in indexed_files


def test_duckdb_manager_get_indexed_files_empty_store(
    duckdb_store: DuckDB, base_directory: Path
) -> None:
    manager = DuckDBVectorStoreManager(duckdb_store, base_directory)

    indexed_files = manager.get_indexed_files()

    assert indexed_files == {}


def test_create_vector_store_manager_returns_duckdb_manager(
    duckdb_store: DuckDB, base_directory: Path
) -> None:
    manager = create_vector_store_manager(duckdb_store, base_directory)

    assert isinstance(manager, DuckDBVectorStoreManager)
