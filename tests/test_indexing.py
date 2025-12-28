from pathlib import Path

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from epistemon.config import Configuration
from epistemon.indexing import (
    collect_markdown_files,
    detect_file_changes,
    load_and_chunk_markdown,
    remove_deleted_embeddings,
    update_embeddings_for_file,
)
from epistemon.vector_store_factory import create_vector_store


@pytest.fixture
def test_data_directory() -> Path:
    return Path("tests/data")


def files_that_produce_chunks(
    files: list[Path], chunk_size: int = 500, chunk_overlap: int = 100
) -> list[Path]:
    return [
        f
        for f in files
        if load_and_chunk_markdown(
            f, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    ]


def create_test_vector_store(
    files: list[Path],
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    old_mtime: bool = False,
) -> InMemoryVectorStore:
    all_chunks = []
    for file in files:
        chunks = load_and_chunk_markdown(
            file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_directory=directory,
        )
        if old_mtime:
            for chunk in chunks:
                chunk.metadata["last_modified"] = 0.0
        all_chunks.extend(chunks)

    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    if all_chunks:
        vector_store.add_documents(all_chunks)
    return vector_store


def test_collect_markdown_files_non_recursively() -> None:
    directory = Path("tests/data")

    markdown_files = collect_markdown_files(directory, recursive=False)

    assert len(markdown_files) > 0
    assert all(f.suffix == ".md" for f in markdown_files)
    assert all(f.parent == directory for f in markdown_files)


def test_collect_markdown_files_recursively() -> None:
    directory = Path("tests/data")

    markdown_files = collect_markdown_files(directory)

    assert len(markdown_files) > 0
    assert all(f.suffix == ".md" for f in markdown_files)

    paths = [str(f.relative_to(directory)) for f in markdown_files]
    assert any("subdir" in path for path in paths)


def test_load_and_chunk_markdown() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)


def test_load_and_chunk_markdown_with_relative_source() -> None:
    base_dir = Path("tests/data")
    test_file = base_dir / "subdir" / "nested_doc.md"
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_dir
    )

    assert all(chunk.metadata["source"] == "subdir/nested_doc.md" for chunk in chunks)


def test_load_and_chunk_markdown_includes_modification_time() -> None:
    test_file = Path("tests/data/sample.md")
    expected_mtime = test_file.stat().st_mtime

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert all("last_modified" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["last_modified"] == expected_mtime for chunk in chunks)


def test_load_and_chunk_markdown_handles_empty_file() -> None:
    test_file = Path("tests/data/empty.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 0


def test_load_and_chunk_markdown_logs_warning_for_empty_file(
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_file = Path("tests/data/empty.md")

    load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert any(
        record.levelname == "WARNING" and "empty.md" in record.message
        for record in caplog.records
    )


def test_load_and_chunk_markdown_handles_whitespace_only_file() -> None:
    test_file = Path("tests/data/whitespace_only.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 0


def test_load_and_chunk_markdown_handles_malformed_markdown() -> None:
    test_file = Path("tests/data/malformed.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)


def test_detect_new_files(test_data_directory: Path) -> None:
    all_files = collect_markdown_files(test_data_directory)
    files_with_chunks = files_that_produce_chunks(all_files)
    vector_store = create_test_vector_store(files_with_chunks[:2], test_data_directory)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes.new) == len(files_with_chunks) - 2
    assert len(changes.modified) == 0
    assert len(changes.deleted) == 0


def test_detect_modified_files(test_data_directory: Path) -> None:
    all_files = collect_markdown_files(test_data_directory, recursive=False)
    files_with_chunks = files_that_produce_chunks(all_files)
    files = files_with_chunks[:2]
    vector_store = create_test_vector_store(files, test_data_directory, old_mtime=True)

    changes = detect_file_changes(test_data_directory, vector_store)

    all_files_with_chunks = files_that_produce_chunks(
        collect_markdown_files(test_data_directory)
    )
    assert len(changes.new) == len(all_files_with_chunks) - 2
    assert len(changes.modified) == 2
    assert len(changes.deleted) == 0
    assert all(str(f) in [str(p) for p in changes.modified] for f in files)


def test_retrieve_indexed_files_from_vector_store(test_data_directory: Path) -> None:
    files = collect_markdown_files(test_data_directory, recursive=False)[:2]
    vector_store = create_test_vector_store(files, test_data_directory)

    indexed_files: dict[str, float] = {}
    for _doc_id, doc_dict in vector_store.store.items():
        source = doc_dict["metadata"]["source"]
        mtime = doc_dict["metadata"]["last_modified"]
        indexed_files[str(test_data_directory / source)] = mtime

    assert len(indexed_files) == len(files)
    for file in files:
        assert str(file) in indexed_files


def test_skip_unchanged_files(test_data_directory: Path) -> None:
    all_files = collect_markdown_files(test_data_directory)
    vector_store = create_test_vector_store(all_files, test_data_directory)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes.new) == 0
    assert len(changes.modified) == 0
    assert len(changes.deleted) == 0


def test_skip_whitespace_only_files(test_data_directory: Path) -> None:
    files_with_content = [
        f
        for f in collect_markdown_files(test_data_directory)
        if f.name not in ["empty.md", "whitespace_only.md"]
    ]
    vector_store = create_test_vector_store(files_with_content, test_data_directory)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes.new) == 0
    assert len(changes.modified) == 0
    assert len(changes.deleted) == 0


def test_add_documents_to_vector_store() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    stored_docs = vector_store.similarity_search("LangChain", k=3)

    assert len(stored_docs) > 0


def test_detect_deleted_files(test_data_directory: Path) -> None:
    all_files = collect_markdown_files(test_data_directory)
    files_with_chunks = files_that_produce_chunks(all_files)
    vector_store = create_test_vector_store(files_with_chunks[:2], test_data_directory)

    fake_deleted_file = test_data_directory / "deleted_file.md"
    fake_chunks = load_and_chunk_markdown(
        files_with_chunks[0],
        chunk_size=500,
        chunk_overlap=100,
        base_directory=test_data_directory,
    )
    for chunk in fake_chunks:
        chunk.metadata["source"] = "deleted_file.md"
        chunk.metadata["last_modified"] = 0.0

    vector_store.add_documents(fake_chunks)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes.deleted) == 1
    assert changes.deleted[0] == fake_deleted_file


def test_remove_deleted_embeddings(test_data_directory: Path) -> None:
    all_files = collect_markdown_files(test_data_directory)
    files_with_chunks = files_that_produce_chunks(all_files)
    vector_store = create_test_vector_store(files_with_chunks[:2], test_data_directory)

    fake_chunks = load_and_chunk_markdown(
        files_with_chunks[0],
        chunk_size=500,
        chunk_overlap=100,
        base_directory=test_data_directory,
    )
    for chunk in fake_chunks:
        chunk.metadata["source"] = "deleted_file.md"
        chunk.metadata["last_modified"] = 0.0

    vector_store.add_documents(fake_chunks)

    initial_doc_count = len(vector_store.store)

    deleted_files = [test_data_directory / "deleted_file.md"]
    remove_deleted_embeddings(deleted_files, vector_store, test_data_directory)

    final_doc_count = len(vector_store.store)

    assert final_doc_count < initial_doc_count
    assert final_doc_count == initial_doc_count - len(fake_chunks)

    remaining_sources = {
        doc_dict["metadata"]["source"] for doc_dict in vector_store.store.values()
    }
    assert "deleted_file.md" not in remaining_sources


def test_create_embeddings_from_chunks() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    assert len(vector_store.store) == len(chunks)
    for doc_id in vector_store.store.keys():
        doc_dict = vector_store.store[doc_id]
        assert "vector" in doc_dict
        assert isinstance(doc_dict["vector"], list)
        assert len(doc_dict["vector"]) == 384


def test_embeddings_preserve_chunk_metadata() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    for chunk in chunks:
        matching_docs = [
            doc_dict
            for doc_dict in vector_store.store.values()
            if doc_dict["text"] == chunk.page_content
        ]
        assert len(matching_docs) == 1
        assert matching_docs[0]["metadata"] == chunk.metadata


def test_update_existing_embeddings(test_data_directory: Path) -> None:
    test_file = test_data_directory / "sample.md"
    old_chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=test_data_directory
    )
    for chunk in old_chunks:
        chunk.metadata["last_modified"] = 0.0

    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(old_chunks)

    new_chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=test_data_directory
    )

    update_embeddings_for_file(test_file, new_chunks, vector_store, test_data_directory)

    final_count = len(vector_store.store)

    assert final_count == len(new_chunks)

    for doc_dict in vector_store.store.values():
        if doc_dict["metadata"]["source"] == "sample.md":
            assert doc_dict["metadata"]["last_modified"] != 0.0


def test_chroma_persistence(tmp_path: Path) -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    persist_directory = tmp_path / "chroma_db"

    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="chroma",
        vector_store_path=str(persist_directory),
        embedding_provider="fake",
        embedding_model="fake",
        chunk_size=500,
        chunk_overlap=100,
        search_results_limit=5,
    )

    vector_store = create_vector_store(config)
    vector_store.add_documents(chunks)

    assert persist_directory.exists()
    assert len(list(persist_directory.iterdir())) > 0

    results = vector_store.similarity_search("LangChain", k=3)
    assert len(results) > 0


def test_load_existing_chroma_vector_store(tmp_path: Path) -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    persist_directory = tmp_path / "chroma_db"

    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="chroma",
        vector_store_path=str(persist_directory),
        embedding_provider="fake",
        embedding_model="fake",
        chunk_size=500,
        chunk_overlap=100,
        search_results_limit=5,
    )

    vector_store = create_vector_store(config)
    vector_store.add_documents(chunks)

    loaded_store = create_vector_store(config)

    loaded_results = loaded_store.similarity_search("LangChain", k=3)

    assert len(loaded_results) > 0
    assert all(result.page_content for result in loaded_results)


def test_vector_store_uses_fake_embeddings() -> None:
    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="inmemory",
        vector_store_path="./data/chroma_db",
        embedding_provider="fake",
        embedding_model="fake-model",
        chunk_size=500,
        chunk_overlap=100,
        search_results_limit=5,
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store.embeddings, FakeEmbeddings)


def test_vector_store_uses_huggingface_embeddings() -> None:
    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="inmemory",
        vector_store_path="./data/chroma_db",
        embedding_provider="huggingface",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100,
        search_results_limit=5,
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store.embeddings, HuggingFaceEmbeddings)
    assert vector_store.embeddings.model_name == "all-MiniLM-L6-v2"


def test_vector_store_uses_openai_embeddings() -> None:
    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="inmemory",
        vector_store_path="./data/chroma_db",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=100,
        search_results_limit=5,
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store.embeddings, OpenAIEmbeddings)
    assert vector_store.embeddings.model == "text-embedding-3-small"
