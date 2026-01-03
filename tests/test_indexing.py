import time
from pathlib import Path

import pytest
from langchain_community.vectorstores import DuckDB
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_weaviate import WeaviateVectorStore

from epistemon.config import Configuration
from epistemon.indexing import (
    collect_markdown_files,
    detect_file_changes,
    load_and_chunk_markdown,
    remove_deleted_embeddings,
)
from epistemon.indexing.vector_store_manager import create_vector_store_manager
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


def create_test_config(
    input_directory: Path | str = "./tests/data",
    vector_store_type: str = "inmemory",
    vector_store_path: str = "./data/chroma_db",
    embedding_provider: str = "fake",
    embedding_model: str = "fake",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    score_threshold: float = 0.0,
) -> Configuration:
    return Configuration(
        input_directory=str(input_directory),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_store_type=vector_store_type,
        vector_store_path=vector_store_path,
        search_results_limit=5,
        score_threshold=score_threshold,
        bm25_k1=1.5,
        bm25_b=0.75,
        bm25_top_k=5,
    )


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
    manager = create_vector_store_manager(vector_store, test_data_directory)

    changes = detect_file_changes(test_data_directory, manager)

    assert len(changes.new) == len(files_with_chunks) - 2
    assert len(changes.modified) == 0
    assert len(changes.deleted) == 0


def test_detect_modified_files(test_data_directory: Path) -> None:
    all_files = collect_markdown_files(test_data_directory, recursive=False)
    files_with_chunks = files_that_produce_chunks(all_files)
    files = files_with_chunks[:2]
    vector_store = create_test_vector_store(files, test_data_directory, old_mtime=True)
    manager = create_vector_store_manager(vector_store, test_data_directory)

    changes = detect_file_changes(test_data_directory, manager)

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
    manager = create_vector_store_manager(vector_store, test_data_directory)

    changes = detect_file_changes(test_data_directory, manager)

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
    manager = create_vector_store_manager(vector_store, test_data_directory)

    changes = detect_file_changes(test_data_directory, manager)

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
    manager = create_vector_store_manager(vector_store, test_data_directory)

    changes = detect_file_changes(test_data_directory, manager)

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
    manager = create_vector_store_manager(vector_store, test_data_directory)

    initial_doc_count = len(vector_store.store)

    deleted_files = [test_data_directory / "deleted_file.md"]
    remove_deleted_embeddings(deleted_files, manager)

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
    manager = create_vector_store_manager(vector_store, test_data_directory)

    new_chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=test_data_directory
    )

    manager.update_documents_for_file(test_file, new_chunks)

    final_count = len(vector_store.store)

    assert final_count == len(new_chunks)

    for doc_dict in vector_store.store.values():
        if doc_dict["metadata"]["source"] == "sample.md":
            assert doc_dict["metadata"]["last_modified"] != 0.0


def test_chroma_persistence(tmp_path: Path) -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    persist_directory = tmp_path / "chroma_db"

    config = create_test_config(
        vector_store_type="chroma", vector_store_path=str(persist_directory)
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

    config = create_test_config(
        vector_store_type="chroma", vector_store_path=str(persist_directory)
    )

    vector_store = create_vector_store(config)
    vector_store.add_documents(chunks)

    loaded_store = create_vector_store(config)

    loaded_results = loaded_store.similarity_search("LangChain", k=3)

    assert len(loaded_results) > 0
    assert all(result.page_content for result in loaded_results)


def test_vector_store_uses_fake_embeddings() -> None:
    config = create_test_config(embedding_model="fake-model")

    vector_store = create_vector_store(config)

    assert isinstance(vector_store.embeddings, FakeEmbeddings)


@pytest.mark.slow
def test_vector_store_uses_huggingface_embeddings() -> None:
    config = create_test_config(
        embedding_provider="huggingface", embedding_model="all-MiniLM-L6-v2"
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store.embeddings, HuggingFaceEmbeddings)
    assert vector_store.embeddings.model_name == "all-MiniLM-L6-v2"


def test_vector_store_uses_openai_embeddings() -> None:
    config = create_test_config(
        embedding_provider="openai", embedding_model="text-embedding-3-small"
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store.embeddings, OpenAIEmbeddings)
    assert vector_store.embeddings.model == "text-embedding-3-small"


@pytest.mark.slow
def test_vector_store_uses_weaviate(tmp_path: Path) -> None:
    config = create_test_config(
        vector_store_type="weaviate", vector_store_path=str(tmp_path / "weaviate_db")
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store, WeaviateVectorStore)
    assert vector_store._client is not None


def test_vector_store_uses_qdrant(tmp_path: Path) -> None:
    config = create_test_config(
        vector_store_type="qdrant", vector_store_path=str(tmp_path / "qdrant_db")
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store, QdrantVectorStore)


def test_vector_store_uses_duckdb(tmp_path: Path) -> None:
    config = create_test_config(
        vector_store_type="duckdb", vector_store_path=str(tmp_path / "duckdb_data")
    )

    vector_store = create_vector_store(config)

    assert isinstance(vector_store, DuckDB)


def test_markdown_structure_based_chunking() -> None:
    test_file = Path("tests/data/markdown_with_headers.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=150, chunk_overlap=0)

    assert len(chunks) > 0

    chunk_contents = [chunk.page_content for chunk in chunks]

    headers_found = [
        "# Document Title",
        "## Section One Header",
        "## Section Two Header",
        "## Section Three Header",
    ]

    for header in headers_found:
        chunks_with_header = [c for c in chunk_contents if header in c]
        assert len(chunks_with_header) == 1

        chunk_with_header = chunks_with_header[0]
        header_line_index = chunk_with_header.find(header)
        text_after_header = chunk_with_header[header_line_index + len(header) :].strip()

        if "Section" in header:
            assert len(text_after_header) > 0


def test_oversized_markdown_chunks_are_split() -> None:
    test_file = Path("tests/data/large_section.md")
    chunk_size = 200
    chunks = load_and_chunk_markdown(test_file, chunk_size=chunk_size, chunk_overlap=0)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size * 1.5


def test_index_function_indexes_all_files(test_data_directory: Path) -> None:
    from epistemon.indexing import index

    config = create_test_config(test_data_directory)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    index(config, vector_store)

    assert len(vector_store.store) > 0

    sources = {doc["metadata"]["source"] for doc in vector_store.store.values()}
    assert len(sources) > 0


def test_index_function_skips_unchanged_files(test_data_directory: Path) -> None:
    from epistemon.indexing import index

    config = create_test_config(test_data_directory)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    index(config, vector_store)
    initial_count = len(vector_store.store)

    index(config, vector_store)
    final_count = len(vector_store.store)

    assert initial_count == final_count


def test_index_function_updates_modified_files(tmp_path: Path) -> None:
    from epistemon.indexing import index

    temp_file = tmp_path / "test_file.md"
    temp_file.write_text("# Original Content\n\nThis is the original content.")

    config = create_test_config(tmp_path)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    index(config, vector_store)
    initial_chunks = [
        doc["text"]
        for doc in vector_store.store.values()
        if doc["metadata"]["source"] == "test_file.md"
    ]

    assert len(initial_chunks) > 0
    assert any("Original Content" in chunk for chunk in initial_chunks)

    time.sleep(0.01)
    temp_file.write_text(
        "# Updated Content\n\nThis is the updated content with different text."
    )

    index(config, vector_store)
    updated_chunks = [
        doc["text"]
        for doc in vector_store.store.values()
        if doc["metadata"]["source"] == "test_file.md"
    ]

    assert len(updated_chunks) > 0
    assert any("Updated Content" in chunk for chunk in updated_chunks)
    assert not any("Original Content" in chunk for chunk in updated_chunks)


def test_index_function_removes_deleted_files(tmp_path: Path) -> None:
    from epistemon.indexing import index

    temp_file = tmp_path / "to_be_deleted.md"
    temp_file.write_text("# File to Delete\n\nThis file will be deleted.")

    config = create_test_config(tmp_path)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    index(config, vector_store)

    sources_before = {doc["metadata"]["source"] for doc in vector_store.store.values()}
    assert "to_be_deleted.md" in sources_before

    temp_file.unlink()

    index(config, vector_store)

    sources_after = {doc["metadata"]["source"] for doc in vector_store.store.values()}
    assert "to_be_deleted.md" not in sources_after


@pytest.mark.slow
def test_indexing_performance_per_file(tmp_path: Path) -> None:
    from epistemon.indexing import index

    num_files = 10
    for i in range(num_files):
        file = tmp_path / f"file_{i}.md"
        file.write_text(f"# File {i}\n\nThis is file number {i} with some content.")

    config = create_test_config(tmp_path)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    start_time = time.perf_counter()
    index(config, vector_store)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_file = total_time / num_files

    print("\nIndexing performance (new files):")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Files indexed: {num_files}")
    print(f"  Time per file: {time_per_file * 1000:.2f}ms")
    print("  Target: 10.00ms per file")

    assert len(vector_store.store) > 0
    assert time_per_file < 0.1


@pytest.mark.slow
def test_reindexing_performance_for_unchanged_files(tmp_path: Path) -> None:
    from epistemon.indexing import index

    num_files = 100
    for i in range(num_files):
        file = tmp_path / f"file_{i}.md"
        content = f"# File {i}\n\n" + "\n\n".join(
            f"This is paragraph {j} of file {i}." for j in range(10)
        )
        file.write_text(content)

    config = create_test_config(tmp_path)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    index(config, vector_store)
    initial_count = len(vector_store.store)

    start_time = time.perf_counter()
    index(config, vector_store)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_file = total_time / num_files

    print("\nRe-indexing performance (unchanged files):")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Files checked: {num_files}")
    print(f"  Time per file: {time_per_file * 1000:.2f}ms")
    print("  Target: 10.00ms per file")

    assert len(vector_store.store) == initial_count
    assert time_per_file < 0.01


@pytest.mark.slow
def test_indexing_performance_with_instrumentation(tmp_path: Path) -> None:
    from epistemon.indexing import index
    from epistemon.instrumentation import (
        disable_instrumentation,
        enable_instrumentation,
        get_metrics,
    )

    num_files = 50
    for i in range(num_files):
        file = tmp_path / f"file_{i}.md"
        content = f"# File {i}\n\n" + "\n\n".join(
            f"This is paragraph {j} of file {i}." for j in range(10)
        )
        file.write_text(content)

    config = create_test_config(tmp_path)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))

    enable_instrumentation()

    index(config, vector_store)

    metrics = get_metrics()
    assert metrics is not None

    summary = metrics.summary()

    print(f"\nPerformance breakdown for {num_files} files:")
    for operation, stats in sorted(
        summary.items(), key=lambda x: x[1]["total"], reverse=True
    ):
        print(f"\n  {operation}:")
        print(f"    Total time: {stats['total'] * 1000:.2f}ms")
        print(f"    Count: {stats['count']}")
        print(f"    Avg: {stats['avg'] * 1000:.4f}ms")
        print(f"    Min: {stats['min'] * 1000:.4f}ms")
        print(f"    Max: {stats['max'] * 1000:.4f}ms")

    disable_instrumentation()

    assert len(vector_store.store) > 0
    assert "index_total" in summary
    assert "detect_file_changes" in summary
    assert "load_and_chunk_markdown" in summary


def test_index_function_works_with_chroma(tmp_path: Path) -> None:
    from epistemon.indexing import index
    from epistemon.vector_store_factory import create_vector_store

    file1 = tmp_path / "file1.md"
    file1.write_text("# Test\n\nThis is a test file.")

    config = create_test_config(
        input_directory=tmp_path,
        vector_store_type="chroma",
        vector_store_path=str(tmp_path / "chroma_db"),
    )

    vector_store = create_vector_store(config)

    index(config, vector_store)

    retriever = vector_store.as_retriever()
    results = retriever.invoke("test")

    assert len(results) > 0
    assert any("test" in doc.page_content.lower() for doc in results)


def test_chunks_do_not_end_with_dangling_headlines() -> None:
    test_file = Path("tests/data/dangling_headline_test.md")
    chunk_size = 300
    chunk_overlap = 0
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    assert len(chunks) > 0

    for chunk in chunks:
        lines = chunk.page_content.strip().split("\n")
        last_line = lines[-1].strip()

        is_headline = last_line.startswith("#")
        is_only_content = len(lines) == 1

        if is_headline and not is_only_content:
            raise AssertionError(
                f"Chunk ends with dangling headline: '{last_line}'\n"
                f"Full chunk:\n{chunk.page_content}"
            )
