from pathlib import Path

from epistemon.indexing import (
    embed_and_index,
    load_and_chunk_markdown,
    scan_markdown_files,
)


def test_scan_markdown_files_non_recursively() -> None:
    directory = Path("tests/data")

    markdown_files = scan_markdown_files(directory, recursive=False)

    assert len(markdown_files) == 3
    assert all(f.suffix == ".md" for f in markdown_files)
    assert all(f.exists() for f in markdown_files)

    file_names = {f.name for f in markdown_files}
    assert "sample.md" in file_names
    assert "doc1.md" in file_names
    assert "doc2.md" in file_names
    assert "not_markdown.txt" not in file_names


def test_scan_markdown_files_recursively() -> None:
    directory = Path("tests/data")

    markdown_files = scan_markdown_files(directory)

    assert len(markdown_files) == 5
    assert all(f.suffix == ".md" for f in markdown_files)
    assert all(f.exists() for f in markdown_files)

    file_names = {f.name for f in markdown_files}
    assert "sample.md" in file_names
    assert "doc1.md" in file_names
    assert "doc2.md" in file_names
    assert "nested_doc.md" in file_names
    assert "deep_doc.md" in file_names


def test_load_and_chunk_markdown() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == str(test_file) for chunk in chunks)


def test_load_and_chunk_markdown_with_relative_source() -> None:
    base_dir = Path("tests/data")
    test_file = base_dir / "subdir" / "nested_doc.md"
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_dir
    )

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "subdir/nested_doc.md" for chunk in chunks)


def test_embed_and_index() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = embed_and_index(chunks)

    stored_docs = vector_store.similarity_search("LangChain", k=3)

    assert len(stored_docs) > 0
    assert all(doc.page_content for doc in stored_docs)
    assert all("source" in doc.metadata for doc in stored_docs)
