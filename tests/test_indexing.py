from pathlib import Path

from epistemon.indexing import embed_and_index, load_and_chunk_markdown


def test_load_and_chunk_markdown() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == str(test_file) for chunk in chunks)


def test_embed_and_index() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = embed_and_index(chunks)

    stored_docs = vector_store.similarity_search("LangChain", k=3)

    assert len(stored_docs) > 0
    assert all(doc.page_content for doc in stored_docs)
    assert all("source" in doc.metadata for doc in stored_docs)
