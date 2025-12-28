from pathlib import Path

from epistemon.indexing import embed_and_index, load_and_chunk_markdown
from epistemon.search import search


def test_search_indexed_content() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = embed_and_index(chunks)

    results = search(vector_store, "LangChain framework", limit=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(result.page_content for result in results)
    assert all("source" in result.metadata for result in results)
