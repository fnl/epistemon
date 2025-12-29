from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from epistemon.indexing import load_and_chunk_markdown


def test_search_indexed_content() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("LangChain framework")

    assert len(results) > 0
    assert len(results) <= 3
    assert all(result.page_content for result in results)
    assert all("source" in result.metadata for result in results)
