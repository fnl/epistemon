from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from epistemon.indexing import load_and_chunk_markdown
from epistemon.web import create_app


def test_root_serves_html() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Epistemon" in response.content
    assert b"search" in response.content.lower()


def test_search_endpoint() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 3})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert len(data["results"]) <= 3
    assert all("content" in result for result in data["results"])
    assert all("source" in result for result in data["results"])
