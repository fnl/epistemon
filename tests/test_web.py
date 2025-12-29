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

    retriever = vector_store.as_retriever()
    app = create_app(retriever)
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

    retriever = vector_store.as_retriever()
    app = create_app(retriever)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 3})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert len(data["results"]) <= 3
    assert all("content" in result for result in data["results"])
    assert all("source" in result for result in data["results"])


def test_search_respects_configurable_limit() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever()
    app = create_app(retriever)
    client = TestClient(app)

    response_limit_1 = client.get("/search", params={"q": "LangChain", "limit": 1})
    assert response_limit_1.status_code == 200
    data_limit_1 = response_limit_1.json()
    assert len(data_limit_1["results"]) == 1

    response_limit_3 = client.get("/search", params={"q": "LangChain", "limit": 3})
    assert response_limit_3.status_code == 200
    data_limit_3 = response_limit_3.json()
    assert len(data_limit_3["results"]) <= 3

    response_limit_10 = client.get("/search", params={"q": "LangChain", "limit": 10})
    assert response_limit_10.status_code == 200
    data_limit_10 = response_limit_10.json()
    assert len(data_limit_10["results"]) <= 10


def test_search_results_ranked_by_score() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever()
    app = create_app(retriever)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 5})

    assert response.status_code == 200
    data = response.json()
    results = data["results"]

    assert len(results) > 0
    assert all("score" in result for result in results)
    assert all("content" in result for result in results)
    assert all("source" in result for result in results)

    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
