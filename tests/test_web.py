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
    app = create_app(retriever, score_threshold=-1000.0)
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
    app = create_app(retriever, score_threshold=-1000.0)
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


def test_search_handles_empty_query() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever()
    app = create_app(retriever)
    client = TestClient(app)

    response = client.get("/search", params={"q": "", "limit": 5})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["results"] == []


def test_search_filters_results_below_score_threshold() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever()
    app_without_threshold = create_app(retriever, score_threshold=0.0)
    client_without = TestClient(app_without_threshold)

    response_without_threshold = client_without.get(
        "/search", params={"q": "LangChain", "limit": 5}
    )
    data_without_threshold = response_without_threshold.json()
    all_results = data_without_threshold["results"]
    assert len(all_results) > 0

    if len(all_results) > 1:
        middle_score = all_results[len(all_results) // 2]["score"]

        app_with_threshold = create_app(retriever, score_threshold=middle_score)
        client_with = TestClient(app_with_threshold)

        response_with_threshold = client_with.get(
            "/search", params={"q": "LangChain", "limit": 5}
        )
        data_with_threshold = response_with_threshold.json()
        filtered_results = data_with_threshold["results"]

        assert len(filtered_results) < len(all_results)
        assert all(result["score"] >= middle_score for result in filtered_results)


def test_search_returns_alert_when_all_results_below_threshold() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever()
    very_high_threshold = 1000000.0
    app = create_app(retriever, score_threshold=very_high_threshold)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 5})

    assert response.status_code == 204
    data = response.json()
    assert "results" in data
    assert data["results"] == []
    assert "alert" in data
    assert "no match" in data["alert"].lower() or "no results" in data["alert"].lower()
