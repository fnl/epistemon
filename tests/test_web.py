from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from epistemon.indexing import load_and_chunk_markdown
from epistemon.web import create_app


@pytest.fixture
def retriever() -> VectorStoreRetriever:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)
    return vector_store.as_retriever()


def test_root_serves_html(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever)
    client = TestClient(app)

    response = client.get("/")

    assert "text/html" in response.headers["content-type"]
    assert b"Epistemon" in response.content


def test_search_returns_results(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever, score_threshold=-1000.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 3})

    results = response.json()["results"]
    assert len(results) > 0
    assert len(results) <= 3


def test_search_respects_limit(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever)
    client = TestClient(app)

    results_1 = client.get("/search", params={"q": "LangChain", "limit": 1}).json()[
        "results"
    ]
    results_3 = client.get("/search", params={"q": "LangChain", "limit": 3}).json()[
        "results"
    ]

    assert len(results_1) == 1
    assert len(results_3) <= 3


def test_search_results_ranked_by_score(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever, score_threshold=-1000.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 5})

    results = response.json()["results"]
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_handles_empty_query(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever)
    client = TestClient(app)

    response = client.get("/search", params={"q": "", "limit": 5})

    assert response.json()["results"] == []


def test_search_filters_by_score_threshold(retriever: VectorStoreRetriever) -> None:
    client_low = TestClient(create_app(retriever, score_threshold=0.0))
    client_high = TestClient(create_app(retriever, score_threshold=1000000.0))

    results_low = client_low.get(
        "/search", params={"q": "LangChain", "limit": 5}
    ).json()["results"]
    results_high = client_high.get(
        "/search", params={"q": "LangChain", "limit": 5}
    ).json()["results"]

    assert len(results_low) > 0
    assert len(results_high) == 0


def test_search_returns_alert_when_no_matches(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever, score_threshold=1000000.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 5})

    assert response.status_code == 204
    data = response.json()
    assert data["results"] == []
    assert "alert" in data
