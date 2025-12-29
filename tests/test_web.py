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
    base_directory = Path("tests/data")
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_directory
    )
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
    app = create_app(retriever, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 3})

    results = response.json()["results"]
    assert len(results) > 0
    assert len(results) <= 3


def test_search_respects_limit(retriever: VectorStoreRetriever) -> None:
    app = create_app(retriever, score_threshold=0.0)
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
    app = create_app(retriever, score_threshold=-1.0)
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
    client_low = TestClient(create_app(retriever, score_threshold=-1.0))
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


def test_search_results_include_source_link(retriever: VectorStoreRetriever) -> None:
    base_url = "http://localhost:8000/files"
    app = create_app(retriever, base_url=base_url, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 1})

    results = response.json()["results"]
    assert len(results) > 0
    result = results[0]
    assert "link" in result
    assert result["link"].startswith(base_url)
    assert "sample.md" in result["link"]


def test_files_endpoint_serves_markdown() -> None:
    from pathlib import Path

    test_data_dir = Path("tests/data")
    retriever_fixture = InMemoryVectorStore(FakeEmbeddings(size=384)).as_retriever()
    app = create_app(retriever_fixture, files_directory=test_data_dir)
    client = TestClient(app)

    response = client.get("/files/sample.md")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "").lower()
    assert b"<html" in response.content.lower()
    assert b"<h1>" in response.content.lower() or b"<h2>" in response.content.lower()


def test_end_to_end_search_and_file_retrieval(retriever: VectorStoreRetriever) -> None:
    test_data_dir = Path("tests/data")
    app = create_app(
        retriever,
        base_url="http://testserver/files",
        files_directory=test_data_dir,
        score_threshold=-1.0,
    )
    client = TestClient(app)

    search_response = client.get("/search", params={"q": "LangChain", "limit": 1})
    search_results = search_response.json()["results"]
    assert len(search_results) > 0

    result = search_results[0]
    assert "link" in result
    link = result["link"]

    file_path = link.replace("http://testserver/files/", "")
    file_response = client.get(f"/files/{file_path}")

    assert file_response.status_code == 200
    content = file_response.text
    assert "LangChain" in content or len(content) > 0
