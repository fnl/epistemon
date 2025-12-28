from pathlib import Path

from fastapi.testclient import TestClient

from epistemon.indexing import embed_and_index, load_and_chunk_markdown
from epistemon.web import create_app


def test_search_endpoint() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = embed_and_index(chunks)

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
