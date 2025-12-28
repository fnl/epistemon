"""FastAPI application for semantic search."""

from fastapi import FastAPI, Query
from langchain_core.vectorstores import VectorStore

from epistemon.search import search


def create_app(vector_store: VectorStore) -> FastAPI:
    app = FastAPI()

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str]]]:
        results = search(vector_store, q, limit)

        return {
            "results": [
                {"content": doc.page_content, "source": doc.metadata.get("source", "")}
                for doc in results
            ]
        }

    return app
