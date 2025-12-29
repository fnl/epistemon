"""FastAPI application for semantic search."""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.retrievers import BaseRetriever

STATIC_DIR = Path(__file__).parent / "static"


def create_app(retriever: BaseRetriever) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str]]]:
        results = retriever.invoke(q, k=limit)

        return {
            "results": [
                {"content": doc.page_content, "source": doc.metadata.get("source", "")}
                for doc in results
            ]
        }

    return app
