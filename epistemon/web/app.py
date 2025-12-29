"""FastAPI application for semantic search."""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.vectorstores import VectorStoreRetriever

STATIC_DIR = Path(__file__).parent / "static"


def create_app(retriever: VectorStoreRetriever) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str | float]]]:
        if not q or not q.strip():
            return {"results": []}

        results_with_scores = retriever.vectorstore.similarity_search_with_score(
            q, k=limit
        )
        return {
            "results": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "score": float(score),
                }
                for doc, score in results_with_scores
            ]
        }

    return app
