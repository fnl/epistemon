"""FastAPI application for semantic search."""

from pathlib import Path
from urllib.parse import quote, unquote

import markdown
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.vectorstores import VectorStoreRetriever

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    retriever: VectorStoreRetriever,
    base_url: str = "",
    score_threshold: float = 0.0,
    files_directory: Path | None = None,
) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if files_directory is not None:

        @app.get("/files/{file_path:path}")
        def serve_markdown_as_html(file_path: str) -> HTMLResponse:
            decoded_path = unquote(file_path)
            full_path = files_directory / decoded_path

            if not full_path.exists():
                raise HTTPException(status_code=404, detail="File not found")

            if not full_path.is_relative_to(files_directory):
                raise HTTPException(status_code=403, detail="Access denied")

            markdown_content = full_path.read_text()
            html_content = markdown.markdown(
                markdown_content,
                extensions=["tables", "fenced_code", "codehilite"],
            )

            html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{decoded_path}</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{ font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
        }}
        a {{
            color: #0969da;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            padding-left: 16px;
            color: #666;
            margin: 16px 0;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
            return HTMLResponse(content=html_template)

    @app.get("/search", response_model=None)
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str | float]] | str] | JSONResponse:
        if not q or not q.strip():
            return {"results": []}

        results_with_scores = retriever.vectorstore.similarity_search_with_score(
            q, k=limit
        )

        metric_type = "similarity"
        if len(results_with_scores) >= 2:
            score1, score2 = results_with_scores[0][1], results_with_scores[1][1]
            if score1 < score2:
                metric_type = "distance"

        results = []
        for doc, score in results_with_scores:
            if score >= score_threshold:
                source = doc.metadata.get("source", "")
                result = {
                    "content": doc.page_content,
                    "source": source,
                    "score": float(score),
                    "metric_type": metric_type,
                }
                if base_url and source:
                    result["link"] = f"{base_url}/{quote(source)}"
                results.append(result)

        response: dict[str, list[dict[str, str | float]] | str] = {"results": results}

        if score_threshold > 0 and len(results) == 0:
            response["alert"] = "No results found matching the score threshold"
            return JSONResponse(content=response, status_code=204)

        return response

    return app
