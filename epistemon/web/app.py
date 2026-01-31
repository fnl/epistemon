"""FastAPI application for semantic search."""

import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import markdown
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from epistemon import __version__
from epistemon.config import Configuration
from epistemon.indexing.bm25_indexer import BM25Indexer
from epistemon.indexing.vector_store_manager import VectorStoreManager
from epistemon.retrieval.rag_chain import RAGChainProtocol

logger = logging.getLogger(__name__)

HTML_404_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - File Not Found</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 600px;
            margin: 100px auto;
            padding: 0 20px;
            text-align: center;
            color: #333;
        }}
        h1 {{
            font-size: 3em;
            margin-bottom: 0.5em;
            color: #d32f2f;
        }}
        p {{
            font-size: 1.2em;
            margin-bottom: 1em;
            color: #666;
        }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        a {{
            color: #0969da;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>404</h1>
    <p>File not found</p>
    <p>The requested file <code>{decoded_path}</code> does not exist.</p>
    <p><a href="/app">Return to search</a></p>
</body>
</html>
"""

HTML_MARKDOWN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{{{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #333;
        }}}}
        h1, h2, h3, h4, h5, h6 {{{{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}}}
        h1 {{{{ font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}}}
        h2 {{{{ font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}}}
        code {{{{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }}}}
        pre {{{{
            background: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
        }}}}
        pre code {{{{
            background: none;
            padding: 0;
        }}}}
        table {{{{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}}}
        th, td {{{{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}}}
        th {{{{
            background: #f6f8fa;
            font-weight: 600;
        }}}}
        a {{{{
            color: #0969da;
            text-decoration: none;
        }}}}
        a:hover {{{{
            text-decoration: underline;
        }}}}
        blockquote {{{{
            border-left: 4px solid #ddd;
            padding-left: 16px;
            color: #666;
            margin: 16px 0;
        }}}}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""


def _serve_markdown_file(
    file_path: str, files_directory: Path
) -> HTMLResponse | JSONResponse | FileResponse:
    decoded_path = unquote(file_path)
    full_path = files_directory / decoded_path

    if not full_path.exists():
        html_404 = HTML_404_TEMPLATE.format(decoded_path=decoded_path)
        return HTMLResponse(content=html_404, status_code=404)

    if not full_path.is_relative_to(files_directory):
        raise HTTPException(status_code=403, detail="Access denied")

    file_suffix = full_path.suffix.lower()
    if file_suffix not in [".md", ".markdown"]:
        mime_type, _ = mimetypes.guess_type(str(full_path))
        return FileResponse(
            path=full_path, media_type=mime_type or "application/octet-stream"
        )

    try:
        markdown_content = full_path.read_text()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to read file", "detail": str(e)},
        )

    html_content = markdown.markdown(
        markdown_content,
        extensions=["tables", "fenced_code", "codehilite"],
    )

    html_template = HTML_MARKDOWN_TEMPLATE.format(
        title=decoded_path, content=html_content
    )
    return HTMLResponse(content=html_template)


def _collect_files_from_manager(
    vector_store_manager: VectorStoreManager,
) -> dict[str, dict[str, str | float | int]]:
    files_dict: dict[str, dict[str, str | float | int]] = {}
    indexed_files = vector_store_manager.get_indexed_files()
    for abs_path, mtime in indexed_files.items():
        abs_path_obj = Path(abs_path)
        try:
            relative_path = abs_path_obj.relative_to(
                vector_store_manager.base_directory
            )
            source = str(relative_path)
            if source not in files_dict:
                iso_date = datetime.fromtimestamp(mtime).isoformat()
                files_dict[source] = {
                    "source": source,
                    "last_modified": iso_date,
                }
        except ValueError:
            continue
    return files_dict


def _collect_files_from_vector_store(
    vector_store: InMemoryVectorStore,
) -> dict[str, dict[str, str | float | int]]:
    files_dict: dict[str, dict[str, str | float | int]] = {}
    for _doc_id, doc in vector_store.store.items():
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "")
        if source and source not in files_dict:
            mtime = metadata.get("last_modified", 0)
            iso_date = datetime.fromtimestamp(mtime).isoformat()
            files_dict[source] = {
                "source": source,
                "last_modified": iso_date,
            }
    return files_dict


def _sort_files_list(
    files_list: list[dict[str, str | float | int]], sort_by: str
) -> list[dict[str, str | float | int]]:
    if sort_by == "name":
        files_list.sort(key=lambda f: f["source"])
    elif sort_by == "date":
        files_list.sort(key=lambda f: f["last_modified"], reverse=True)
    return files_list


def _list_indexed_files(
    sort_by: str,
    vector_store: VectorStore,
    vector_store_manager: Optional[VectorStoreManager],
) -> dict[str, list[dict[str, str | float | int]]] | JSONResponse:
    if sort_by not in ["name", "date"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid sort_by parameter",
                "detail": "sort_by must be 'name' or 'date'",
            },
        )

    try:
        if vector_store_manager:
            files_dict = _collect_files_from_manager(vector_store_manager)
        elif isinstance(vector_store, InMemoryVectorStore):
            files_dict = _collect_files_from_vector_store(vector_store)
        else:
            files_dict = {}

        files_list = list(files_dict.values())
        files_list = _sort_files_list(files_list, sort_by)

        return {"files": files_list}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve file list", "detail": str(e)},
        )


def _execute_semantic_search(
    q: str,
    limit: int,
    vector_store: VectorStore,
    score_threshold: float,
    base_url: str,
) -> dict[str, list[dict[str, str | float | int]] | str] | JSONResponse:
    if not q or not q.strip():
        return {"results": []}

    try:
        results_with_scores = vector_store.similarity_search_with_score(q, k=limit)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Search failed", "detail": str(e)},
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
            last_modified = doc.metadata.get("last_modified", 0)
            result = {
                "content": doc.page_content,
                "source": source,
                "last_modified": last_modified,
                "score": float(score),
                "metric_type": metric_type,
            }
            if base_url and source:
                result["link"] = f"{base_url}/{quote(source)}"
            results.append(result)

    response: dict[str, list[dict[str, str | float | int]] | str] = {"results": results}

    if score_threshold > 0 and len(results) == 0:
        response["alert"] = "No results found matching the score threshold"
        return JSONResponse(content=response, status_code=204)

    return response


def create_app(
    vector_store: VectorStore,
    base_url: str = "",
    score_threshold: float = 0.0,
    files_directory: Path | None = None,
    vector_store_manager: Optional[VectorStoreManager] = None,
    bm25_retriever: Optional[BM25Indexer] = None,
    rag_chain: Optional[RAGChainProtocol] = None,
    config: Optional[Configuration] = None,
) -> FastAPI:
    if config is not None:
        score_threshold = config.score_threshold

    app = FastAPI(
        title="Epistemon API",
        description="Semantic Markdown Search API using LangChain and vector embeddings",
        version=__version__,
    )

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/", status_code=307)

    if files_directory is not None:
        logger.info("Serving files from directory: %s", files_directory)

        @app.get("/files/{file_path:path}", response_model=None)
        def serve_markdown_as_html(
            file_path: str,
        ) -> HTMLResponse | JSONResponse | FileResponse:
            return _serve_markdown_file(file_path, files_directory)

    @app.get("/files", response_model=None)
    def list_files_endpoint(
        sort_by: str = Query("name", description="Sort by 'name' or 'date'"),
    ) -> dict[str, list[dict[str, str | float | int]]] | JSONResponse:
        return _list_indexed_files(sort_by, vector_store, vector_store_manager)

    @app.get("/search", response_model=None)
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str | float | int]] | str] | JSONResponse:
        return _execute_semantic_search(
            q, limit, vector_store, score_threshold, base_url
        )

    from epistemon.web.shiny_ui import create_shiny_app

    shiny_app = create_shiny_app(
        vector_store,
        base_url=base_url,
        score_threshold=score_threshold,
        bm25_retriever=bm25_retriever,
        rag_chain=rag_chain,
    )
    app.mount("/app", shiny_app)

    return app
