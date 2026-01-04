"""Shiny UI for semantic search."""

from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from epistemon.indexing.bm25_indexer import BM25Indexer, highlight_keywords
from epistemon.retrieval.rag_chain import RAGChain


def _validate_search_inputs(query: str, limit: Optional[int]) -> Optional[ui.TagList]:
    """Validate search query and limit inputs.

    Args:
        query: Search query string
        limit: Result limit number

    Returns:
        TagList with error message if validation fails, None if valid
    """
    if limit is None or limit < 1:
        return ui.TagList(
            ui.div(
                ui.p("Result limit must be at least 1", class_="text-dark"),
                class_="alert alert-warning",
            )
        )

    if not query or not query.strip():
        return ui.TagList(
            ui.div(
                ui.p("Please enter a search query", class_="text-dark"),
                class_="alert alert-warning",
            )
        )

    return None


def _create_result_card(
    doc: Document,
    score: float,
    idx: int,
    base_url: str,
    score_class: str,
    score_label: str,
    content: str,
) -> Any:
    """Create a result card for a single document.

    Args:
        doc: Document to display
        score: Relevance score
        idx: Result index number
        base_url: Base URL for source links
        score_class: CSS class for score badge
        score_label: Label to display with score
        content: Document content (possibly highlighted)

    Returns:
        Shiny Card component with result
    """
    source = doc.metadata.get("source", "Unknown")
    last_modified = doc.metadata.get("last_modified", 0)

    if base_url and source:
        source_link = ui.a(
            source,
            href=f"{base_url}/{quote(source)}",
            target="_blank",
            class_="text-primary",
        )
    else:
        source_link = ui.span(source, class_="text-muted")

    modified_display = ""
    if last_modified:
        dt = datetime.fromtimestamp(last_modified)
        modified_display = dt.strftime("%Y-%m-%d %H:%M:%S")

    metadata_items = [
        ui.span(ui.strong("Source: "), source_link),
    ]
    if modified_display:
        metadata_items.append(
            ui.span(
                ui.strong("Modified: "),
                ui.span(modified_display, class_="text-muted"),
            )
        )

    return ui.card(
        ui.card_header(
            ui.div(
                ui.strong(f"Result {idx}", class_="me-2"),
                ui.span(
                    f"{score_label}: {score:.4f}",
                    class_=f"badge {score_class} rounded-pill",
                ),
            ),
            class_="d-flex align-items-center",
        ),
        ui.div(
            ui.HTML(
                f'<pre class="bg-light p-2 rounded mb-2" style="white-space: pre-wrap;">{content}</pre>'
            ),
            ui.tags.ul(
                *[ui.tags.li(item, class_="small") for item in metadata_items],
                class_="list-unstyled mb-0",
            ),
        ),
        class_="mb-3",
    )


def _create_results_list(
    result_cards: list[Any],
    filtered_count: int,
    score_threshold: float,
) -> ui.TagList:
    """Assemble final results list with header and alerts.

    Args:
        result_cards: List of result cards
        filtered_count: Number of results filtered by threshold
        score_threshold: Score threshold used for filtering

    Returns:
        TagList with complete results UI
    """
    cards_with_metadata = list(result_cards)

    if filtered_count > 0:
        alert = ui.div(
            ui.p(
                f"{filtered_count} result(s) filtered by score threshold "
                f"({score_threshold})",
                class_="mb-0",
            ),
            class_="alert alert-warning mb-3",
        )
        cards_with_metadata.insert(0, alert)

    header = ui.h3(
        f"Found {len(result_cards)} results",
        class_="mb-3",
    )
    cards_with_metadata.insert(0, header)

    return ui.TagList(*cards_with_metadata)


def _create_search_ui() -> Any:
    """Create the search UI layout.

    Returns:
        Shiny UI page structure
    """
    return ui.page_fluid(
        ui.panel_title("Epistemon Semantic Search"),
        ui.div(
            ui.row(
                ui.column(
                    10,
                    ui.input_text(
                        "query",
                        "Search Query",
                        placeholder="Enter your search query...",
                    ),
                ),
                ui.column(
                    1,
                    ui.input_numeric(
                        "limit",
                        "Result Limit",
                        value=5,
                        min=1,
                    ),
                ),
                ui.column(
                    1,
                    ui.div(
                        ui.input_action_button(
                            "search",
                            "Search",
                            class_="btn-primary w-100",
                        ),
                        style="padding-top: 25px;",
                    ),
                ),
            ),
            class_="mb-3",
        ),
        ui.tags.script(
            """
            $(document).ready(function() {
                $('#query').on('keypress', function(e) {
                    if (e.which === 13) {
                        $('#search').click();
                    }
                });
            });
            """
        ),
        ui.layout_columns(
            ui.div(
                ui.h4("BM25 (Keyword Search)"),
                ui.output_ui("bm25_results"),
            ),
            ui.div(
                ui.h4("Semantic (Embedding Search)"),
                ui.output_ui("results"),
            ),
            ui.div(
                ui.h4("RAG Answer"),
                ui.output_ui("rag_answer"),
            ),
        ),
    )


def _execute_bm25_search(
    bm25_retriever: Optional[BM25Indexer],
    base_url: str,
    score_threshold: float,
    query: str,
    limit: Optional[int],
) -> ui.TagList:
    """Execute BM25 search and return results UI.

    Args:
        bm25_retriever: Optional BM25 indexer
        base_url: Base URL for source links
        score_threshold: Minimum score threshold
        query: Search query string
        limit: Result limit

    Returns:
        TagList with BM25 search results
    """
    if bm25_retriever is None:
        return ui.TagList(
            ui.div(
                ui.p("BM25 search not available", class_="text-dark"),
                class_="alert alert-info",
            )
        )

    validation_error = _validate_search_inputs(query, limit)
    if validation_error:
        return validation_error

    if limit is None:
        raise ValueError("Limit cannot be None after validation")

    try:
        results_with_scores = bm25_retriever.retrieve(query, top_k=limit)
    except Exception as e:
        return ui.TagList(
            ui.div(
                ui.p(f"BM25 search error: {str(e)}", class_="text-dark"),
                class_="alert alert-danger",
            )
        )

    if not results_with_scores:
        return ui.TagList(
            ui.div(
                ui.p("No results found", class_="text-dark"),
                class_="alert alert-info",
            )
        )

    result_cards = []
    filtered_count = 0

    for idx, (doc, score) in enumerate(results_with_scores, 1):
        if score < score_threshold:
            filtered_count += 1
            continue

        content = highlight_keywords(doc.page_content, query)
        card = _create_result_card(
            doc=doc,
            score=score,
            idx=idx,
            base_url=base_url,
            score_class="bg-info",
            score_label="Score",
            content=content,
        )
        result_cards.append(card)

    return _create_results_list(result_cards, filtered_count, score_threshold)


def _execute_semantic_search(
    vector_store: VectorStore,
    base_url: str,
    score_threshold: float,
    query: str,
    limit: Optional[int],
) -> ui.TagList:
    """Execute semantic search and return results UI.

    Args:
        vector_store: LangChain vector store
        base_url: Base URL for source links
        score_threshold: Minimum score threshold
        query: Search query string
        limit: Result limit

    Returns:
        TagList with semantic search results
    """
    validation_error = _validate_search_inputs(query, limit)
    if validation_error:
        return validation_error

    if limit is None:
        raise ValueError("Limit cannot be None after validation")

    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=limit)
    except Exception as e:
        return ui.TagList(
            ui.div(
                ui.p(f"Semantic search error: {str(e)}", class_="text-dark"),
                class_="alert alert-danger",
            )
        )

    if not results_with_scores:
        return ui.TagList(
            ui.div(
                ui.p("No results found", class_="text-dark"),
                class_="alert alert-info",
            )
        )

    metric_type = "similarity"
    if len(results_with_scores) >= 2:
        score1, score2 = results_with_scores[0][1], results_with_scores[1][1]
        if score1 < score2:
            metric_type = "distance"

    result_cards = []
    filtered_count = 0

    for idx, (doc, score) in enumerate(results_with_scores, 1):
        if score < score_threshold:
            filtered_count += 1
            continue

        score_class = "bg-success" if score > 0.7 else "bg-primary"
        if metric_type == "distance":
            score_class = "bg-primary" if score < 0.5 else "bg-secondary"

        card = _create_result_card(
            doc=doc,
            score=score,
            idx=idx,
            base_url=base_url,
            score_class=score_class,
            score_label=metric_type.title(),
            content=doc.page_content,
        )
        result_cards.append(card)

    return _create_results_list(result_cards, filtered_count, score_threshold)


def _execute_rag_answer(
    rag_chain: Optional[RAGChain],
    base_url: str,
    score_threshold: float,
    query: str,
    limit: Optional[int],
) -> ui.TagList:
    """Execute RAG answer generation and return results UI.

    Args:
        rag_chain: Optional RAG chain
        base_url: Base URL for source links
        score_threshold: Minimum score threshold (unused for RAG)
        query: Search query string
        limit: Result limit (unused for RAG)

    Returns:
        TagList with RAG answer and source documents
    """
    if rag_chain is None:
        return ui.TagList(
            ui.div(
                ui.p(
                    "RAG functionality not available (no RAG chain provided).",
                    class_="text-dark",
                ),
                class_="alert alert-info",
            )
        )

    validation_error = _validate_search_inputs(query, limit)
    if validation_error:
        return validation_error

    try:
        response = rag_chain.invoke(query)
    except Exception as e:
        return ui.TagList(
            ui.div(
                ui.p(f"RAG error: {str(e)}", class_="text-dark"),
                class_="alert alert-danger",
            )
        )

    answer_section = ui.card(
        ui.card_header(ui.strong("Answer"), class_="bg-success text-white"),
        ui.div(
            ui.p(response.answer, class_="mb-0"),
        ),
        class_="mb-3",
    )

    source_cards = []
    for idx, doc in enumerate(response.source_documents, 1):
        source = doc.metadata.get("source", "Unknown")
        last_modified = doc.metadata.get("last_modified", 0)

        if base_url and source:
            source_link = ui.a(
                source,
                href=f"{base_url}/{quote(source)}",
                target="_blank",
                class_="text-primary",
            )
        else:
            source_link = ui.span(source, class_="text-muted")

        modified_display = ""
        if last_modified:
            dt = datetime.fromtimestamp(last_modified)
            modified_display = dt.strftime("%Y-%m-%d %H:%M:%S")

        metadata_items = [
            ui.span(ui.strong("Source: "), source_link),
        ]
        if modified_display:
            metadata_items.append(
                ui.span(
                    ui.strong("Modified: "),
                    ui.span(modified_display, class_="text-muted"),
                )
            )

        source_card = ui.card(
            ui.card_header(
                ui.strong(f"Source {idx}"),
                class_="d-flex align-items-center",
            ),
            ui.div(
                ui.HTML(
                    f'<pre class="bg-light p-2 rounded mb-2" style="white-space: pre-wrap;">{doc.page_content}</pre>'
                ),
                ui.tags.ul(
                    *[ui.tags.li(item, class_="small") for item in metadata_items],
                    class_="list-unstyled mb-0",
                ),
            ),
            class_="mb-3",
        )
        source_cards.append(source_card)

    if not source_cards:
        source_section = ui.div(
            ui.p("No source documents available", class_="text-muted small"),
            class_="mt-2",
        )
    else:
        source_section = ui.div(
            ui.h5("Source Documents", class_="mt-3 mb-3"),
            *source_cards,
        )

    return ui.TagList(answer_section, source_section)


def create_shiny_app(
    vector_store: VectorStore,
    base_url: str = "",
    score_threshold: float = 0.0,
    bm25_retriever: Optional[BM25Indexer] = None,
    rag_chain: Optional[RAGChain] = None,
) -> App:
    """Create a Shiny app for semantic search.

    Args:
        vector_store: LangChain vector store
        base_url: Base URL for source file links
        score_threshold: Minimum score threshold for results
        bm25_retriever: Optional BM25 indexer for keyword search
        rag_chain: Optional RAG chain for question answering

    Returns:
        Configured Shiny App instance
    """
    app_ui = _create_search_ui()

    def server(input: Inputs, output: Outputs, session: Session) -> None:
        @render.ui
        @reactive.event(input.search)
        def bm25_results() -> ui.TagList:
            return _execute_bm25_search(
                bm25_retriever, base_url, score_threshold, input.query(), input.limit()
            )

        @render.ui
        @reactive.event(input.search)
        def results() -> ui.TagList:
            return _execute_semantic_search(
                vector_store, base_url, score_threshold, input.query(), input.limit()
            )

        @render.ui
        @reactive.event(input.search)
        def rag_answer() -> ui.TagList:
            return _execute_rag_answer(
                rag_chain, base_url, score_threshold, input.query(), input.limit()
            )

    return App(app_ui, server)
