"""Shiny UI for semantic search."""

from datetime import datetime
from typing import Optional
from urllib.parse import quote

from langchain_core.vectorstores import VectorStore
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from epistemon.indexing.bm25_indexer import BM25Indexer, highlight_keywords


def create_shiny_app(
    vector_store: VectorStore,
    base_url: str = "",
    score_threshold: float = 0.0,
    bm25_retriever: Optional[BM25Indexer] = None,
) -> App:
    """Create a Shiny app for semantic search.

    Args:
        vector_store: LangChain vector store
        base_url: Base URL for source file links
        score_threshold: Minimum score threshold for results
        bm25_retriever: Optional BM25 indexer for keyword search

    Returns:
        Configured Shiny App instance
    """
    app_ui = ui.page_fluid(
        ui.panel_title("Epistemon Semantic Search"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_text(
                    "query",
                    "Search Query",
                    placeholder="Enter your search query...",
                ),
                ui.input_numeric(
                    "limit",
                    "Result Limit",
                    value=5,
                    min=1,
                ),
                ui.input_action_button(
                    "search",
                    "Search",
                    class_="btn-primary",
                ),
                width=300,
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
            ),
        ),
    )

    def server(input: Inputs, output: Outputs, session: Session) -> None:
        @render.ui
        @reactive.event(input.search)
        def bm25_results() -> ui.TagList:
            if bm25_retriever is None:
                return ui.TagList(
                    ui.div(
                        ui.p("BM25 search not available", class_="text-dark"),
                        class_="alert alert-info",
                    )
                )

            query = input.query()
            limit = input.limit()

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

                source = doc.metadata.get("source", "Unknown")
                last_modified = doc.metadata.get("last_modified", 0)
                content = highlight_keywords(doc.page_content, query)

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

                score_class = "bg-info"

                card = ui.card(
                    ui.card_header(
                        ui.div(
                            ui.strong(f"Result {idx}", class_="me-2"),
                            ui.span(
                                f"Score: {score:.4f}",
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
                            *[
                                ui.tags.li(item, class_="small")
                                for item in metadata_items
                            ],
                            class_="list-unstyled mb-0",
                        ),
                    ),
                    class_="mb-3",
                )
                result_cards.append(card)

            if filtered_count > 0:
                alert = ui.div(
                    ui.p(
                        f"{filtered_count} result(s) filtered by score threshold "
                        f"({score_threshold})",
                        class_="mb-0",
                    ),
                    class_="alert alert-warning mb-3",
                )
                result_cards.insert(0, alert)

            header = ui.h3(
                f"Found {len(result_cards) - (1 if filtered_count > 0 else 0)} results",
                class_="mb-3",
            )
            result_cards.insert(0, header)

            return ui.TagList(*result_cards)

        @render.ui
        @reactive.event(input.search)
        def results() -> ui.TagList:
            query = input.query()
            limit = input.limit()

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

            results_with_scores = vector_store.similarity_search_with_score(
                query, k=limit
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

                source = doc.metadata.get("source", "Unknown")
                last_modified = doc.metadata.get("last_modified", 0)
                content = doc.page_content

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

                score_class = "bg-success" if score > 0.7 else "bg-primary"
                if metric_type == "distance":
                    score_class = "bg-primary" if score < 0.5 else "bg-secondary"

                card = ui.card(
                    ui.card_header(
                        ui.div(
                            ui.strong(f"Result {idx}", class_="me-2"),
                            ui.span(
                                f"{metric_type.title()}: {score:.4f}",
                                class_=f"badge {score_class} rounded-pill",
                            ),
                        ),
                        class_="d-flex align-items-center",
                    ),
                    ui.div(
                        ui.tags.pre(
                            content,
                            class_="bg-light p-2 rounded mb-2",
                            style="white-space: pre-wrap;",
                        ),
                        ui.tags.ul(
                            *[
                                ui.tags.li(item, class_="small")
                                for item in metadata_items
                            ],
                            class_="list-unstyled mb-0",
                        ),
                    ),
                    class_="mb-3",
                )
                result_cards.append(card)

            if filtered_count > 0:
                alert = ui.div(
                    ui.p(
                        f"{filtered_count} result(s) filtered by score threshold "
                        f"({score_threshold})",
                        class_="mb-0",
                    ),
                    class_="alert alert-warning mb-3",
                )
                result_cards.insert(0, alert)

            header = ui.h3(
                f"Found {len(result_cards) - (1 if filtered_count > 0 else 0)} results",
                class_="mb-3",
            )
            result_cards.insert(0, header)

            return ui.TagList(*result_cards)

    return App(app_ui, server)
