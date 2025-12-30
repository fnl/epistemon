"""Shiny UI for semantic search."""

from langchain_core.vectorstores import VectorStore
from shiny import App, Inputs, Outputs, Session, ui


def create_shiny_app(
    vector_store: VectorStore,
    base_url: str = "",
    score_threshold: float = 0.0,
) -> App:
    """Create a Shiny app for semantic search.

    Args:
        vector_store: LangChain vector store
        base_url: Base URL for source file links
        score_threshold: Minimum score threshold for results

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
                    max=20,
                ),
                ui.input_action_button(
                    "search",
                    "Search",
                    class_="btn-primary",
                ),
                width=300,
            ),
            ui.output_ui("results"),
        ),
    )

    def server(input: Inputs, output: Outputs, session: Session) -> None:
        pass

    return App(app_ui, server)
