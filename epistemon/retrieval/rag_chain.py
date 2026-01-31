"""RAG chain for question answering over documents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import quote

from langchain_core.documents import Document


class RetrieverProtocol(Protocol):
    """Protocol for document retrievers."""

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores.

        Args:
            query: The search query

        Returns:
            List of (Document, score) tuples
        """
        ...


class LLMProtocol(Protocol):
    """Protocol for Language Models."""

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """Invoke the LLM with input.

        Args:
            input: The input to send to the LLM (prompt string or message sequence)
            **kwargs: Additional arguments for the LLM

        Returns:
            Response object with a .content attribute containing the generated text
        """
        ...


@dataclass
class RAGResponse:
    """Response from the RAG chain."""

    answer: str
    source_documents: list[Document]
    query: str


def load_default_prompt_template() -> str:
    """Load the default RAG prompt template from file.

    Loads from ./prompts/rag_answer_prompt.txt relative to the current working directory.

    Returns:
        The default prompt template string with {context} and {query} placeholders
    """
    default_path = Path("./prompts/rag_answer_prompt.txt")
    with default_path.open("r") as f:
        return f.read()


class RAGChain:
    """RAG chain that retrieves documents and generates answers using an LLM."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        llm: LLMProtocol,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initialize the RAG chain.

        Args:
            retriever: A retriever implementing the retrieve(query) method that
                returns list[tuple[Document, float]]. Examples include
                HybridRetriever, BM25Indexer, or VectorStore.as_retriever().
            llm: A language model implementing the invoke(prompt) method that
                returns an object with a .content attribute. Examples include
                ChatOpenAI, FakeListLLM, or any LangChain LLM.
            prompt_template: Template for the RAG prompt. Must contain {context}
                and {query} placeholders. If None, loads from ./prompts/rag_answer_prompt.txt
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or load_default_prompt_template()

    def format_context(self, documents: list[Document], base_url: str = "") -> str:
        """Format retrieved documents into context for the LLM.

        Args:
            documents: List of retrieved documents
            base_url: Optional base URL for constructing full URLs to sources

        Returns:
            Formatted context string with document content and metadata
        """
        if not documents:
            return ""

        formatted_docs = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            if base_url and source != "Unknown":
                full_url = f"{base_url}/{quote(source)}"
                formatted_docs.append(
                    f"Source: {source}\nURL: {full_url}\n{doc.page_content}"
                )
            else:
                formatted_docs.append(f"Source: {source}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_docs)

    def retrieve_documents(self, query: str) -> list[Document]:
        """Retrieve documents for a query, stripping relevance scores.

        Args:
            query: The user's question

        Returns:
            Flat list of Documents without scores
        """
        results = self.retriever.retrieve(query)
        return [doc for doc, _score in results]

    def generate_answer(
        self,
        query: str,
        source_documents: list[Document],
        base_url: str = "",
    ) -> RAGResponse:
        """Generate an answer from retrieved documents using the LLM.

        Args:
            query: The user's question
            source_documents: Documents to use as context
            base_url: Optional base URL for constructing full URLs to sources

        Returns:
            RAGResponse with the generated answer and source documents
        """
        context = self.format_context(source_documents, base_url=base_url)
        prompt = self.prompt_template.format(context=context, query=query)
        response = self.llm.invoke(prompt)
        return RAGResponse(
            answer=response.content,
            source_documents=source_documents,
            query=query,
        )

    def invoke(
        self, query: str, k: Optional[int] = None, base_url: str = ""
    ) -> RAGResponse:
        """Generate an answer to the query using retrieved documents.

        Args:
            query: The user's question
            k: Optional maximum number of documents to use for context.
                If None, uses all retrieved documents.
            base_url: Optional base URL for constructing full URLs to sources
                in the context provided to the LLM.

        Returns:
            RAGResponse with the generated answer and source documents
        """
        source_documents = self.retrieve_documents(query)

        if k is not None and k > 0:
            source_documents = source_documents[:k]

        if not source_documents:
            return RAGResponse(
                answer="No relevant documents were found to answer your question.",
                source_documents=[],
                query=query,
            )

        return self.generate_answer(query, source_documents, base_url=base_url)
