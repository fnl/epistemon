"""RAG chain for question answering over documents."""

from dataclasses import dataclass
from typing import Any, Protocol

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

    def invoke(self, prompt: str) -> Any:
        """Invoke the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM

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


DEFAULT_PROMPT_TEMPLATE = """Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""


class RAGChain:
    """RAG chain that retrieves documents and generates answers using an LLM."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        llm: LLMProtocol,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
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
                and {query} placeholders. Defaults to a standard question-answering
                template.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template

    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into context for the LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string with document content and metadata
        """
        if not documents:
            return ""

        formatted_docs = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            formatted_docs.append(f"Source: {source}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_docs)

    def invoke(self, query: str) -> RAGResponse:
        """Generate an answer to the query using retrieved documents.

        Args:
            query: The user's question

        Returns:
            RAGResponse with the generated answer and source documents
        """
        results = self.retriever.retrieve(query)
        source_documents = [doc for doc, _score in results]

        if not source_documents:
            return RAGResponse(
                answer="No relevant documents were found to answer your question.",
                source_documents=[],
                query=query,
            )

        context = self.format_context(source_documents)
        prompt = self.prompt_template.format(context=context, query=query)

        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        return RAGResponse(
            answer=answer, source_documents=source_documents, query=query
        )
