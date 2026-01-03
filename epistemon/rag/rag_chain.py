"""RAG chain for question answering over documents."""

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class RAGResponse:
    """Response from the RAG chain."""

    answer: str
    source_documents: list[Document]
    query: str


class RAGChain:
    """RAG chain that retrieves documents and generates answers using an LLM."""

    def __init__(self, retriever: Any, llm: Any) -> None:
        self.retriever = retriever
        self.llm = llm

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

        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content

        return RAGResponse(
            answer=answer, source_documents=source_documents, query=query
        )
