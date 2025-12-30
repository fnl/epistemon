# Project Roadmap

The project is expected to evolve in three stages.

## V1: Semantic Search

A very basic semantic search engine over Markdown files:

- It creates a vector store over MD file chunks.
- A web frontend can be used to submit queries to the vector store.
- Matching is similarity-based, comparing the embeddings of the query and chunks.
- The vector store can be updated by re-running the indexing.
- In parallel, provide keyword-based search results to compare (BM25).

## V2: RAG Agent

A RAG Agent that executes the searches over the vector store from V1.
Instead of just returning the matching chunks, the Agent generates an answer.
So this is a two-step RAG chain that uses a single embedding and a single completion call per user query.
This is a fast and effective method for simple RAG queries.

## V3: Research Agent

The Agent now refines the user's query to properly match their intention before embedding the query. And the Agent can evaluate if the returned documents are relevant in the first place, once more leading to a refined query. The Agent can make keyword-based searches, to enrich the context. Only once the agent is satisfied with the query and the retrieved documents it proceeds to the answer generation stage.
