# About Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, is a technique that lets a language model answer questions using information from your own documents. Instead of relying only on what the model memorised during training, a RAG system first searches a document collection and then gives the most relevant passages to the model as context.

## Why use RAG

RAG reduces hallucinations because the model is told to answer from supplied text. It also keeps answers up to date, since you can add or change documents without retraining the model. Another benefit is traceability: the system can show which source files were used for each answer.

## How the pipeline works

First, documents are loaded and split into small chunks, usually a hundred to five hundred words each, with some overlap so that sentences are not cut off from their context. Second, the chunks are indexed. Simple systems use keyword scoring such as BM25, while advanced systems use vector embeddings stored in a vector database. Third, when a user asks a question, the retriever scores every chunk and returns the top few matches. Finally, the question and the retrieved chunks are sent to the language model, which writes the answer.

## BM25 retrieval

BM25 is a ranking function that scores chunks by how often query words appear in them, how rare those words are across the whole collection, and how long the chunk is. Rare words contribute more to the score than common words. BM25 needs no training and works well for small projects.

## Limitations

Keyword retrieval can miss answers that use different wording from the question. Embedding-based retrieval handles synonyms better but needs an embedding model. Chunk size matters: chunks that are too small lose context, and chunks that are too large dilute relevance.
