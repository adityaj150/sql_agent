# Advanced RAG Pipeline Tasks

This document tracks our progress through Phase 1 and Phase 2 of building the advanced hybrid RAG and SQL agent system.

## Phase 1: Standalone RAG Agent

- `[x]` **1. Folder Structure Setup**
  - Create `documents/` directory and add sample PDFs/MD/TXT.
  - Create `data/` directory and move `demo.sqlite` there.
  - Create empty placeholder files (`embeddings.py`, `retriever.py`, `chunking.py`, `reranker.py`, `compressor.py`, `evaluator.py`, `ingest.py`).

- `[ ]` **2. Context Aware Chunking & Ingestion (`chunking.py`, `ingest.py`)**
  - Implement PDF, TXT, and Markdown loaders.
  - Implement context-aware splitters (e.g., `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`).

- `[ ]` **3. Embeddings & Vector Store (`embeddings.py`)**
  - Implement BGE Embeddings using `HuggingFaceBgeEmbeddings`.
  - Set up a local vector store (Chroma or FAISS) and save it to the `vectorstore/` directory.

- `[ ]` **4. BM25 & Hybrid Retrieval (`retriever.py`)**
  - Implement `BM25Retriever` for keyword search.
  - Combine Vector Store retriever and BM25 retriever into an `EnsembleRetriever`.

- `[ ]` **5. Cross Encoder Re-ranking (`reranker.py`)**
  - Implement a `CrossEncoder` model to score and re-rank the hybrid results.

- `[ ]` **6. Context Compression (`compressor.py`)**
  - Implement LangChain's compression capabilities to shrink the context before feeding it to the LLM.

- `[ ]` **7. The RAG Chatbot (`rag_agent.py`)**
  - Wire all the modules (`chunking` -> `embeddings` -> `retriever` -> `reranker` -> `compressor`) together.
  - Connect it to the LLM to provide final answers.

- `[ ]` **8. RAGAS Evaluation (`evaluator.py`)**
  - Implement metrics (`context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`).
  - Create a small test dataset to evaluate the RAG pipeline.

## Phase 2: Final Integration

- `[ ]` **9. Multi-Agent Routing (`multi_agent.py`)**
  - Implement an LLM router to decide whether a query goes to the `SQL Agent` or the `RAG Agent`.
  - Wire the output to provide a seamless answer.

- `[ ]` **10. Final Verification**
  - Test asking SQL-specific questions (e.g. revenue).
  - Test asking Policy-specific questions (e.g. refunds).
