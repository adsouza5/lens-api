# Lens — Semantic Code Search Engine

> Index any GitHub repository and query it in plain English. Sub-second vector similarity search across functions and classes.

## Overview

Lens clones a GitHub repository, chunks the source code at the **function and class level** using **tree-sitter**, embeds each chunk with your choice of embedding model, and stores the vectors in **Qdrant**. Natural-language queries are embedded at search time and ranked by cosine similarity — results include file path, line numbers, symbol name, and relevance score.

Three embedding back-ends are supported: a self-hosted **jina-embeddings-v2-base-code** model, **OpenAI text-embedding-3-small**, or a fully local **Ollama** model.

## Features

- **Semantic search** — find code by intent, not just keywords
- **tree-sitter chunking** — function/class-level granularity, not line-by-line
- **Three embedding models** — jina (self-hosted), OpenAI, Ollama (local)
- **Qdrant vector store** — sub-second approximate nearest-neighbour search
- **FastAPI** — REST indexing and search endpoints with OpenAPI docs
- **React UI** — model selector, live indexing progress, result cards with file path, line numbers, score, and code preview
- **Cloud Run deployment** — containerised, auto-scaling

## Stack

| Layer | Technology |
|---|---|
| Parser | tree-sitter (Python, JS, TS, Go, Rust, Java, C/C++) |
| Embeddings | jina-embeddings-v2-base-code / OpenAI / Ollama |
| Vector DB | Qdrant |
| API | FastAPI (Python) |
| Frontend | React |
| Deployment | Cloud Run, Docker |

## Architecture

```
GitHub Repo URL
      ↓
   git clone
      ↓
tree-sitter chunker → [function/class chunks]
      ↓
embedding model (jina / OpenAI / Ollama)
      ↓
Qdrant (upsert vectors)

Query (plain English)
      ↓
embedding model
      ↓
Qdrant ANN search → ranked results
```

## Live Demo

Available at [adamdsouza.com](https://adamdsouza.com) → Lens project card.

## License

MIT
