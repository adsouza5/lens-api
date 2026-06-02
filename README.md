# Lens — Semantic Code Search Engine

> Index any GitHub repo and search it in plain English. tree-sitter chunks code at function/class level, vectors are stored in Qdrant, and queries are ranked by cosine similarity in under a second.

Live at **[www.iadamdsouza.com](https://www.iadamdsouza.com)**

## How It Works

```
GitHub Repo URL
      │
      ▼
  Git clone  ─▶  tree-sitter parse  ─▶  function / class chunks
                                                │
                                    ┌───────────┼───────────┐
                               OpenAI      local model    Ollama
                          text-embed-3-small  (ST)     (fully local)
                                    └───────────┼───────────┘
                                                │  vector
                                          Qdrant collection
                                                │
                    Query  ─▶  embed  ─▶  cosine similarity search
                                                │
                                      Ranked code chunks
```

## Features

- **AST-aware chunking** — tree-sitter splits files at function/class boundaries, not arbitrary line counts
- **Three embedding backends** — `OpenAI text-embedding-3-small`, local Sentence Transformers, or Ollama (fully local, no external API calls)
- **Multi-language** — Python, JavaScript, TypeScript, Go, Rust, Java
- **Sub-second retrieval** — cosine similarity search over Qdrant HNSW index
- **Streaming indexing** — Server-Sent Events stream progress in real time while a repo is being indexed
- **Collection management** — list, query across, and delete indexed repos

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI · Python |
| Parsing | tree-sitter (6 language grammars) |
| Vector DB | Qdrant |
| Embeddings | OpenAI API · sentence-transformers · Ollama |
| Streaming | Server-Sent Events (sse-starlette) |
| Deployment | Cloud Run · Docker |

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/index` | Index a repo (SSE progress stream) |
| POST | `/search` | Semantic search |
| GET | `/collections` | List indexed repos |
| DELETE | `/collections/{name}` | Remove a collection |

## Local Development

```bash
git clone https://github.com/adsouza5/lens-api
cd lens-api
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY + QDRANT_URL
uvicorn main:app --reload
# API on :8000
```

```bash
# Index a repo
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo", "provider": "openai"}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"collection": "repo", "query": "retry failed HTTP requests", "top_k": 5}'
```

## Deployment

```bash
gcloud builds submit --tag gcr.io/<PROJECT>/lens-api
gcloud run deploy lens-api --image gcr.io/<PROJECT>/lens-api --allow-unauthenticated
```

## License

MIT
