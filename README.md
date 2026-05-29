# Lens — Semantic Code Search Engine

Index any Git repository and search it with natural language. Lens parses source files with tree-sitter (AST-aware chunking), embeds them with OpenAI or local sentence-transformers, stores vectors in Qdrant, and returns semantically similar code chunks ranked by relevance.

**[Live Demo](https://adsouza5.github.io/portfolio-react/projects/lens)**

## How It Works

```
GitHub Repo URL
      │
      ▼
  Git clone  ──▶  tree-sitter parse  ──▶  AST-aware chunks
                                                │
                                     Embedding model (OpenAI / local)
                                                │
                                          Qdrant collection
                                                │
                               Query ──▶  embed query ──▶  vector search
                                                │
                                      Ranked code results
```

## Features

- **AST-aware chunking** — tree-sitter splits files at function/class boundaries, not arbitrary line counts
- **Multi-language** — Python, JavaScript, TypeScript, Go, Rust, Java
- **Flexible embeddings** — OpenAI `text-embedding-3-small`, local Sentence Transformers, or Ollama
- **Streaming indexing** — Server-Sent Events stream progress in real time while the repo is being indexed
- **Collection management** — list, search across, and delete indexed repos

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Python |
| Parsing | tree-sitter (6 language grammars) |
| Vector DB | Qdrant |
| Embeddings | OpenAI API, sentence-transformers |
| Streaming | Server-Sent Events (sse-starlette) |
| Deployment | Cloud Run, Docker |

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/index` | Index a repo (SSE stream) |
| POST | `/search` | Semantic search |
| GET | `/collections` | List indexed repos |
| DELETE | `/collections/{name}` | Remove a collection |

## Local Development

```bash
git clone https://github.com/adsouza5/lens-api
cd lens-api
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
# API on :8000
```

```bash
# Index a repo
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo", "provider": "openai", "api_key": "sk-..."}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"collection": "repo", "query": "authentication middleware", "top_k": 5}'
```

## Deployment

```bash
gcloud builds submit --tag <IMAGE>
gcloud run deploy lens-api --image <IMAGE> --allow-unauthenticated
```
