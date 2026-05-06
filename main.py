from __future__ import annotations
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import chunker
import store
from embeddings import build_provider, LocalProvider

app = FastAPI(title="Lens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

BATCH_SIZE = 32


@app.on_event("startup")
async def warm_model():
    loop = asyncio.get_event_loop()
    def _load():
        from sentence_transformers import SentenceTransformer
        if LocalProvider._model is None:
            LocalProvider._model = SentenceTransformer(
                "jinaai/jina-embeddings-v2-base-code",
                trust_remote_code=True,
                local_files_only=True,
            )
    await loop.run_in_executor(None, _load)


# ── Models ────────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    repo_url:   str
    provider:   str = "local"
    api_key:    str | None = None
    ollama_url: str | None = None

class SearchRequest(BaseModel):
    query:      str
    collection: str
    provider:   str = "local"
    api_key:    str | None = None
    ollama_url: str | None = None
    top_k:      int = 8
    language:   str | None = None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Collections ───────────────────────────────────────────────────────────────

@app.get("/collections")
async def get_collections():
    client = store.get_client()
    cols = await store.list_collections(client)
    return {"collections": cols}


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    if not name.startswith("lens-"):
        raise HTTPException(status_code=400, detail="Can only delete lens-* collections")
    client = store.get_client()
    await client.delete_collection(name)
    return {"deleted": name}


# ── Index ─────────────────────────────────────────────────────────────────────

@app.post("/index")
async def index_repo(req: IndexRequest):
    async def stream():
        def event(data: dict) -> str:
            return json.dumps(data)

        tmpdir = tempfile.mkdtemp()
        try:
            yield {"data": event({"type": "status", "message": "Cloning repository…"})}
            await asyncio.sleep(0)

            try:
                import git
                git.Repo.clone_from(req.repo_url, tmpdir, depth=1)
            except Exception as e:
                yield {"data": event({"type": "error", "message": f"Clone failed: {e}"})}
                return

            yield {"data": event({"type": "status", "message": "Scanning files…"})}
            await asyncio.sleep(0)

            root = Path(tmpdir)
            files = chunker.iter_repo_files(root)
            total_files = len(files)

            if total_files == 0:
                yield {"data": event({"type": "error", "message": "No indexable source files found."})}
                return

            yield {"data": event({
                "type": "progress",
                "message": f"Found {total_files} source files",
                "files_total": total_files,
                "files_done": 0,
            })}
            await asyncio.sleep(0)

            try:
                provider = build_provider(req.provider, req.api_key, req.ollama_url)
            except ValueError as e:
                yield {"data": event({"type": "error", "message": str(e)})}
                return

            collection = store.repo_collection(req.repo_url)
            client = store.get_client()

            # Drop and recreate so re-indexing is always clean
            existing = {c.name for c in (await client.get_collections()).collections}
            if collection in existing:
                await client.delete_collection(collection)
            await store.ensure_collection(client, collection, provider.dimensions)

            all_chunks: list[chunker.Chunk] = []
            for i, fp in enumerate(files, start=1):
                all_chunks.extend(chunker.chunk_file(fp, root))
                if i % 10 == 0 or i == total_files:
                    yield {"data": event({
                        "type": "progress",
                        "message": f"Parsed {i}/{total_files} files — {len(all_chunks)} chunks",
                        "files_total": total_files,
                        "files_done": i,
                        "chunks": len(all_chunks),
                    })}
                    await asyncio.sleep(0)

            total_chunks = len(all_chunks)
            yield {"data": event({
                "type": "status",
                "message": f"Embedding {total_chunks} chunks with {provider.name}…",
            })}
            await asyncio.sleep(0)

            point_offset = 0
            for batch_start in range(0, total_chunks, BATCH_SIZE):
                batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
                texts = [c.content for c in batch]
                try:
                    vecs = await provider.embed(texts)
                except Exception as e:
                    yield {"data": event({"type": "error", "message": f"Embedding failed: {e}"})}
                    return

                try:
                    await store.upsert_chunks(client, collection, batch, vecs, offset=point_offset)
                except Exception as e:
                    yield {"data": event({"type": "error", "message": f"Qdrant upsert failed: {e}"})}
                    return
                point_offset += len(batch)

                done = min(batch_start + BATCH_SIZE, total_chunks)
                yield {"data": event({
                    "type": "embedding_progress",
                    "message": f"Embedded {done}/{total_chunks} chunks",
                    "chunks_total": total_chunks,
                    "chunks_done": done,
                })}
                await asyncio.sleep(0)

            yield {"data": event({
                "type": "done",
                "message": f"Indexed {total_chunks} chunks from {total_files} files",
                "collection": collection,
                "repo_url": req.repo_url,
                "chunks": total_chunks,
                "files": total_files,
                "provider": provider.name,
            })}

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return EventSourceResponse(stream())


# ── Search ────────────────────────────────────────────────────────────────────

@app.post("/search")
async def search_repo(req: SearchRequest):
    try:
        provider = build_provider(req.provider, req.api_key, req.ollama_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        [query_vec] = await provider.embed([req.query])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    client = store.get_client()
    try:
        results = await store.search(
            client, req.collection, query_vec, req.top_k, req.language
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e}")

    return {"results": results, "query": req.query, "provider": provider.name}
