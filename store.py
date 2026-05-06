from __future__ import annotations
import os
import re
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
)
from chunker import Chunk


def _collection_name(repo_url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", repo_url.lower().rstrip("/"))
    return f"lens-{slug[-60:]}"


def get_client() -> AsyncQdrantClient:
    url    = os.environ["QDRANT_URL"]
    api_key = os.getenv("QDRANT_API_KEY")
    return AsyncQdrantClient(url=url, api_key=api_key)


async def ensure_collection(client: AsyncQdrantClient, name: str, dims: int) -> None:
    existing = {c.name for c in (await client.get_collections()).collections}
    if name not in existing:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )


async def upsert_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    offset: int = 0,
) -> None:
    points = [
        PointStruct(
            id=offset + i,
            vector=vec,
            payload={
                "file_path":  c.file_path,
                "language":   c.language,
                "start_line": c.start_line,
                "end_line":   c.end_line,
                "symbol":     c.symbol,
                "content":    c.content,
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, embeddings))
    ]
    await client.upsert(collection_name=collection, points=points)


async def search(
    client: AsyncQdrantClient,
    collection: str,
    query_vec: list[float],
    top_k: int = 8,
    language: str | None = None,
) -> list[dict]:
    filt = None
    if language:
        filt = Filter(must=[FieldCondition(key="language", match=MatchValue(value=language))])

    hits = await client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=top_k,
        query_filter=filt,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in hits]


async def list_collections(client: AsyncQdrantClient) -> list[str]:
    cols = await client.get_collections()
    return [c.name for c in cols.collections if c.name.startswith("lens-")]


def repo_collection(repo_url: str) -> str:
    return _collection_name(repo_url)
