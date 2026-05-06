from __future__ import annotations
import os
from typing import Protocol, runtime_checkable
import httpx


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def dimensions(self) -> int: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    @property
    def dimensions(self) -> int:
        return 1536 if "small" in self._model else 3072

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key)
        resp = await client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]


class LocalProvider:
    _model = None

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"local/{self._model_name.split('/')[-1]}"

    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()

        def _encode():
            if LocalProvider._model is None:
                from sentence_transformers import SentenceTransformer
                LocalProvider._model = SentenceTransformer(
                    self._model_name, trust_remote_code=True, local_files_only=True
                )
            return LocalProvider._model.encode(texts).tolist()

        return await loop.run_in_executor(None, _encode)


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            results = []
            for text in texts:
                resp = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                results.append(resp.json()["embedding"])
            return results


def build_provider(provider: str, api_key: str | None, ollama_url: str | None) -> EmbeddingProvider:
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI provider requires an API key")
        return OpenAIProvider(key)
    if provider == "ollama":
        return OllamaProvider(base_url=ollama_url or "http://localhost:11434")
    return LocalProvider()
