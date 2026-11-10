"""Query/document embeddings with an explicit provider and vector dimension."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Protocol

import requests


class Embeddings(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimension: int | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()
        default = "voyage-4" if provider == "voyage" else os.getenv("GEMINI_EMB_MODEL", "gemini-embedding-001")
        model = os.getenv("EMBEDDING_MODEL") or default
        dimension = os.getenv("EMBEDDING_DIMENSION", "")
        return cls(provider, model, int(dimension) if dimension else (1024 if provider == "voyage" else None))

    def metadata(self, dimension: int) -> dict:
        return {"provider": self.provider, "model": self.model, "dimension": dimension}


def validate_collection(config: EmbeddingConfig, description: dict) -> None:
    dimension = int(next(f for f in description["fields"] if f["name"] == "vector")["params"]["dim"])
    if config.dimension is not None and config.dimension != dimension:
        raise ValueError("Embedding dimension does not match the Milvus collection")
    marker = description.get("description", "")
    if marker.startswith("embedding:"):
        if json.loads(marker.removeprefix("embedding:")) != config.metadata(dimension):
            raise ValueError("Embedding provider/model does not match collection metadata")
    elif config.provider != "google":
        raise ValueError("Collection lacks embedding identity; migrate into a new collection first")


class VoyageEmbeddings:
    """Use the ordinary embeddings endpoint, with bounded transient retries.

    No truncation: an oversized document must fail instead of silently losing text.
    HTTP authentication/permission errors are not retried or logged with credentials.
    """
    def __init__(self, model: str = "voyage-4", dimension: int = 1024,
                 api_key: str | None = None, timeout: float = 60.0) -> None:
        self.model, self.dimension, self.timeout = model, dimension, timeout
        self.api_key = (api_key if api_key is not None else os.getenv("VOYAGE_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is not set")
        if dimension not in (256, 512, 1024, 2048):
            raise ValueError("Voyage output dimension must be 256, 512, 1024 or 2048")
        self.total_tokens = 0
        self.successful_requests = 0

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > 1000:
            raise ValueError("Voyage embedding accepts at most 1000 texts per request")
        payload = {"input": texts, "model": self.model, "input_type": input_type,
                   "output_dimension": self.dimension, "output_dtype": "float", "truncation": False}
        for attempt in range(4):
            try:
                response = requests.post("https://api.voyageai.com/v1/embeddings",
                                         headers={"Authorization": f"Bearer {self.api_key}"},
                                         json=payload, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"Voyage embedding transport failed: {type(exc).__name__}") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 3:
                    try:
                        wait = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                    except ValueError:
                        wait = 2 ** (attempt + 1)
                    time.sleep(max(0, min(wait, 30)))
                    continue
            if not response.ok:
                raise RuntimeError(f"Voyage embedding HTTP {response.status_code}; check account access, quota and input limits")
            body = response.json()
            if body.get("model") != self.model:
                raise RuntimeError("Voyage returned a different embedding model")
            data = body.get("data", [])
            if len(data) != len(texts) or {d.get("index") for d in data} != set(range(len(texts))):
                raise RuntimeError("Voyage embedding indices/count do not match input")
            vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
            for vector in vectors:
                if len(vector) != self.dimension or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                    raise RuntimeError("Voyage returned an invalid embedding vector")
            self.total_tokens += int(body.get("usage", {}).get("total_tokens", 0))
            self.successful_requests += 1
            return vectors
        raise RuntimeError("Voyage embedding retries exhausted")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")


def create_embeddings(config: EmbeddingConfig | None = None) -> Embeddings:
    config = config or EmbeddingConfig.from_env()
    if config.provider == "voyage":
        return VoyageEmbeddings(config.model, config.dimension or 1024)
    if config.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if config.dimension is not None:
            raise ValueError("Explicit EMBEDDING_DIMENSION is currently supported only for Voyage")
        return GoogleGenerativeAIEmbeddings(model=config.model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {config.provider}")
