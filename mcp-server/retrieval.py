"""Milvus collection with both a dense vector and a BM25 sparse vector per chunk.

Milvus computes the BM25 vector itself from the `text` field (a BM25 function
on the collection), so inserts only need the text and the dense embedding.
Search can then go dense only, BM25 only, or both merged with RRF.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

logger = logging.getLogger("mcp-server.retrieval")

MODES = ("dense", "bm25", "hybrid")
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
OUTPUT_FIELDS = ["text", "filename", "domain", "chunk_id"]

# lowercase + English stemming, so "datasets" matches "dataset"
ANALYZER = {
    "tokenizer": "standard",
    "filter": ["lowercase", {"type": "stemmer", "language": "english"}],
}


def create_collection(client: MilvusClient, name: str, dim: int, dense_index: dict) -> None:
    schema = client.create_schema(auto_id=True)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=ANALYZER)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("filename", DataType.VARCHAR, max_length=1024)
    schema.add_field("domain", DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_function(Function(
        name="text_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse"],
    ))

    index_params = client.prepare_index_params()
    index_params.add_index("vector", **dense_index)
    index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(name, schema=schema, index_params=index_params)


def _hits(res: Any) -> list[dict[str, Any]]:
    out = []
    for hit in res[0] if res else []:
        ent = hit.get("entity", {})
        out.append({
            "content": ent.get("text", ""),
            "filename": ent.get("filename", ""),
            "domain": ent.get("domain", ""),
            "chunk_id": int(ent.get("chunk_id", -1)),
            "score": float(hit.get("distance", 0.0)),
        })
    return out


def search(
    client: MilvusClient,
    collection: str,
    query: str,
    query_vector: list[float] | None,
    *,
    mode: str,
    top_k: int,
    expr: str | None,
    dense_params: dict,
    candidates: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    """query_vector is only needed for dense and hybrid."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    filt = expr or ""

    if mode == "dense":
        res = client.search(collection, data=[query_vector], anns_field="vector", limit=top_k,
                            filter=filt, search_params=dense_params, output_fields=OUTPUT_FIELDS)
        return _hits(res)

    if mode == "bm25":
        res = client.search(collection, data=[query], anns_field="sparse", limit=top_k,
                            filter=filt, search_params={"metric_type": "BM25"}, output_fields=OUTPUT_FIELDS)
        return _hits(res)

    # hybrid: take a deeper candidate list from each side, then merge by rank.
    # RRF only looks at positions, so cosine and BM25 scores never have to be put on one scale.
    depth = max(top_k, candidates)
    requests = [
        AnnSearchRequest([query_vector], "vector", dense_params, limit=depth, expr=filt or None),
        AnnSearchRequest([query], "sparse", {"metric_type": "BM25"}, limit=depth, expr=filt or None),
    ]
    res = client.hybrid_search(collection, requests, RRFRanker(rrf_k), limit=top_k, output_fields=OUTPUT_FIELDS)
    return _hits(res)


class RerankError(RuntimeError):
    pass


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    model: str,
    api_key: str,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Re-score candidates with a cross-encoder (Voyage rerank API) and keep the best top_k.

    Unlike the vector search, the reranker reads the query and each passage
    together, so it can tell "on topic" from "actually answers the question".
    """
    if not hits:
        return []
    if not api_key:
        raise RerankError("VOYAGE_API_KEY is not set")
    body = {"query": query, "documents": [h["content"] for h in hits], "model": model, "top_k": top_k}
    headers = {"Authorization": f"Bearer {api_key}"}

    resp = None
    for attempt in range(3):
        try:
            resp = requests.post(VOYAGE_RERANK_URL, json=body, headers=headers, timeout=timeout_s)
        except requests.RequestException as e:
            if attempt == 2:
                raise RerankError(f"rerank request failed: {e!r}") from e
            time.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(1.0 * (attempt + 1))
            continue
        break
    if resp is None or not resp.ok:
        detail = resp.text[:300] if resp is not None else "no response"
        raise RerankError(f"rerank failed: {getattr(resp, 'status_code', '?')} {detail}")

    payload = resp.json()
    ranked = payload.get("data", payload) if isinstance(payload, dict) else payload
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    logger.info("rerank model=%s docs=%d tokens=%s", model, len(hits), usage.get("total_tokens"))

    out = []
    for item in ranked[:top_k]:
        hit = dict(hits[item["index"]])
        hit["retrieval_score"] = hit["score"]
        hit["score"] = float(item["relevance_score"])
        out.append(hit)
    return out
