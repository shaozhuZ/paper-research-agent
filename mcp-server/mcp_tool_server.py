from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote_plus

import requests as http_requests
from fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import MilvusClient

import retrieval
from chunking import split_pdf

#—FastMCP——
mcp = FastMCP("Research Assistant Tools")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("mcp-server")

#—Config (env-driven)——
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION = os.getenv("MILVUS_COLLECTION", "research_papers_v2")
INDEX_TYPE = os.getenv("INDEX_TYPE", "HNSW").upper()
TOP_K = int(os.getenv("TOP_K", "4"))
# what vector_search (the agent's tool) uses: dense | bm25 | hybrid
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")
# hybrid: how many hits each side contributes before RRF, and the RRF constant
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "50"))
RRF_K = int(os.getenv("RRF_K", "60"))
# rerank: re-score this many candidates from the first-stage search with a cross-encoder
RETRIEVAL_RERANK = os.getenv("RETRIEVAL_RERANK", "false").lower() in ("1", "true", "yes")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-3")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
EMB_MODEL = os.getenv("GEMINI_EMB_MODEL", "gemini-embedding-001")
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:7000")
PORT = int(os.getenv("MCP_PORT", "9000"))
DOMAINS = ("AI", "Security", "Other")
MAX_TOP_K = 50
# mcp-host writes uploaded PDFs here; index_paper only reads from inside it
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads")).resolve()

#—Milvus setup——
def _build_index_params() -> tuple[Dict, Dict]:
    if INDEX_TYPE == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    elif INDEX_TYPE == "DISKANN":
        index_params = {"index_type": "DISKANN", "metric_type": "COSINE", "params": {}}
        search_params = {
            "metric_type": "COSINE",
            "params": {"search_list": max(16, TOP_K)},
        }
    elif INDEX_TYPE == "IVF_PQ":
        index_params = {
            "index_type": "IVF_PQ",
            "metric_type": "COSINE",
            "params": {"nlist": 128, "m": 16, "nbits": 8},
        }
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    else:
        raise ValueError(f"Unsupported INDEX_TYPE: {INDEX_TYPE}. Use HNSW, DISKANN, or IVF_PQ.")

    return index_params, search_params


_client: MilvusClient | None = None
_embeddings: GoogleGenerativeAIEmbeddings | None = None


_loaded = False


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or "")
    return _client


def ensure_loaded() -> None:
    # search needs the collection in memory; a Milvus restart can leave it released
    global _loaded
    if not _loaded:
        get_client().load_collection(COLLECTION)
        _loaded = True


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMB_MODEL)
    return _embeddings


def _search(query: str, domain: str, top_k: int, mode: str, rerank: bool = False) -> Dict[str, Any]:
    if mode not in retrieval.MODES:
        raise ValueError(f"mode must be one of {retrieval.MODES}")
    # domain goes into a filter expression, so only allow known values
    if domain != "All" and domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS} or 'All'")
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    # with rerank, fetch a wider first-stage list and let the reranker pick from it
    first_k = max(top_k, RERANK_CANDIDATES) if rerank else top_k
    expr = None if domain == "All" else f'domain == "{domain}"'
    # BM25 alone doesn't need the query embedding, so skip that API call
    qvec = None if mode == "bm25" else get_embeddings().embed_query(query)
    _, dense_params = _build_index_params()
    ensure_loaded()
    results = retrieval.search(
        get_client(), COLLECTION, query, qvec,
        mode=mode, top_k=first_k, expr=expr, dense_params=dense_params,
        candidates=HYBRID_CANDIDATES, rrf_k=RRF_K,
    )
    reranked = False
    if rerank:
        try:
            results = retrieval.rerank(query, results, top_k=top_k, model=RERANK_MODEL, api_key=VOYAGE_API_KEY)
            reranked = True
        except retrieval.RerankError as e:
            # keep answering with the first-stage order rather than failing the search
            logger.warning("rerank unavailable, using %s order: %s", mode, e)
            results = results[:top_k]
    return {"results": results, "reranked": reranked}


#—Tools——
@mcp.tool
def translate(text: str, source_lang: str, target_lang: str) -> Dict[str, str]:
    """Translate text between English, Spanish, French, and Italian."""
    if source_lang == target_lang:
        return {"translated_text": text}
    resp = http_requests.post(
        f"{TRANSLATION_SERVICE_URL}/translate",
        json={"text": text, "source_lang": source_lang, "target_lang": target_lang},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


@mcp.tool
def search_paper_url(title: str) -> Dict[str, str]:
    logger.info("search_paper_url title=%r", title)

    search_fallback_url = f"https://www.semanticscholar.org/search?q={quote_plus(title)}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AcademicResearch/1.0)"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    try:
        resp = None
        for attempt in range(3):
            resp = http_requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "fields": "title,url", "limit": 1},
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 429:
                break
            wait_s = 1.5 * (attempt + 1)
            logger.info("semantic scholar 429, retry in %.1fs", wait_s)
            time.sleep(wait_s)

        if resp is None:
            return {"title": title, "url": search_fallback_url}

        if resp.status_code == 429:
            logger.warning("semantic scholar still rate-limited, using search URL fallback")
            return {"title": title, "url": search_fallback_url}

        data = resp.json()
        papers = data.get("data", [])
        if not papers:
            return {"title": title, "url": search_fallback_url}
        return {
            "title": papers[0].get("title", title),
            "url": papers[0].get("url", search_fallback_url),
        }
    except Exception as e:
        logger.warning("search_paper_url failed: %r", e)
        return {"title": title, "url": search_fallback_url}


@mcp.tool
def index_paper(path: str, filename: str, domain: str) -> Dict[str, Any]:
    """Index a PDF that has already been saved under UPLOAD_DIR.

    The file is passed by path, not by content: a PDF of a few MB sent as
    base64 goes over the MCP message size limit and the request is rejected.
    """
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}")
    pdf_path = Path(path).resolve()
    if not pdf_path.is_relative_to(UPLOAD_DIR):
        raise ValueError("path must be inside the upload directory")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"no such upload: {pdf_path.name}")

    chunks = split_pdf(pdf_path.read_bytes(), filename, domain)
    vectors = get_embeddings().embed_documents([c.page_content for c in chunks])
    client = get_client()
    if not client.has_collection(COLLECTION):
        index_params, _ = _build_index_params()
        retrieval.create_collection(client, COLLECTION, len(vectors[0]), index_params)
    client.insert(COLLECTION, [
        {"text": c.page_content, "vector": v, "filename": filename, "domain": domain,
         "chunk_id": c.metadata["chunk_id"]}
        for c, v in zip(chunks, vectors)
    ])

    return {"filename": filename, "domain": domain, "chunks_indexed": len(chunks)}


@mcp.tool
def vector_search(query: str, domain: str = "All", top_k: int = 4) -> Dict[str, Any]:
    """Search indexed paper chunks. domain: AI | Security | Other | All."""
    return _search(query, domain, top_k, RETRIEVAL_MODE, RETRIEVAL_RERANK)


@mcp.tool
def retrieve(
    query: str, domain: str = "All", top_k: int = 4, mode: str = "dense", rerank: bool = False
) -> Dict[str, Any]:
    """Same as vector_search with an explicit mode (dense | bm25 | hybrid) and optional rerank. For evaluation."""
    return _search(query, domain, top_k, mode, rerank)


@mcp.tool
def get_stats() -> Dict[str, int]:
    """Number of distinct indexed papers per domain."""
    client = get_client()
    counts: Dict[str, int] = {"AI": 0, "Security": 0, "Other": 0}

    try:
        if not client.has_collection(collection_name=COLLECTION):
            logger.info("collection %s not found", COLLECTION)
            return counts
        client.load_collection(collection_name=COLLECTION)
    except Exception as e:
        logger.warning("get_stats: milvus unavailable: %r", e)
        return counts

    for domain in counts:
        try:
            res = client.query(
                collection_name=COLLECTION,
                filter=f'domain == "{domain}"',
                output_fields=["filename"],
                limit=16384,
            )
            filenames = {
                str(row.get("filename", "")).strip()
                for row in res
                if isinstance(row, dict) and str(row.get("filename", "")).strip()
            }
            counts[domain] = len(filenames)
        except Exception as e:
            logger.warning("get_stats query failed for %s: %r", domain, e)
            counts[domain] = 0

    return counts

#—Entry——
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)