from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List
from urllib.parse import quote_plus

import requests as http_requests
from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pymilvus import MilvusClient

#—FastMCP——
mcp = FastMCP("Research Assistant Tools")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("mcp-server")

#—Config (env-driven)——
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION = os.getenv("MILVUS_COLLECTION", "research_papers")
INDEX_TYPE = os.getenv("INDEX_TYPE", "HNSW").upper()
TOP_K = int(os.getenv("TOP_K", "4"))
EMB_MODEL = os.getenv("GEMINI_EMB_MODEL", "gemini-embedding-001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:7000")
PORT = int(os.getenv("MCP_PORT", "9000"))
DOMAINS = ("AI", "Security", "Other")
MAX_TOP_K = 50

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


def _get_milvus() -> Milvus:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMB_MODEL)

    conn_args = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        conn_args["token"] = MILVUS_TOKEN

    index_params, search_params = _build_index_params()

    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION,
        connection_args=conn_args,
        index_params=index_params,
        search_params=search_params,
        auto_id=True,
        drop_old=False,
    )

#declares a global variable
_milvus: Milvus | None = None


def get_milvus() -> Milvus:
    global _milvus
    if _milvus is None:
        _milvus = _get_milvus()
    return _milvus


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
def index_paper(pdf_base64: str, filename: str, domain: str) -> Dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}")

    pdf_bytes = base64.b64decode(pdf_base64)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if not full_text:
        raise ValueError(f"Could not extract text from {filename}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    base_doc = Document(
        page_content=full_text,
        metadata={"filename": filename, "domain": domain},
    )
    chunks: List[Document] = splitter.split_documents([base_doc])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    get_milvus().add_documents(chunks)

    return {"filename": filename, "domain": domain, "chunks_indexed": len(chunks)}


@mcp.tool
def vector_search(query: str, domain: str = "All", top_k: int = 4) -> Dict[str, Any]:
    """Semantic search over indexed paper chunks. domain: AI | Security | Other | All."""
    # domain goes into a filter expression, so only allow known values
    if domain != "All" and domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS} or 'All'")
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    expr = None if domain == "All" else f'domain == "{domain}"'

    docs_and_scores = get_milvus().similarity_search_with_score(
        query=query,
        k=top_k,
        expr=expr,
    )

    results = [
        {
            "content": doc.page_content,
            "filename": doc.metadata.get("filename", ""),
            "domain": doc.metadata.get("domain", ""),
            "score": float(score),
        }
        for doc, score in docs_and_scores
    ]
    return {"results": results}


_stats_client: MilvusClient | None = None


def _milvus_client() -> MilvusClient:
    global _stats_client
    if _stats_client is None:
        _stats_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or "")
    return _stats_client


@mcp.tool
def get_stats() -> Dict[str, int]:
    """Number of distinct indexed papers per domain."""
    client = _milvus_client()
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