"""Tiny stand-in for mcp-server with the same tool names. Used by integration tests."""
import sys

from fastmcp import FastMCP

mcp = FastMCP("fake-research-tools")
CORPUS = ["distill.pdf", "specdec.pdf", "paged.pdf", "effnet.pdf"]


@mcp.tool
def translate(text: str, source_lang: str, target_lang: str) -> dict:
    return {"translated_text": text if source_lang == target_lang else f"[{target_lang}] {text}"}


@mcp.tool
def vector_search(query: str, domain: str = "All", top_k: int = 4) -> dict:
    return {"results": [{"content": f"chunk about {f}", "filename": f, "domain": "AI", "score": 0.9, "chunk_id": i}
                        for i, f in enumerate(CORPUS[:top_k])]}


@mcp.tool
def search_paper_url(title: str) -> dict:
    return {"title": title, "url": f"https://example.org/{title}"}


@mcp.tool
def get_stats() -> dict:
    return {"AI": len(CORPUS), "Security": 0, "Other": 0}


@mcp.tool
def index_paper(path: str, filename: str, domain: str) -> dict:
    return {"filename": filename, "domain": domain, "chunks_indexed": 1}


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]), show_banner=False)
