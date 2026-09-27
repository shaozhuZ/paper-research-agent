"""HTTP API in front of the research agent."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from agent.config import DOMAINS, settings
from agent.graph import ResearchAgent
from agent.logging_setup import log_fields, request_id, setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger("api")

# Shared with mcp-server (a docker volume). PDFs go through here instead of
# inside the MCP message, which has a size limit.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads"))

Language = Literal["English", "Spanish", "French", "Italian"]
QueryDomain = Literal["AI", "Security", "Other", "All"]


class InvokeRequest(BaseModel):
    query: str
    language: Language = "English"
    domain: QueryDomain = "All"


class PaperResult(BaseModel):
    title: str
    url: str = ""


class Context(BaseModel):
    filename: str
    chunk_id: int
    content: str


class InvokeResponse(BaseModel):
    answer: str
    papers: list[PaperResult]
    recommended_papers: list[PaperResult] = []
    language: str
    # the chunks the agent retrieved; used to check answers against what it actually saw
    contexts: list[Context] = []


class StatsResponse(BaseModel):
    AI: int = 0
    Security: int = 0
    Other: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the MCP session and compiled graph once; every request reuses them.
    app.state.agent = await ResearchAgent.create()
    yield
    await app.state.agent.close()


app = FastAPI(title="Research Assistant API", version="1.1", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id.set(rid)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        log_fields(
            logger,
            "request",
            method=request.method,
            path=request.url.path,
            status=status,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        request_id.reset(token)


def _agent(request: Request) -> ResearchAgent:
    return request.app.state.agent


@app.get("/ok")
async def ok(request: Request) -> dict[str, Any]:
    return {"status": "ok", "mcp": await _agent(request).toolbox.healthy()}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
    try:
        result = await _agent(request).run(req.query, req.language, req.domain)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="agent timed out")
    except Exception as e:
        logger.exception("invoke failed")
        raise HTTPException(status_code=502, detail=f"agent failed: {type(e).__name__}")
    return InvokeResponse(**result)


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), domain: str = Form(...)) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise HTTPException(status_code=400, detail=f"domain must be one of {DOMAINS}")
    data = await file.read()
    # name by content hash: the same PDF uploaded twice lands on the same file
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()}.pdf"
    if not saved.exists():
        saved.write_bytes(data)
    try:
        detail = await _agent(request).call_tool(
            "index_paper", path=str(saved), filename=file.filename, domain=domain
        )
    except Exception as e:
        logger.exception("upload failed")
        raise HTTPException(status_code=502, detail=f"index_paper failed: {e}")
    return {"status": "ok", "filename": file.filename, "domain": domain, "detail": detail}


@app.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    try:
        result = await _agent(request).call_tool("get_stats")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"get_stats failed: {e!r}")
    if not isinstance(result, dict):
        raise HTTPException(status_code=503, detail=f"unexpected get_stats result: {result!r}")
    return StatsResponse(**{k: int(result.get(k, 0) or 0) for k in DOMAINS})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
