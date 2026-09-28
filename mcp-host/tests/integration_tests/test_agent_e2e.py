"""End-to-end: real LangGraph loop + real MCP session against a local fake tool server.

The LLM is scripted, so this runs offline and deterministically.
"""
import asyncio
import json
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.config import settings
from agent.graph import ResearchAgent
from agent.mcp_tools import MCPToolbox

pytestmark = pytest.mark.anyio
SERVER = Path(__file__).resolve().parents[1] / "fake_mcp_server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(port: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, str(SERVER), str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError("fake MCP server did not start")


@pytest.fixture()
def server():
    port = _free_port()
    state = {"port": port, "proc": _start(port)}
    yield state
    state["proc"].kill()


class ScriptedLLM(BaseChatModel):
    """Returns pre-written AIMessages in order; ignores its input."""

    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def _tool_call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


FINAL = json.dumps({
    "answer": "Distillation trains a small model on soft targets.",
    "papers": [{"title": "distill.pdf", "url": ""}, {"title": "specdec.pdf", "url": ""}],
    # only one usable recommendation (one duplicates a source) -> fallback must fill the second
    "recommended_papers": [{"title": "paged.pdf", "url": "https://example.org/paged.pdf"},
                           {"title": "Distill.pdf", "url": ""}],
})


def _script():
    return [
        _tool_call("vector_search", {"query": "distillation", "domain": "AI", "top_k": 4}, 1),
        _tool_call("search_paper_url", {"title": "paged.pdf"}, 2),
        AIMessage(content=f"```json\n{FINAL}\n```"),
    ]


def _cfg(port):
    return replace(settings, mcp_tool_url=f"http://127.0.0.1:{port}/mcp")


async def test_toolbox_direct_calls_and_concurrency(server):
    tb = MCPToolbox(f"http://127.0.0.1:{server['port']}/mcp")
    await tb.start()
    try:
        assert await tb.healthy()
        assert (await tb.call("get_stats"))["AI"] == 4
        # many concurrent calls over the one shared session
        res = await asyncio.gather(*(tb.call("search_paper_url", title=f"t{i}") for i in range(20)))
        assert [r["title"] for r in res] == [f"t{i}" for i in range(20)]
    finally:
        await tb.close()


async def test_agent_run_with_fallback(server):
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=_script()))
    try:
        out = await agent.run("What is distillation?", "English", "AI")
    finally:
        await agent.close()
    assert out["answer"].startswith("Distillation")
    assert [p["title"] for p in out["papers"]] == ["distill.pdf", "specdec.pdf"]
    recs = out["recommended_papers"]
    assert [p["title"] for p in recs] == ["paged.pdf", "effnet.pdf"]  # effnet came from fallback
    assert recs[1]["url"] == "https://example.org/effnet.pdf"
    assert set(out) == {"answer", "papers", "recommended_papers", "language", "contexts", "usage"}
    assert out["usage"]["llm_turns"] == 3
    assert out["usage"]["tool_calls"] == ["vector_search", "search_paper_url"]
    # only the chunks from the LLM's vector_search call, not the fallback's
    assert [c["filename"] for c in out["contexts"]] == ["distill.pdf", "specdec.pdf", "paged.pdf", "effnet.pdf"]


async def test_agent_recovers_after_tool_server_restart(server):
    llm = ScriptedLLM(script=_script())
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=llm)
    try:
        server["proc"].kill()
        server["proc"].wait()
        server["proc"] = _start(server["port"])
        assert not await agent.toolbox.healthy()
        llm.script.extend(_script())  # second attempt after reconnect
        out = await agent.run("What is distillation?", "English", "AI")
        assert await agent.toolbox.healthy()
        assert [p["title"] for p in out["papers"]] == ["distill.pdf", "specdec.pdf"]
    finally:
        await agent.close()


async def test_direct_tool_call_recovers_after_restart(server):
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=[]))
    try:
        assert (await agent.call_tool("get_stats"))["AI"] == 4
        server["proc"].kill()
        server["proc"].wait()
        server["proc"] = _start(server["port"])
        # the old session is dead; the call should reconnect instead of failing
        assert (await agent.call_tool("get_stats"))["AI"] == 4
        res = await agent.call_tool("index_paper", path="/data/uploads/x.pdf", filename="x.pdf", domain="AI")
        assert res["chunks_indexed"] == 1
    finally:
        await agent.close()
