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
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agent.config import settings
from agent.graph import ResearchAgent
from agent.mcp_tools import MCPToolbox, ToolCallError, ToolTimeout

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
    assert recs[1]["url"] == ""  # URL lookup is switched off
    assert set(out) == {"answer", "papers", "recommended_papers", "language", "contexts", "usage"}
    assert out["usage"]["llm_turns"] == 2
    assert out["usage"]["tool_calls"] == ["vector_search"]
    # only the chunks from the LLM's vector_search call, not the fallback's
    assert [c["filename"] for c in out["contexts"]] == ["distill.pdf", "specdec.pdf", "paged.pdf", "effnet.pdf"]


async def test_fast_mode_single_call(server):
    # fake vector_search returns distill, specdec, paged, effnet in that order
    reply = AIMessage(content='```json\n{"answer": "Soft targets from a large model.", "sources": [2, 1, 9]}\n```')
    llm = ScriptedLLM(script=[reply])
    cfg = replace(_cfg(server["port"]), agent_mode="fast", fast_context_k=3, fast_search_k=4)
    agent = await ResearchAgent.create(cfg, llm=llm)
    try:
        out = await agent.run("What is distillation?", "English", "AI")
    finally:
        await agent.close()
    assert out["answer"] == "Soft targets from a large model."
    assert llm.script == []  # exactly one model call
    # sources follow the cited passages (9 is out of range and ignored)
    assert [p["title"] for p in out["papers"]] == ["specdec.pdf", "distill.pdf"]
    # recommendations come from the other search hits, never from the model
    assert [p["title"] for p in out["recommended_papers"]] == ["paged.pdf", "effnet.pdf"]
    # the model only saw the first fast_context_k passages
    assert [c["filename"] for c in out["contexts"]] == ["distill.pdf", "specdec.pdf", "paged.pdf"]
    assert out["usage"]["llm_turns"] == 1 and out["usage"]["tool_calls"] == ["vector_search"]
    assert out["usage"]["mode"] == "fast" and out["usage"]["cited"] == [2, 1]


async def test_fast_mode_plain_text_reply(server):
    # a model that ignores the JSON instruction still produces an answer
    llm = ScriptedLLM(script=[AIMessage(content="Distillation trains a small model.")])
    cfg = replace(_cfg(server["port"]), agent_mode="fast", fast_context_k=2, fast_search_k=4)
    agent = await ResearchAgent.create(cfg, llm=llm)
    try:
        out = await agent.run("What is distillation?", "English", "AI")
    finally:
        await agent.close()
    assert out["answer"] == "Distillation trains a small model."
    assert [p["title"] for p in out["papers"]] == ["distill.pdf", "specdec.pdf"]


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


async def test_read_tool_error_is_retried_once(server):
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=[]))
    agent.tool_retry_delay_s = 0
    try:
        # fails the first time, succeeds on the retry
        res = await agent.call_tool("vector_search", query="flaky:distillation", top_k=2)
        assert [h["filename"] for h in res["results"]] == ["distill.pdf", "specdec.pdf"]
        # a tool that keeps failing still surfaces the error after one retry
        with pytest.raises(ToolCallError, match="upstream unavailable"):
            await agent.call_tool("vector_search", query="broken:distillation")
    finally:
        await agent.close()


async def test_write_tool_error_is_not_retried(server):
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=[]))
    agent.tool_retry_delay_s = 0
    try:
        with pytest.raises(ToolCallError):
            await agent.call_tool("index_paper", path="/data/uploads/x.pdf", filename="flaky:x.pdf", domain="AI")
        # the next call succeeds, so the first failure was not silently retried
        res = await agent.call_tool("index_paper", path="/data/uploads/x.pdf", filename="flaky:x.pdf", domain="AI")
        assert res["chunks_indexed"] == 1
    finally:
        await agent.close()


async def test_loop_answer_survives_failed_fallback(server):
    # the model's own search works; the fallback search (which uses the raw query) fails
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=_script()))
    agent.tool_retry_delay_s = 0
    try:
        out = await agent.run("broken: what is distillation?", "English", "AI")
    finally:
        await agent.close()
    assert out["answer"].startswith("Distillation")
    # one recommendation short instead of a failed request
    assert [p["title"] for p in out["recommended_papers"]] == ["paged.pdf"]


async def test_fast_mode_survives_flaky_search(server):
    llm = ScriptedLLM(script=[AIMessage(content='{"answer": "Soft targets.", "sources": [1]}')])
    cfg = replace(_cfg(server["port"]), agent_mode="fast", fast_context_k=2, fast_search_k=4)
    agent = await ResearchAgent.create(cfg, llm=llm)
    agent.tool_retry_delay_s = 0
    try:
        out = await agent.run("flaky:what is distillation?", "English", "AI")
    finally:
        await agent.close()
    assert out["answer"] == "Soft targets."
    assert [c["filename"] for c in out["contexts"]] == ["distill.pdf", "specdec.pdf"]


async def test_loop_reconnects_while_server_is_starting(server):
    # the real case: `docker compose up -d mcp-server` and a request arrives
    # before the new server is listening
    llm = ScriptedLLM(script=_script())
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=llm)
    agent.reconnect_delays_s = (0.5, 1.0, 2.0)
    try:
        server["proc"].kill()
        server["proc"].wait()
        server["proc"] = subprocess.Popen([sys.executable, str(SERVER), str(server["port"])],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = await asyncio.wait_for(agent.run("What is distillation?", "English", "AI"), timeout=20)
        assert [p["title"] for p in out["papers"]] == ["distill.pdf", "specdec.pdf"]
    finally:
        await agent.close()


async def test_loop_fails_fast_when_server_is_down(server):
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=_script()))
    agent.reconnect_delays_s = (0.1,)
    try:
        server["proc"].kill()
        server["proc"].wait()
        t0 = time.perf_counter()
        # an error within seconds, not a hang until the request timeout
        with pytest.raises(Exception):
            await asyncio.wait_for(agent.run("What is distillation?", "English", "AI"), timeout=20)
        assert time.perf_counter() - t0 < 15
    finally:
        await agent.close()


async def test_loop_tool_call_has_a_time_limit(server):
    script = [_tool_call("vector_search", {"query": "slow:distillation", "domain": "AI", "top_k": 4}, 1)]
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=script))
    agent.toolbox.call_timeout_s = 0.5
    try:
        with pytest.raises(ToolTimeout, match="vector_search got no reply"):
            await asyncio.wait_for(agent.run("What is distillation?", "English", "AI"), timeout=4)
    finally:
        await agent.close()


class StreamingLLM(ScriptedLLM):
    """Scripted model that streams each reply a few characters at a time."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.script.pop(0).content
        for i in range(0, len(text), 4):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text[i:i + 4]))


async def test_fast_mode_streams_answer_text(server):
    llm = StreamingLLM(script=[AIMessage(content='{"answer": "Soft targets from a large model.", "sources": [2]}')])
    cfg = replace(_cfg(server["port"]), agent_mode="fast", fast_context_k=3, fast_search_k=4)
    agent = await ResearchAgent.create(cfg, llm=llm)
    try:
        events = [e async for e in agent.stream("What is distillation?", "English", "AI")]
    finally:
        await agent.close()
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert len(deltas) > 3  # arrived in pieces, not all at the end
    assert "".join(deltas) == "Soft targets from a large model."  # no JSON leaked into the text
    assert [e["type"] for e in events][-1] == "done"
    result = events[-1]["result"]
    assert result["answer"] == "Soft targets from a large model."
    assert [p["title"] for p in result["papers"]] == ["specdec.pdf"]
    assert result["usage"]["cited"] == [2] and result["usage"]["llm_turns"] == 1
    assert not any(k.startswith("_") for k in result)


async def test_loop_mode_stream_sends_only_the_final_result(server):
    agent = await ResearchAgent.create(_cfg(server["port"]), llm=ScriptedLLM(script=_script()))
    try:
        events = [e async for e in agent.stream("What is distillation?", "English", "AI")]
    finally:
        await agent.close()
    assert [e["type"] for e in events] == ["done"]
    assert events[0]["result"]["answer"].startswith("Distillation")
    assert not any(k.startswith("_") for k in events[0]["result"])
