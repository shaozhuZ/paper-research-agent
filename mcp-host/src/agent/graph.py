"""LangGraph research agent.

Graph shape is unchanged from v1 (llm <-> tools loop). What changed is the
lifecycle: the graph and MCP session are built once and reused, the LLM is
called asynchronously, and the output is validated with Pydantic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import AGENT_TOOLS, DOMAINS, RETRYABLE_TOOLS, Settings, settings as default_settings
from agent.llm import json_output, make_llm
from agent.logging_setup import log_fields
from agent.mcp_tools import MCPToolbox, ToolCallError
from agent.parsing import (
    FinalAnswer,
    Paper,
    content_to_text,
    dedupe_papers,
    extract_json_object,
    parse_final_answer,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
#Role: You are a multilingual academic research assistant.

#Tools available:
- translate(text, source_lang, target_lang): Translate text between English, Spanish, French, Italian.
- vector_search(query, domain, top_k): Search the academic paper database using cosine similarity.

#Workflow for answering a research query:
1. If the query is not in English, call translate() to convert it to English first.
2. Call vector_search() with the English query and the specified domain filter.
3. Using ONLY the retrieved context, generate a thorough answer in English.
4. If the target language is not English, call translate() to convert the answer to the target language.
5. Return the final answer, 2 reference papers, and 2 recommended papers.

#Output format:
Always respond with a JSON object in this exact format:
{
  "answer": "<answer in the target language>",
  "papers": [
    {"title": "<filename of source paper>", "url": ""},
    {"title": "<filename of source paper>", "url": ""}
  ],
  "recommended_papers": [
    {"title": "<filename of recommended paper>", "url": ""},
    {"title": "<filename of recommended paper>", "url": ""}
  ]
}

#Rules:
- Never fabricate information. Only use what is in the retrieved context.
- If no relevant context is found, say so clearly in the target language.
- Always respond with valid JSON. No extra text outside the JSON.
- Always include exactly 2 source papers in "papers".
- Always include exactly 2 additional papers in "recommended_papers", and do not duplicate titles from "papers".
- Once you have generated the final JSON output, stop immediately. Do not make any additional tool calls after returning the JSON.
"""


FAST_PROMPT = """You answer questions about academic papers using only the numbered passages you are given.

Rules:
- Every statement in the answer must be directly supported by a passage. Do not add background knowledge, or purposes, consequences or importance that the passages do not state.
- If the passages don't contain the answer, say so plainly instead of guessing.
- Give a complete answer: include every detail in the passages that bears on the question (numbers, conditions, reasons, contrasts), not just the first fact that fits. Being complete never overrides the first rule.
- Don't put passage numbers like [1] in the answer text; list them in "sources" instead.
- Write the answer in {language}.

Reply with JSON only, no other text:
{{"answer": "<the answer>", "sources": [<numbers of the passages you used>]}}
"""


# strict mode needs every property listed in "required" and no extra keys
FAST_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "sources"],
    "additionalProperties": False,
}


def _format_passages(hits: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{i}] ({h.get('filename', '')})\n{h.get('content', '')}" for i, h in enumerate(hits, 1))


def _distinct_papers(filenames: list[str], exclude: set[str] = frozenset(), limit: int = 2) -> list[Paper]:
    out: list[Paper] = []
    for f in filenames:
        if f and f not in exclude and f not in {p.title for p in out}:
            out.append(Paper(title=f))
        if len(out) == limit:
            break
    return out


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(llm: Any, tools: list) -> Any:
    llm_with_tools = llm.bind_tools(tools)
    system = SystemMessage(content=SYSTEM_PROMPT)

    async def llm_node(state: AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if not msgs or msgs[0].type != "system":
            msgs = [system, *msgs]
        t0 = time.perf_counter()
        resp = await llm_with_tools.ainvoke(msgs)
        # time per LLM call, so a run can be split into model time vs tool time
        resp.response_metadata["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        return {"messages": [resp]}

    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile()


def _retrieved_contexts(messages: list) -> list[dict[str, Any]]:
    """Chunks returned by every vector_search call in this run, in order, de-duplicated.

    These are what the model actually saw, so they are what an answer has to be
    faithful to.
    """
    seen, out = set(), []
    for m in messages:
        if not isinstance(m, ToolMessage) or m.name != "vector_search":
            continue
        data = extract_json_object(content_to_text(m.content))
        for hit in (data or {}).get("results", []):
            key = (hit.get("filename"), hit.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append({"filename": hit.get("filename", ""), "chunk_id": hit.get("chunk_id", -1),
                        "content": hit.get("content", "")})
    return out


def _trace_stats(messages: list) -> dict[str, Any]:
    ai = [m for m in messages if isinstance(m, AIMessage)]
    calls = [c["name"] for m in ai for c in (m.tool_calls or [])]
    usage = [m.usage_metadata or {} for m in ai]
    return {
        "llm_turns": len(ai),
        "tool_calls": calls,
        "tool_errors": sum(1 for m in messages if isinstance(m, ToolMessage) and m.status == "error"),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        # part of input_tokens served from the provider's prompt cache (billed lower)
        "cached_tokens": sum((u.get("input_token_details") or {}).get("cache_read", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        # thinking tokens; not every provider reports them separately
        "reasoning_tokens": sum((u.get("output_token_details") or {}).get("reasoning", 0) for u in usage),
        "llm_ms": sum(m.response_metadata.get("latency_ms", 0) for m in ai),
    }


class ResearchAgent:
    """Owns the MCP session and the compiled graph for the life of the process."""

    # pause before retrying a read-only tool that reported an error
    tool_retry_delay_s: float = 1.0

    def __init__(self, cfg: Settings, llm: Any, toolbox: MCPToolbox) -> None:
        self.cfg = cfg
        self.llm = llm
        self.fast_llm = json_output(llm, cfg, "fast_answer", FAST_SCHEMA)
        self.toolbox = toolbox
        self.graph = build_graph(llm, toolbox.langchain_tools(AGENT_TOOLS))
        self._reconnect_lock = asyncio.Lock()

    @classmethod
    async def create(cls, cfg: Settings = default_settings, llm: Any = None) -> "ResearchAgent":
        if llm is None:
            llm = make_llm(cfg)
        toolbox = MCPToolbox(cfg.mcp_tool_url)
        await toolbox.start()
        return cls(cfg, llm, toolbox)

    async def close(self) -> None:
        await self.toolbox.close()

    async def reconnect(self) -> None:
        """Re-open the MCP session (e.g. after the tool server restarted) and rebind the graph."""
        async with self._reconnect_lock:
            if await self.toolbox.healthy():
                return  # another request already reconnected
            old = self.toolbox
            fresh = MCPToolbox(self.cfg.mcp_tool_url)
            await fresh.start()
            self.toolbox = fresh
            self.graph = build_graph(self.llm, fresh.langchain_tools(AGENT_TOOLS))
            await old.close()

    async def call_tool(self, name: str, **arguments: Any) -> Any:
        """Direct tool call (no LLM) that survives a dropped MCP session."""
        if not await self.toolbox.healthy():
            logger.warning("MCP session is down before %s, reconnecting", name)
            await self.reconnect()
        try:
            return await self.toolbox.call(name, **arguments)
        except ToolCallError as e:
            # the tool ran and reported an error, so reconnecting won't help. For reads,
            # one retry covers the usual cause (a transient upstream 503/429).
            if name not in RETRYABLE_TOOLS:
                raise
            logger.warning("%s reported an error, retrying once: %s", name, e)
            await asyncio.sleep(self.tool_retry_delay_s)
            return await self.toolbox.call(name, **arguments)
        except Exception:
            if await self.toolbox.healthy():
                raise
            logger.warning("MCP session dropped during %s, reconnecting and retrying", name)
            await self.reconnect()
            return await self.toolbox.call(name, **arguments)

    async def run(self, query: str, language: str, domain: str) -> dict[str, Any]:
        if self.cfg.agent_mode == "fast":
            # tool calls go through call_tool, which already reconnects a dropped session
            return self._strip(await asyncio.wait_for(
                self._run_fast(query, language, domain), timeout=self.cfg.request_timeout_s
            ))
        try:
            result = await self._run_once(query, language, domain)
        except asyncio.TimeoutError:
            raise
        except Exception:
            if await self.toolbox.healthy():
                raise  # not an MCP problem (e.g. LLM quota); let the API report it
            logger.warning("MCP session is down, reconnecting and retrying", exc_info=True)
            await self.reconnect()
            return self._strip(await self._run_once(query, language, domain))

        # ToolNode turns tool exceptions into error messages instead of raising, so a
        # dead session shows up here as tool errors rather than as an exception.
        if result["_tool_errors"] and not await self.toolbox.healthy():
            logger.warning("tool errors with a dead MCP session, reconnecting and retrying")
            await self.reconnect()
            result = await self._run_once(query, language, domain)
        return self._strip(result)

    async def _run_fast(self, query: str, language: str, domain: str) -> dict[str, Any]:
        """Search once, answer with one model call. No tool loop, so no step limit to hit."""
        t0 = time.perf_counter()
        res = await self.call_tool(
            "vector_search", query=query, domain=domain if domain in DOMAINS else "All",
            top_k=max(self.cfg.fast_search_k, self.cfg.fast_context_k),
        )
        hits = [h for h in (res.get("results", []) if isinstance(res, dict) else []) if isinstance(h, dict)]
        passages = hits[: self.cfg.fast_context_k]

        messages = [
            SystemMessage(content=FAST_PROMPT.format(language=language)),
            HumanMessage(content=f"Question: {query}\n\nPassages:\n\n{_format_passages(passages)}"),
        ]
        t_llm = time.perf_counter()
        reply = await self.fast_llm.ainvoke(messages)
        reply.response_metadata["latency_ms"] = round((time.perf_counter() - t_llm) * 1000)

        raw = content_to_text(reply.content)
        data = extract_json_object(raw) or {}
        answer = str(data.get("answer") or "").strip() or raw.strip()
        cited = [n for n in data.get("sources", []) if isinstance(n, int) and 1 <= n <= len(passages)]

        # sources = papers of the passages the model cited; recommendations = other papers
        # the search turned up. Both come from the index, so nothing is made up.
        source_files = [passages[n - 1]["filename"] for n in cited] or [h["filename"] for h in passages]
        papers = _distinct_papers(source_files)
        recommended = _distinct_papers([h["filename"] for h in hits], exclude={p.title for p in papers})
        if len(recommended) < 2:
            recommended += await self.fallback_recommendations(
                query, domain, exclude={p.title for p in papers + recommended}, needed=2 - len(recommended)
            )

        stats = _trace_stats([reply])
        stats["tool_calls"] = ["vector_search"]
        latency_ms = round((time.perf_counter() - t0) * 1000)
        log_fields(logger, "agent run", mode="fast", latency_ms=latency_ms, parsed_ok=bool(data),
                   cited=cited, **stats)
        return {
            "answer": answer,
            "papers": [p.model_dump() for p in papers],
            "recommended_papers": [p.model_dump() for p in recommended[:2]],
            "language": language,
            "contexts": [{"filename": h.get("filename", ""), "chunk_id": h.get("chunk_id", -1),
                          "content": h.get("content", "")} for h in passages],
            "usage": {**stats, "latency_ms": latency_ms, "model": self.cfg.llm_model, "mode": "fast",
                      "cited": cited},
        }

    @staticmethod
    def _strip(result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if not k.startswith("_")}

    async def _run_once(self, query: str, language: str, domain: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        user_message = (
            f"Query: {query}\n"
            f"Target language for the answer: {language}\n"
            f"Domain filter: {domain}"
        )
        out = await asyncio.wait_for(
            self.graph.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={"recursion_limit": self.cfg.recursion_limit},
            ),
            timeout=self.cfg.request_timeout_s,
        )
        raw = content_to_text(out["messages"][-1].content)
        final, parsed_ok = parse_final_answer(raw)
        result = await self._assemble(final, query, domain, language)

        stats = _trace_stats(out["messages"])
        result["_tool_errors"] = stats["tool_errors"]
        result["contexts"] = _retrieved_contexts(out["messages"])
        latency_ms = round((time.perf_counter() - t0) * 1000)
        result["usage"] = {**stats, "latency_ms": latency_ms, "model": self.cfg.llm_model, "mode": "loop"}
        log_fields(
            logger,
            "agent run",
            latency_ms=latency_ms,
            parsed_ok=parsed_ok,
            fallback_used=result["_fallback_used"],
            **stats,
        )
        return result

    async def _assemble(self, final: FinalAnswer, query: str, domain: str, language: str) -> dict[str, Any]:
        papers = dedupe_papers(final.papers, limit=2)
        recommended = dedupe_papers(final.recommended_papers, exclude={p.title for p in papers}, limit=2)
        fallback_used = len(recommended) < 2
        if fallback_used:
            recommended += await self.fallback_recommendations(
                query, domain, exclude={p.title for p in papers + recommended}, needed=2 - len(recommended)
            )
        return {
            "answer": final.answer,
            "papers": [p.model_dump() for p in papers],
            "recommended_papers": [p.model_dump() for p in recommended[:2]],
            "language": language,
            "_fallback_used": fallback_used,
        }

    async def fallback_recommendations(self, query: str, domain: str, exclude: set[str], needed: int) -> list[Paper]:
        """Fill missing recommendations straight from the vector DB, bypassing the LLM."""
        if needed <= 0:
            return []
        try:
            res = await self.call_tool(
                "vector_search",
                query=query,
                domain=domain if domain in DOMAINS else "All",
                top_k=max(6, needed + len(exclude)),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # recommendations are extra; an answer with fewer of them beats a 502
            logger.warning("fallback recommendations unavailable: %r", e)
            return []
        hits = res.get("results", []) if isinstance(res, dict) else []
        # no URL lookup here: see AGENT_TOOLS in config.py
        return dedupe_papers(
            [Paper(title=h.get("filename", "")) for h in hits if isinstance(h, dict) and h.get("filename")],
            exclude=exclude,
            limit=needed,
        )


if __name__ == "__main__":
    import json
    import sys

    async def _main() -> None:
        agent = await ResearchAgent.create()
        try:
            q = sys.argv[1] if len(sys.argv) > 1 else "What is knowledge distillation?"
            print(json.dumps(await agent.run(q, "English", "All"), indent=2, ensure_ascii=False))
        finally:
            await agent.close()

    asyncio.run(_main())
