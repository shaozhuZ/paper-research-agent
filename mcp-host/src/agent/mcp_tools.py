"""One long-lived MCP session shared by the whole process.

The original code opened a fresh MCP client, listed tools and rebuilt the
LangGraph graph on every request, and opened yet another client for every
direct tool call. Here the session is opened once at startup; LLM tool calls
and direct calls (upload, stats) all reuse it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

SERVER = "research_tools"


class ToolCallError(RuntimeError):
    pass


class ToolTimeout(RuntimeError):
    """A tool call got no reply in time, usually because the session died under it."""


class MCPToolbox:
    def __init__(self, url: str, call_timeout_s: float = 60.0) -> None:
        self.url = url
        self.call_timeout_s = call_timeout_s
        self._session: Any = None
        self._tools: dict[str, BaseTool] = {}
        self._owner: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Open the session inside a dedicated owner task.

        anyio requires a cancel scope to be exited by the task that entered it, in
        LIFO order. Keeping each session's context inside its own task means a
        reconnect can open a new session and close the old one in any order.
        """
        ready: asyncio.Future = asyncio.get_running_loop().create_future()

        async def owner() -> None:
            client = MultiServerMCPClient({SERVER: {"transport": "streamable_http", "url": self.url}})
            try:
                async with client.session(SERVER) as session:
                    self._session = session
                    tools = await load_mcp_tools(session, tool_interceptors=[self._with_timeout])
                    self._tools = {t.name: t for t in tools}
                    ready.set_result(None)
                    await self._stop.wait()
            except BaseException as e:  # includes the session dying under us
                if not ready.done():
                    ready.set_exception(e)
                elif not isinstance(e, asyncio.CancelledError):
                    logger.warning("mcp session ended: %r", e)
            finally:
                self._session = None

        self._owner = asyncio.create_task(owner(), name=f"mcp-session:{self.url}")
        await ready
        logger.info("mcp session ready", extra={"fields": {"url": self.url, "tools": sorted(self._tools)}})

    async def close(self) -> None:
        self._stop.set()
        if self._owner is not None:
            owner, self._owner = self._owner, None
            try:
                await asyncio.wait_for(owner, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                owner.cancel()

    async def _with_timeout(self, request: Any, handler: Any) -> Any:
        # Calls the LLM makes through the graph. On a dead session they never get a
        # reply and never raise, so without a limit the request just hangs until
        # the overall timeout. ToolTimeout is not a TimeoutError on purpose: the
        # caller treats it like any other session failure and reconnects.
        try:
            return await asyncio.wait_for(handler(request), timeout=self.call_timeout_s)
        except asyncio.TimeoutError:
            raise ToolTimeout(f"{request.name} got no reply in {self.call_timeout_s:g}s") from None

    async def healthy(self, timeout_s: float = 3.0) -> bool:
        if self._session is None:
            return False
        try:
            await asyncio.wait_for(self._session.send_ping(), timeout=timeout_s)
            return True
        except Exception:
            return False

    def langchain_tools(self, names: tuple[str, ...]) -> list[BaseTool]:
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise RuntimeError(f"MCP server is missing tools: {missing}")
        return [self._tools[n] for n in names]

    async def call(self, name: str, **arguments: Any) -> Any:
        """Call a tool directly (no LLM) and return its decoded result."""
        if self._session is None:
            raise RuntimeError("MCPToolbox.start() was not called")
        result = await asyncio.wait_for(
            self._session.call_tool(name, arguments), timeout=self.call_timeout_s
        )
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []))
        if result.isError:
            raise ToolCallError(f"{name} failed: {text[:500]}")
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            # FastMCP wraps non-dict return values as {"result": value}.
            if set(structured) == {"result"}:
                return structured["result"]
            return structured
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
