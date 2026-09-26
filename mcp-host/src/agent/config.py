"""Runtime settings for the agent service, read once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mcp_tool_url: str = os.getenv("MCP_TOOL_URL", "http://localhost:9000/mcp")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    # Hard ceiling on llm<->tools round trips so a confused model can't loop forever.
    recursion_limit: int = int(os.getenv("AGENT_RECURSION_LIMIT", "16"))
    request_timeout_s: float = float(os.getenv("AGENT_TIMEOUT_S", "120"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

# Only these tools are exposed to the LLM. index_paper / get_stats stay callable
# from the API directly but never show up in the model's tool list.
AGENT_TOOLS = ("translate", "vector_search", "search_paper_url")
DOMAINS = ("AI", "Security", "Other")
