"""Chat model factory, so the agent can be pointed at a different provider from config."""
from __future__ import annotations

from typing import Any

from agent.config import Settings


def make_llm(cfg: Settings) -> Any:
    provider, sep, model = cfg.llm_model.partition(":")
    if not sep or not model:
        raise ValueError(f"LLM_MODEL must look like provider:model, got {cfg.llm_model!r}")

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=cfg.temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # OpenAI reasoning models only accept the default temperature, so it isn't passed.
        kwargs: dict[str, Any] = {}
        if cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        return ChatOpenAI(model=model, **kwargs)

    raise ValueError(f"unknown LLM provider {provider!r} (expected google or openai)")
