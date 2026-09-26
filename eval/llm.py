"""Pick a chat model from a "provider:model" string.

    get_llm("deepseek:deepseek-chat")
    get_llm("google:gemini-3-flash-preview")

The agent itself answers with Gemini. Eval scripts default to DeepSeek so the
judge is never grading answers written by its own model family.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # pick up keys from the repo-root .env so they don't have to be exported by hand
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

DEFAULT_EVAL_MODEL = os.getenv("EVAL_MODEL", "deepseek:deepseek-chat")


def get_llm(spec: str = DEFAULT_EVAL_MODEL, temperature: float = 0.0):
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"expected provider:model, got {spec!r}")

    if provider == "deepseek":
        # DeepSeek speaks the OpenAI API, so the OpenAI client works with a different base URL.
        from langchain_openai import ChatOpenAI

        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url="https://api.deepseek.com",
            temperature=temperature,
            max_retries=3,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(f"unknown provider {provider!r} (use deepseek or google)")
