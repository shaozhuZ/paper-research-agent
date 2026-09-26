"""Pure helpers for turning model output into the API contract.

Nothing in here touches the network, which keeps it cheap to unit test.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class Paper(BaseModel):
    title: str
    url: str = ""

    @field_validator("title", "url", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()


class FinalAnswer(BaseModel):
    """Shape the system prompt asks the model to return."""

    answer: str = ""
    papers: list[Paper] = Field(default_factory=list)
    recommended_papers: list[Paper] = Field(default_factory=list)

    @field_validator("papers", "recommended_papers", mode="before")
    @classmethod
    def _drop_bad_items(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return []
        return [p for p in v if isinstance(p, dict) and str(p.get("title") or "").strip()]


def content_to_text(content: Any) -> str:
    """Flatten a chat message's content (str, list of parts, or dict) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") if isinstance(content.get("text"), str) else ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(content)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first top-level JSON object in text, tolerating code fences and chatter."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def parse_final_answer(raw: str) -> tuple[FinalAnswer, bool]:
    """Parse model output. Returns (answer, parsed_ok); falls back to raw text as the answer."""
    obj = extract_json_object(raw)
    if obj is not None:
        try:
            return FinalAnswer.model_validate(obj), True
        except ValidationError:
            pass
    return FinalAnswer(answer=raw.strip()), False


def dedupe_papers(papers: list[Paper], exclude: set[str] | None = None, limit: int | None = None) -> list[Paper]:
    """Case-insensitive de-dup by title, skipping anything in `exclude`."""
    seen = {t.lower() for t in (exclude or set())}
    out: list[Paper] = []
    for p in papers:
        key = p.title.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if limit is not None and len(out) >= limit:
            break
    return out
