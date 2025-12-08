from __future__ import annotations

import json
import os
import re
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger("shared.llm")
if os.getenv("LLM_DEBUG") == "1":
    logging.basicConfig(level=logging.INFO)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _default_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties") or {}
    out: Dict[str, Any] = {}
    for k, v in props.items():
        t = (v or {}).get("type")
        if t == "object":
            out[k] = {}
        elif t == "array":
            out[k] = []
        elif t == "string":
            out[k] = ""
        elif t == "boolean":
            out[k] = False
        elif t in ("integer", "number"):
            out[k] = 0
        else:
            out[k] = None

    # heuristics for common schemas in this repo
    if "route" in out and not out["route"]:
        out["route"] = "support"
    if "args" in out and not isinstance(out["args"], dict):
        out["args"] = {}
    if "needs_more_info" in out and not isinstance(out["needs_more_info"], bool):
        out["needs_more_info"] = False

    return out


def _normalize_json_output(content: str, json_schema: Optional[Dict[str, Any]]) -> Any:
    text = content.strip()

    if not json_schema:
        return text

    cleaned = _strip_code_fences(text)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_candidates = [cleaned.find("{"), cleaned.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if start_candidates:
        start = min(start_candidates)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass

    fallback = _default_from_schema(json_schema)
    fallback["_raw"] = text
    fallback["_error"] = "invalid_json_from_model"
    if os.getenv("LLM_DEBUG") == "1":
        logger.info("Failed to parse JSON from model. Returning fallback. raw=%r", text)
    return fallback


class OpenAIChatLLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)

        if os.getenv("LLM_DEBUG") == "1":
            logger.info("[LLM] Initialized OpenAI client with model=%s", self.model)

    def complete(
        self,
        system: str,
        user: str,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response_format = {"type": "json_object"} if json_schema else None

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else 0,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        content = (completion.choices[0].message.content or "").strip()
        return _normalize_json_output(content, json_schema)
