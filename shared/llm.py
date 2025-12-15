"""Shared OpenAI helpers for agent LLM calls.

Set ``OPENAI_API_KEY`` in the environment to enable model calls. Configure
``AGENT_LLM_MODEL`` (defaults to ``gpt-4o-mini``) to change the model used by
all specialist agents.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx

_AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")
_OPENAI_CLIENT = None


def _get_openai_client():
    """Return a shared AsyncOpenAI client when available.

    Import errors are swallowed so agents can fall back to deterministic logic
    when the OpenAI SDK is not installed in the environment.
    """

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception:
        return None

    _OPENAI_CLIENT = AsyncOpenAI(http_client=httpx.AsyncClient(timeout=30))
    return _OPENAI_CLIENT


async def call_llm_json(
    system_prompt: str, user_payload: Dict[str, Any], model: Optional[str] = None, *, max_tokens: int = 800
) -> Optional[Dict[str, Any]]:
    """Call the LLM and parse strict JSON output.

    Returns a parsed dictionary on success, otherwise ``None``. All specialists
    share the same model controlled via ``AGENT_LLM_MODEL``.
    """

    client = _get_openai_client()
    if client is None:
        return None
    try:
        response = await client.chat.completions.create(
            model=model or _AGENT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content if response.choices else None
        return json.loads(content) if content else None
    except Exception:
        return None
