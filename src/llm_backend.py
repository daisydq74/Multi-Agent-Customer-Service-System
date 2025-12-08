"""LLM backend with OpenAI integration and deterministic fallback."""
from __future__ import annotations

import logging
import os
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)


def _render_responses_output(output: List) -> str:
    """Render text content from OpenAI responses output blocks."""
    parts: List[str] = []
    for block in output or []:
        for content in getattr(block, "content", []) or []:
            if getattr(content, "type", None) == "text":
                text_val = getattr(content, "text", "")
                if isinstance(text_val, str):
                    parts.append(text_val)
    return "\n".join(parts)


def generate_text(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Generate text using OpenAI or deterministic fallback.

    If the environment lacks an API key, a heuristic response is returned to keep the
    demo runnable while preserving the same interface.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    mode = os.getenv("OPENAI_MODE", "responses")

    if not api_key:
        logger.warning("OPENAI_API_KEY not set; using deterministic fallback response")
        return f"[fallback] {user_prompt}\nInstructions: {system_prompt[:160]}"

    client = OpenAI(api_key=api_key)

    try:
        if mode == "responses":
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            if hasattr(response, "output"):
                rendered = _render_responses_output(response.output) or str(response)
                return rendered
            return str(response)

        # Default to chat completions for compatibility
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return chat.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - safety net for demo stability
        logger.error("LLM generation failed: %s", exc)
        return f"[fallback-error] {user_prompt}\nreason={exc}"
