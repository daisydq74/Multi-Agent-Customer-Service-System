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

    def _with_temperature_retry(callable_fn, kwargs):
        try:
            return callable_fn(**kwargs)
        except Exception as exc:
            message = str(exc)
            is_temp_error = (
                getattr(exc, "status_code", None) == 400
                and "Unsupported parameter: 'temperature' is not supported" in message
            )
            if is_temp_error and "temperature" in kwargs:
                logger.warning(
                    "OpenAI backend does not support temperature; retrying without it"
                )
                kwargs_no_temp = {k: v for k, v in kwargs.items() if k != "temperature"}
                return callable_fn(**kwargs_no_temp)
            logger.error("LLM generation failed: %s", exc)
            raise

    if mode == "responses":
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        response = _with_temperature_retry(client.responses.create, kwargs)
        if hasattr(response, "output"):
            rendered = _render_responses_output(response.output) or str(response)
            return rendered
        return str(response)

    # Default to chat completions for compatibility
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    chat = _with_temperature_retry(client.chat.completions.create, kwargs)
    return chat.choices[0].message.content or ""
