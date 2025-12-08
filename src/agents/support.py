"""SupportAgent crafts customer-facing responses using LLM."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from ..llm_backend import generate_text

logger = logging.getLogger(__name__)


class SupportAgent:
    """Agent that drafts concise support responses."""

    id: str = "support-agent"
    name: str = "SupportAgent"
    version: str = "0.1.0"

    @property
    def card(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "Creates customer-facing support replies using LLM",
            "capabilities": ["message/send"],
        }

    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        context = params.get("context", {})
        user_message = params.get("message", {}).get("content", "")
        system_prompt = (
            "You are a senior customer support agent. Respond succinctly, include next steps, "
            "mention any ticket ids created, and be empathetic."
        )
        context_summary = json.dumps(context, indent=2)
        reply = generate_text(system_prompt=system_prompt, user_prompt=f"User message: {user_message}\nContext: {context_summary}")
        logger.info("SupportAgent generated reply")
        return {"message": {"role": "assistant", "content": reply}, "context": context}


support_agent = SupportAgent()
