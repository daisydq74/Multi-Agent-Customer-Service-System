import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


def _ensure_additional_properties_false(schema: Dict[str, Any]) -> Dict[str, Any]:
    def _normalize(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        normalized_node = dict(node)

        if "properties" in normalized_node and isinstance(normalized_node["properties"], dict):
            normalized_node["properties"] = {
                key: _normalize(value) for key, value in normalized_node["properties"].items()
            }

        if "items" in normalized_node:
            normalized_node["items"] = _normalize(normalized_node["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if key in normalized_node and isinstance(normalized_node[key], list):
                normalized_node[key] = [_normalize(option) for option in normalized_node[key]]

        if isinstance(normalized_node.get("additionalProperties"), dict):
            normalized_node["additionalProperties"] = _normalize(normalized_node["additionalProperties"])

        if (
            normalized_node.get("type") == "object"
            or "properties" in normalized_node
            or "required" in normalized_node
            or "additionalProperties" in normalized_node
        ) and "additionalProperties" not in normalized_node:
            normalized_node["additionalProperties"] = False

        return normalized_node

    return _normalize(deepcopy(schema))


@dataclass
class OpenAILLM:
    model: str
    temperature: float
    max_tokens: int

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required to run this system. "
                "Set the environment variable and try again."
            )
        self._client = OpenAI(api_key=api_key)
        print(f"[LLM] Initialized OpenAI client with model={self.model}")

    def complete(self, system: str, user: str, *, json_schema: Optional[Dict[str, Any]] = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text_format: Dict[str, Any] | None = None
        if json_schema:
            normalized_schema = _ensure_additional_properties_false(json_schema)
            text_format = {
                "format": {
                    "type": "json_schema",
                    "name": "response_schema",
                    "schema": normalized_schema,
                    "strict": True,
                }
            }
        completion = self._client.responses.create(
            model=self.model,
            input=messages,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            text=text_format,
        )
        content = completion.output[0].content[0].text
        if text_format:
            try:
                # Ensure JSON string to keep downstream parsing predictable.
                json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
        return content


def get_default_llm() -> OpenAILLM:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "600"))
    return OpenAILLM(model=model, temperature=temperature, max_tokens=max_tokens)
