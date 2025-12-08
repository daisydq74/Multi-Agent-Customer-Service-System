import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


def _extract_json_block(text: str) -> str:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    generic_fenced = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if generic_fenced:
        return generic_fenced.group(1)

    return text.strip()


def _normalize_json_output(raw_text: str, json_schema: Optional[Dict[str, Any]]) -> str:
    json_text = _extract_json_block(raw_text)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        print(f"[LLM] Failed to parse JSON: {exc}; raw text: {raw_text!r}")
        raise

    if not json_schema:
        return json.dumps(parsed, separators=(",", ":"))

    properties = json_schema.get("properties") or {}
    if isinstance(properties, dict) and properties:
        parsed_obj = parsed if isinstance(parsed, dict) else {}
        normalized: Dict[str, Any] = {key: parsed_obj.get(key) for key in properties}
        for key in properties:
            if key not in normalized:
                normalized[key] = None
        return json.dumps(normalized, separators=(",", ":"))

    return json.dumps(parsed, separators=(",", ":"))


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
        if json_schema:
            schema_props = list(json_schema.get("properties", {}).keys())
            keys_text = ", ".join(schema_props) if schema_props else "the specified schema keys"
            schema_instruction = (
                "Respond with a single JSON object. Include only the following keys: "
                f"{keys_text}. Use null for any unknown values. Do not include any extra text."
            )
            messages.append({"role": "system", "content": schema_instruction})

        response_format: Dict[str, str] | None = None
        if json_schema:
            response_format = {"type": "json_object"}

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        content = completion.choices[0].message.content or ""
        return _normalize_json_output(content, json_schema)


def get_default_llm() -> OpenAILLM:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "600"))
    return OpenAILLM(model=model, temperature=temperature, max_tokens=max_tokens)
