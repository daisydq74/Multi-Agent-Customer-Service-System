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

    bracketed = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if bracketed:
        return bracketed.group(0)

    return text.strip()


def _parse_json_output(raw_text: str) -> Dict[str, Any]:
    json_text = _extract_json_block(raw_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        preview = raw_text.strip().replace("\n", " ")[:300]
        raise ValueError(f"Failed to parse JSON from model response. Preview: {preview}")


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

    def complete(self, system: str, user: str, *, json_schema: Optional[Dict[str, Any]] = None) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        response_format: Dict[str, str] | None = None
        if json_schema:
            schema_props = list(json_schema.get("properties", {}).keys())
            keys_text = ", ".join(schema_props) if schema_props else "the specified schema keys"
            schema_instruction = (
                "Respond with a single JSON object. Include only the following keys: "
                f"{keys_text}. Use null for any unknown values. Do not include any extra text."
            )
            messages.append({"role": "system", "content": schema_instruction})
            response_format = {"type": "json_object"}

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        content = completion.choices[0].message.content or ""

        if not json_schema:
            return content.strip()

        parsed = _parse_json_output(content)
        return parsed


def get_default_llm() -> OpenAILLM:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "600"))
    return OpenAILLM(model=model, temperature=temperature, max_tokens=max_tokens)
