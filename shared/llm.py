import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


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
        response_format: Dict[str, Any] | None = None
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        completion = self._client.responses.create(
            model=self.model,
            input=messages,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_format=response_format,
        )
        content = completion.output_text
        if response_format:
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
