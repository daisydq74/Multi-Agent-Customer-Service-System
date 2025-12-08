import json
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.llm import get_default_llm
from shared.message_utils import build_text_message

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")


async def call_mcp(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MCP_URL}/tools/call", json={"name": tool, "arguments": arguments})
        response.raise_for_status()
        return response.json()["result"]


class CustomerDataAgent:
    def __init__(self) -> None:
        self.llm = get_default_llm()
        self.available_tools = {
            "get_customer": "Fetch a single customer by id",
            "list_customers": "List customers optionally filtered by status",
            "update_customer": "Update customer fields",
            "create_ticket": "Create a support ticket",
            "get_customer_history": "Fetch customer interaction history",
        }

    def _tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(self.available_tools.keys())},
                "args": {"type": "object"},
                "needs_more_info": {"type": "boolean"},
                "question": {"type": ["string", "null"]},
            },
            "required": ["tool", "args", "needs_more_info", "question"],
            "additionalProperties": False,
        }

    async def _decide_tool(self, user_message: str, task: Dict[str, Any], previous_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        system = (
            "You choose a single MCP tool to fulfill the request. "
            "Return ONLY JSON matching the schema. "
            "If the request cannot proceed, set needs_more_info true and include a short question."
        )
        user = json.dumps(
            {
                "user_message": user_message,
                "task": task,
                "previous_result": previous_result,
                "tools": self.available_tools,
            },
            ensure_ascii=False,
        )
        print(f"[CustomerData] Calling LLM model={self.llm.model}")
        raw = self.llm.complete(system=system, user=user, json_schema=self._tool_schema())
        return json.loads(raw)

    async def _summarize(self, tool_decision: Dict[str, Any], tool_result: Any) -> Dict[str, Any]:
        system = (
            "Summarize the tool output into structured fields. Use customer for customer records, updates for changes, history for chronological events. "
            "Include raw_tool_result to preserve the original response. Keep null when data is not present."
        )
        user = json.dumps(
            {
                "decision": tool_decision,
                "tool_result": tool_result,
            },
            ensure_ascii=False,
        )
        schema = {
            "type": "object",
            "properties": {
                "customer": {"type": ["object", "null"]},
                "updates": {"type": ["object", "null"]},
                "history": {"type": ["array", "null"]},
                "raw_tool_result": {},
            },
            "required": ["customer", "updates", "history", "raw_tool_result"],
            "additionalProperties": False,
        }
        raw = self.llm.complete(system=system, user=user, json_schema=schema)
        return json.loads(raw)

    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        user_message = task.get("user_message", "")
        last_result: Dict[str, Any] | None = None
        decision: Dict[str, Any] | None = None

        for _ in range(2):
            decision = await self._decide_tool(user_message, task, last_result)
            if decision.get("needs_more_info"):
                return {
                    "customer": None,
                    "updates": None,
                    "history": None,
                    "raw_tool_result": decision,
                }
            tool_name = decision.get("tool")
            tool_args = decision.get("args", {})
            print(f"[CustomerData] Tool decision={tool_name} args={tool_args}")
            tool_result = await call_mcp(tool_name, tool_args)
            last_result = tool_result
            break

        if decision is None:
            return {"customer": None, "updates": None, "history": None, "raw_tool_result": None}

        return await self._summarize(decision, last_result)


customer_data_agent = CustomerDataAgent()


async def data_skill(message: Message) -> Message:
    payload_text = message.parts[0].text if message.parts else "{}"
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = {"user_message": payload_text}

    result = await customer_data_agent.handle(payload)
    return build_text_message(json.dumps(result))


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Customer Data Agent",
        description="LLM-guided agent that selects MCP tools for customer data tasks.",
        url="http://localhost:8011",
        version="2.0.0",
        skills=[
            AgentSkill(
                id="customer-data",
                name="Customer Database Tools",
                description="Calls MCP tools to get and update customer data",
                tags=["mcp", "database", "llm"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["List customers", "Get history for customer"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        provider=AgentProvider(organization="Assignment 5", url="http://localhost:8011"),
        documentationUrl="https://example.com/docs/data",
        preferredTransport="JSONRPC",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Customer Data Agent")
    handler = SimpleAgentRequestHandler("data", data_skill)
    register_agent_routes(app, build_agent_card(), handler)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8011)
