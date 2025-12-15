import json
import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")


async def call_mcp(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MCP_URL}/tools/call", json={"name": tool, "arguments": arguments})
        response.raise_for_status()
        return response.json()["result"]


async def data_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        error_payload = {
            "tool": "none",
            "args": {},
            "result": {},
            "summary": "Invalid structured request: expected JSON with 'tool' and 'args'.",
        }
        return build_text_message(json.dumps(error_payload))

    tool = payload.get("tool")
    args: Dict[str, Any] = payload.get("args", {}) if isinstance(payload, dict) else {}

    if not tool:
        response_payload = {
            "tool": "none",
            "args": args,
            "result": {},
            "summary": "Missing tool in request.",
        }
        return build_text_message(json.dumps(response_payload))

    result: Dict[str, Any] | Any = {}
    summary = ""

    try:
        result = await call_mcp(tool, args)
        summary = f"Executed {tool}"
    except Exception as exc:  # noqa: BLE001
        summary = f"Failed to execute {tool}: {exc}"

    response_payload = {"tool": tool, "args": args, "result": result, "summary": summary}
    return build_text_message(json.dumps(response_payload))


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Customer Data Agent",
        description="Executes MCP database tools for customer records and tickets.",
        url="http://localhost:8011",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="customer-data",
                name="Customer Database Tools",
                description="Calls MCP tools to get and update customer data",
                tags=["mcp", "database"],
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
