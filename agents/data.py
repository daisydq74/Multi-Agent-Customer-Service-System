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
    if "list" in prompt.lower():
        result = await call_mcp("list_customers", {"limit": 5})
        text = f"Customer Data Agent list: {result}"
    elif "history" in prompt.lower():
        result = await call_mcp("get_customer_history", {"customer_id": 1})
        text = f"History for customer 1: {result}"
    else:
        result = await call_mcp("get_customer", {"customer_id": 1})
        text = f"Fetched customer: {result}"
    return build_text_message(text)


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
