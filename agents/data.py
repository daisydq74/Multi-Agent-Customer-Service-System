import json
import os
import re
from typing import Any, Dict, List

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
            "tool_calls": [],
            "summary": "Invalid structured request: expected JSON with request, customer_id, and email.",
        }
        return build_text_message(json.dumps(error_payload))

    request_text: str = payload.get("request", "")
    customer_id = payload.get("customer_id")
    email = payload.get("email")

    tool_calls: List[Dict[str, Any]] = []
    summaries: List[str] = []

    async def run_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await call_mcp(name, arguments)
            summaries.append(f"Executed {name}")
            tool_calls.append({"tool": name, "args": arguments, "result": result})
            return result
        except Exception as exc:  # noqa: BLE001
            summaries.append(f"Failed {name}: {exc}")
            tool_calls.append({"tool": name, "args": arguments, "result": {"error": str(exc)}})
            return {}

    if customer_id and email:
        await run_tool("update_customer", {"customer_id": customer_id, "data": {"email": email}})
        await run_tool("get_customer_history", {"customer_id": customer_id})
    elif customer_id:
        await run_tool("get_customer", {"customer_id": customer_id})
    else:
        aggregate_query = bool(re.search(r"\bactive customers\b|\breport\b", request_text, re.IGNORECASE))
        if aggregate_query:
            customers_result = await run_tool("list_customers", {"status": "active"})
            customers = customers_result.get("result", []) if isinstance(customers_result, dict) else []
            open_ticket_context: List[Dict[str, Any]] = []
            for customer in customers:
                cid = customer.get("id") if isinstance(customer, dict) else None
                if cid is None:
                    continue
                history_result = await run_tool("get_customer_history", {"customer_id": cid})
                records = history_result.get("result", []) if isinstance(history_result, dict) else []
                open_items = [r for r in records if isinstance(r, dict) and r.get("status") in {"open", "in_progress"}]
                if open_items:
                    open_ticket_context.append({"customer": customer, "open_tickets": open_items})
            if open_ticket_context:
                summaries.append(f"Found open items for {len(open_ticket_context)} active customers")

    response_payload = {
        "tool_calls": tool_calls,
        "summary": "; ".join(summaries) if summaries else "No tools executed",
        "customer_id": customer_id,
        "email": email,
        "request": request_text,
    }
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
