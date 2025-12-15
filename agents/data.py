import json
import os
import re
from typing import Any, Dict, Optional

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


def parse_customer_id(text: str) -> Optional[int]:
    match = re.search(r"(?:customer\\s*id|id)\s*[:#]?\s*(\\d+)|customer\\s+(\\d+)", text, re.IGNORECASE)
    if not match:
        return None
    matched_id = next((group for group in match.groups() if group), None)
    return int(matched_id) if matched_id else None


def parse_email(text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def parse_status(text: str) -> Optional[str]:
    lower = text.lower()
    if "disabled" in lower or "inactive" in lower:
        return "disabled"
    if "active" in lower:
        return "active"
    return None


def parse_limit(text: str) -> int:
    match = re.search(r"limit\s+(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 50


def parse_priority(text: str) -> str:
    urgent_markers = ["immediately", "charged twice", "refund", "urgent", "asap"]
    lower = text.lower()
    return "high" if any(marker in lower for marker in urgent_markers) else "medium"


async def data_skill(message: Message) -> Message:
    raw_text = message.parts[0].text if message.parts else ""
    prompt = raw_text
    lower_prompt = raw_text.lower()
    provided_customer_id: Optional[int] = None
    provided_email: Optional[str] = None

    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            prompt = str(payload.get("user_text", raw_text))
            lower_prompt = prompt.lower()
            provided_customer_id = payload.get("customer_id") if isinstance(payload.get("customer_id"), int) else None
            provided_email = payload.get("new_email") if isinstance(payload.get("new_email"), str) else None
    except Exception:
        prompt = raw_text

    customer_id = provided_customer_id if provided_customer_id is not None else parse_customer_id(prompt)
    status = parse_status(prompt)
    limit = parse_limit(prompt)
    email = provided_email if provided_email is not None else parse_email(prompt)
    priority = parse_priority(prompt)

    tool = ""
    args: Dict[str, Any] = {}
    summary = ""
    result: Dict[str, Any] | Any = {}

    if "history" in lower_prompt:
        if customer_id is None:
            summary = "No customer id provided for history lookup."
        else:
            tool = "get_customer_history"
            args = {"customer_id": customer_id}
            try:
                result = await call_mcp(tool, args)
                summary = f"History fetched for customer {customer_id}"
            except httpx.HTTPStatusError as exc:
                summary = f"History lookup failed ({exc.response.status_code})."
    elif "list" in lower_prompt:
        tool = "list_customers"
        args = {"status": status, "limit": limit}
        try:
            result = await call_mcp(tool, args)
            summary = f"Listed {len(result)} customers"
        except httpx.HTTPStatusError as exc:
            summary = f"Customer listing failed ({exc.response.status_code})."
    elif "update" in lower_prompt or "change" in lower_prompt:
        if customer_id is None:
            summary = "No customer id provided for update."
        else:
            update_fields = {k: v for k, v in {"email": email, "status": status}.items() if v is not None}
            if "name" in lower_prompt:
                update_fields["name"] = prompt
            if not update_fields:
                summary = "No valid fields provided for update."
            else:
                tool = "update_customer"
                args = {"customer_id": customer_id, "data": update_fields}
                try:
                    result = await call_mcp(tool, args)
                    summary = f"Updated customer {customer_id}"
                except httpx.HTTPStatusError as exc:
                    summary = f"Update failed ({exc.response.status_code}) for customer {customer_id}."
    elif "ticket" in lower_prompt or "issue" in lower_prompt:
        if customer_id is None:
            summary = "No customer id provided for ticket creation."
        else:
            tool = "create_ticket"
            args = {"customer_id": customer_id, "issue": prompt, "priority": priority}
            try:
                result = await call_mcp(tool, args)
                summary = f"Created ticket for customer {customer_id}"
            except httpx.HTTPStatusError as exc:
                summary = f"Ticket creation failed ({exc.response.status_code}) for customer {customer_id}."
    else:
        if customer_id is None:
            summary = "No customer id provided for lookup."
        else:
            tool = "get_customer"
            args = {"customer_id": customer_id}
            try:
                result = await call_mcp(tool, args)
                summary = f"Fetched customer {customer_id}"
            except httpx.HTTPStatusError as exc:
                summary = f"Lookup failed ({exc.response.status_code}) for customer {customer_id}."

    response_payload = {"tool": tool or "none", "args": args, "result": result, "summary": summary}
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
