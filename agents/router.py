import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message, MessageSendParams, Role, Task
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

DATA_AGENT_RPC = os.getenv("DATA_AGENT_RPC", "http://127.0.0.1:8011/rpc")
SUPPORT_AGENT_RPC = os.getenv("SUPPORT_AGENT_RPC", "http://127.0.0.1:8012/rpc")
BILLING_AGENT_RPC = os.getenv("BILLING_AGENT_RPC", "http://127.0.0.1:8013/rpc")


def parse_request(text: str) -> Dict[str, Any]:
    customer_match = re.search(r"(?:customer\s*id|customer|id)\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    return {
        "customer_id": int(customer_match.group(1)) if customer_match else None,
        "email": email_match.group(0) if email_match else None,
    }


async def send_agent_message(agent_rpc_url: str, text: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": os.urandom(8).hex(),
        "method": "message/send",
        "params": MessageSendParams(
            message=Message(messageId=os.urandom(8).hex(), role=Role.user, parts=[build_text_message(text, role=Role.user).parts[0]])
        ).model_dump(),
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(agent_rpc_url, json=payload)
        response.raise_for_status()
        result = response.json().get("result")
    if not result:
        return ""
    task = Task.model_validate(result)
    if task.history and len(task.history) > 1:
        reply = task.history[-1]
        return reply.parts[0].text if reply.parts else ""
    return ""


def _parse_json_payload(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _summarize_result(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("summary"):
        return str(result.get("summary"))
    return str({k: v for k, v in result.items() if k != "tool_calls"})


async def call_data_agent(payload: Dict[str, Any], logs: List[str]) -> Dict[str, Any]:
    logs.append("Router -> Data: context sent")
    reply = await send_agent_message(DATA_AGENT_RPC, json.dumps(payload))
    parsed = _parse_json_payload(reply)
    logs.append(f"Data -> Router: {_summarize_result(parsed)}")
    return parsed


async def call_support(context: Dict[str, Any], logs: List[str]) -> str:
    payload = json.dumps(context)
    logs.append("Router -> Support: context sent")
    reply = await send_agent_message(SUPPORT_AGENT_RPC, payload)
    logs.append("Support -> Router: response captured")
    return reply


async def call_billing(context: Dict[str, Any], logs: List[str]) -> str:
    payload = json.dumps(context)
    logs.append("Router -> Billing: context sent")
    reply = await send_agent_message(BILLING_AGENT_RPC, payload)
    logs.append("Billing -> Router: response captured")
    return reply


async def router_skill(message: Message) -> Message:
    user_text = message.parts[0].text if message.parts else ""
    parsed = parse_request(user_text)
    logs: List[str] = []
    customer_id: Optional[int] = parsed.get("customer_id")

    data_payload = {
        "request": user_text,
        "customer_id": customer_id,
        "email": parsed.get("email"),
    }
    data_context = await call_data_agent(data_payload, logs)
    data_handled = isinstance(data_context, dict) and data_context.get("handled")

    support_context = {
        "request": user_text,
        "customer_id": customer_id,
        "email": parsed.get("email"),
        "data_context": data_context if data_handled else {},
    }
    support_reply = await call_support(support_context, logs)
    support_payload = _parse_json_payload(support_reply) or {"reply": support_reply}

    if support_payload.get("escalate_to_billing"):
        billing_context = {
            "request": user_text,
            "customer_id": customer_id,
            "email": parsed.get("email"),
            "data_context": data_context,
            "billing_issue": support_payload.get("billing_issue", ""),
        }
        billing_reply = await call_billing(billing_context, logs)
        answer = billing_reply
    else:
        answer = support_payload.get("reply") or support_reply

    final_text = f"{answer}\n\nA2A log:\n- " + "\n- ".join(logs)
    return build_text_message(final_text)


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Router Agent",
        description="Orchestrates task routing across specialist A2A agents using deterministic planning.",
        url="http://localhost:8010",
        version="1.1.0",
        skills=[
            AgentSkill(
                id="router",
                name="Router",
                description="Parses user intents, builds plans, and calls specialist agents",
                tags=["router", "planning"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["Handle a billing question", "Get customer history then respond"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        provider=AgentProvider(organization="Assignment 5", url="http://localhost:8010"),
        documentationUrl="https://example.com/docs/router",
        preferredTransport="JSONRPC",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Router Agent")
    handler = SimpleAgentRequestHandler("router", router_skill)
    register_agent_routes(app, build_agent_card(), handler)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
