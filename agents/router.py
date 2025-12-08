from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message, MessageSendParams, Role, Task
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.llm import get_default_llm
from shared.message_utils import build_text_message

DATA_AGENT_RPC = os.getenv("DATA_AGENT_RPC", "http://localhost:8011/rpc")
SUPPORT_AGENT_RPC = os.getenv("SUPPORT_AGENT_RPC", "http://localhost:8012/rpc")


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


class RouterAgent:
    def __init__(self) -> None:
        self.llm = get_default_llm()

    async def route(self, user_message: str, conversation_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        system = (
            "You are a routing orchestrator. Decide where to send the request strictly following the JSON schema. "
            "Routes: 'support' (general guidance only), 'customer_data' (call data tools only), or 'both' (data then support). "
            "Select data_task based on intent: lookup for fetching customer info, update for modifying records/tickets, history for interaction logs. "
            "Infer customer_id only if explicitly provided. Keep notes concise."
        )
        user = json.dumps(
            {
                "message": user_message,
                "conversation_state": conversation_state or {},
            },
            ensure_ascii=False,
        )
        schema = {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ["support", "customer_data", "both"]},
                "customer_id": {"type": ["integer", "null"]},
                "data_task": {"type": ["string", "null"], "enum": ["lookup", "update", "history", None]},
                "notes": {"type": "string"},
            },
            "required": ["route", "customer_id", "data_task", "notes"],
            "additionalProperties": False,
        }
        print(f"[Router] Calling LLM model={self.llm.model}")
        raw = self.llm.complete(system=system, user=user, json_schema=schema)
        parsed = json.loads(raw)
        return parsed


router_agent = RouterAgent()


async def router_skill(message: Message) -> Message:
    user_text = message.parts[0].text if message.parts else ""
    route_result = await router_agent.route(user_text)

    print(f"[Router] Route={route_result.get('route')} data_task={route_result.get('data_task')} customer_id={route_result.get('customer_id')}")

    data_context = ""
    if route_result.get("route") in {"customer_data", "both"}:
        data_payload = json.dumps({
            "user_message": user_text,
            "route_decision": route_result,
        })
        data_context = await send_agent_message(DATA_AGENT_RPC, data_payload)

    support_response = ""
    if route_result.get("route") in {"support", "both"}:
        support_context = {
            "user_message": user_text,
            "route_decision": route_result,
        }
        if data_context:
            try:
                support_context["data_result"] = json.loads(data_context)
            except json.JSONDecodeError:
                support_context["data_result"] = {"raw": data_context}
        support_payload = json.dumps(support_context, ensure_ascii=False)
        support_response = await send_agent_message(SUPPORT_AGENT_RPC, support_payload)

    final_response = support_response or data_context or ""
    return build_text_message(final_response)


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Router Agent",
        description="LLM-driven router that coordinates support and customer data agents.",
        url="http://localhost:8010",
        version="2.0.0",
        skills=[
            AgentSkill(
                id="router",
                name="Router",
                description="Routes user intents to specialist agents using an LLM decision.",
                tags=["router", "llm"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["Handle a support case", "Lookup customer history then respond"],
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
