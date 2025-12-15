import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message, MessageSendParams, Role, Task
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

DATA_AGENT_RPC = os.getenv("DATA_AGENT_RPC", "http://localhost:8011/rpc")
SUPPORT_AGENT_RPC = os.getenv("SUPPORT_AGENT_RPC", "http://localhost:8012/rpc")
BILLING_AGENT_RPC = os.getenv("BILLING_AGENT_RPC", "http://localhost:8013/rpc")
EMBEDDING_MODEL = "text-embedding-3-small"

_embedding_client = AsyncOpenAI()

TOOL_CARDS: List[Dict[str, str]] = [
    {
        "name": "get_customer",
        "text": "get_customer: Retrieve a single customer profile using customer_id. args: customer_id (int)",
    },
    {
        "name": "list_customers",
        "text": "list_customers: List customers with optional status filters. args: status (string|optional)",
    },
    {
        "name": "update_customer",
        "text": "update_customer: Update a customer's profile fields like email. args: customer_id (int), data (object)",
    },
    {
        "name": "create_ticket",
        "text": "create_ticket: Open a support ticket describing an issue. args: customer_id (int), issue (string), priority (string)",
    },
    {
        "name": "get_customer_history",
        "text": "get_customer_history: Retrieve a customer's ticket or interaction history. args: customer_id (int)",
    },
]

AGENT_CARDS: List[Dict[str, str]] = [
    {
        "name": "data agent",
        "text": "data agent: Handles database lookups and MCP tool invocations for customer records and tickets.",
    },
    {
        "name": "support agent",
        "text": "support agent: Provides product troubleshooting, onboarding, and upgrade guidance for customers.",
    },
    {
        "name": "billing agent",
        "text": "billing agent: Resolves payment, invoice, and refund questions for customer accounts.",
    },
]

_tool_embeddings: Dict[str, List[float]] = {}
_agent_embeddings: Dict[str, List[float]] = {}


def parse_request(text: str) -> Dict[str, Any]:
    customer_match = re.search(r"(?:customer\s*id|customer|id)\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    return {
        "customer_id": int(customer_match.group(1)) if customer_match else None,
        "new_email": email_match.group(0) if email_match else None,
    }


async def _embed_text(text: str) -> List[float]:
    response = await _embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_card_embeddings(cards: List[Dict[str, str]], cache: Dict[str, List[float]]) -> Dict[str, List[float]]:
    if cache:
        return cache
    for card in cards:
        cache[card["name"]] = await _embed_text(card["text"])
    return cache


async def _score_cards(query_embedding: List[float], cards: List[Dict[str, str]], cache: Dict[str, List[float]]) -> Dict[str, float]:
    embeddings = await _get_card_embeddings(cards, cache)
    return {name: _cosine_similarity(query_embedding, emb) for name, emb in embeddings.items()}


def _pick_top_choices(scores: Dict[str, float], tolerance: float = 0.02, limit: int = 2) -> List[str]:
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ordered[0]
    selections = [top_name]
    for name, score in ordered[1:]:
        if len(selections) >= limit:
            break
        if top_score - score <= tolerance:
            selections.append(name)
    return selections


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


def _parse_data_reply(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"tool": "none", "args": {}, "result": {}, "summary": text or "unable to parse data reply"}


def _summarize_result(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    summary = result.get("summary") or ""
    tool = result.get("tool") or ""
    args = result.get("args") or {}
    return f"tool={tool} args={args} summary={summary}"


async def call_data_tool(tool: str, args: Dict[str, Any], logs: List[str]) -> Dict[str, Any]:
    payload = json.dumps({"tool": tool, "args": args})
    logs.append(f"Router -> Data: tool={tool} args={args}")
    reply = await send_agent_message(DATA_AGENT_RPC, payload)
    parsed = _parse_data_reply(reply)
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
    data_results: List[Dict[str, Any]] = []
    customer_id: Optional[int] = parsed.get("customer_id")
    query_embedding = await _embed_text(user_text)

    tool_scores = await _score_cards(query_embedding, TOOL_CARDS, _tool_embeddings)
    agent_scores = await _score_cards(query_embedding, AGENT_CARDS, _agent_embeddings)

    logs.append(
        "Tool similarity: "
        + ", ".join(f"{name}={score:.3f}" for name, score in sorted(tool_scores.items(), key=lambda item: item[1], reverse=True))
    )
    logs.append(
        "Agent similarity: "
        + ", ".join(f"{name}={score:.3f}" for name, score in sorted(agent_scores.items(), key=lambda item: item[1], reverse=True))
    )

    selected_tools = _pick_top_choices(tool_scores)
    selected_agents = _pick_top_choices(agent_scores)

    if customer_id and selected_tools and selected_tools[0] != "get_customer":
        selected_tools = ["get_customer"] + selected_tools

    for tool_name in selected_tools:
        if tool_name == "get_customer":
            args = {"customer_id": customer_id}
        elif tool_name == "list_customers":
            args = {"status": None}
        elif tool_name == "update_customer":
            args = {"customer_id": customer_id, "data": {"email": parsed.get("new_email")}} if parsed.get("new_email") else {"customer_id": customer_id, "data": {}}
        elif tool_name == "create_ticket":
            args = {"customer_id": customer_id, "issue": user_text, "priority": "normal"}
        elif tool_name == "get_customer_history":
            args = {"customer_id": customer_id}
        else:
            args = {}

        result = await call_data_tool(tool_name, args, logs)
        data_results.append(result)

    support_reply = ""
    billing_reply = ""

    primary_agent = selected_agents[0] if selected_agents else "data agent"
    if primary_agent == "support agent":
        support_context = {
            "request": user_text,
            "parsed_flags": parsed,
            "data_results": data_results,
            "selected_tools": selected_tools,
        }
        support_reply = await call_support(support_context, logs)
    elif primary_agent == "billing agent":
        billing_context = {
            "request": user_text,
            "parsed_flags": parsed,
            "data_results": data_results,
            "selected_tools": selected_tools,
        }
        billing_reply = await call_billing(billing_context, logs)

    if support_reply:
        answer = support_reply
    elif billing_reply:
        answer = billing_reply
    elif data_results:
        def extract_result(entry: Dict[str, Any]) -> Any:
            payload = entry.get("result")
            if isinstance(payload, dict) and "result" in payload:
                return payload.get("result")
            return payload

        customer_entry = next((item for item in data_results if item.get("tool") == "get_customer"), data_results[-1])
        history_entry = next((item for item in data_results if item.get("tool") == "get_customer_history"), None)
        update_entry = next((item for item in data_results if item.get("tool") == "update_customer"), None)

        customer_data = extract_result(customer_entry)
        history_data = extract_result(history_entry) if history_entry else []
        update_data = extract_result(update_entry) if update_entry else None

        if parsed.get("new_email") or "get_customer_history" in selected_tools:
            lines = []
            if update_data:
                lines.append(f"Updated customer record: {update_data}")
            if history_data:
                lines.append("Ticket history:")
                for item in history_data:
                    if isinstance(item, dict):
                        lines.append(
                            f"- Ticket #{item.get('id')} ({item.get('status')}): {item.get('issue')} [priority {item.get('priority')}]"
                        )
            if not lines:
                lines.append(f"Customer data: {customer_data}")
            answer = "\n".join(lines)
        else:
            answer = f"Customer data: {customer_data}"
    else:
        answer = "I'm not sure how to help without more details."

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
