import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message, MessageSendParams, Role, Task
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

DATA_AGENT_RPC = os.getenv("DATA_AGENT_RPC", "http://localhost:8011/rpc")
SUPPORT_AGENT_RPC = os.getenv("SUPPORT_AGENT_RPC", "http://localhost:8012/rpc")
BILLING_AGENT_RPC = os.getenv("BILLING_AGENT_RPC", "http://localhost:8013/rpc")


def parse_request(text: str) -> Dict[str, Any]:
    customer_match = re.search(r"(?:customer\s*id|customer|id)\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    lower = text.lower()
    wants_history = "history" in lower or "tickets" in lower
    active_open_report = "active" in lower and "open ticket" in lower
    urgent_markers = ["urgent", "immediately", "asap", "charged twice", "double charge", "refund"]
    upgrade_markers = ["upgrade", "upgrading"]
    billing_markers = ["refund", "charge", "billing", "payment", "invoice", "charged twice"]

    return {
        "customer_id": int(customer_match.group(1)) if customer_match else None,
        "new_email": email_match.group(0) if email_match else None,
        "wants_ticket_history": wants_history,
        "wants_active_open_tickets_report": active_open_report,
        "is_urgent": any(marker in lower for marker in urgent_markers),
        "needs_upgrade_help": any(marker in lower for marker in upgrade_markers),
        "has_billing_issue": any(marker in lower for marker in billing_markers),
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


async def handle_active_open_report(parsed: Dict[str, Any], user_text: str, logs: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    list_result = await call_data_tool("list_customers", {"status": "active"}, logs)
    customers = list_result.get("result", {}).get("result", []) if isinstance(list_result.get("result"), dict) else list_result.get("result", [])
    open_summaries = []

    for customer in customers:
        customer_id = customer.get("id")
        if customer_id is None:
            continue
        history = await call_data_tool("get_customer_history", {"customer_id": customer_id}, logs)
        records = history.get("result", {}).get("result", []) if isinstance(history.get("result"), dict) else history.get("result", [])
        open_records = [item for item in records if isinstance(item, dict) and item.get("status") == "open"]
        if open_records:
            open_summaries.append({"customer": customer, "open_tickets": open_records})

    if not open_summaries:
        answer = "No active customers currently have open tickets."
    else:
        lines = ["Active customers with open tickets:"]
        for entry in open_summaries:
            cust = entry.get("customer", {})
            tickets = entry.get("open_tickets", [])
            lines.append(f"- Customer {cust.get('id')} ({cust.get('name')}):")
            for ticket in tickets:
                lines.append(f"  • Ticket #{ticket.get('id')} - {ticket.get('issue')} (status: {ticket.get('status')})")
        answer = "\n".join(lines)

    return answer, open_summaries


async def router_skill(message: Message) -> Message:
    user_text = message.parts[0].text if message.parts else ""
    parsed = parse_request(user_text)
    logs: List[str] = []
    data_results: List[Dict[str, Any]] = []

    if parsed["wants_active_open_tickets_report"]:
        answer, ticket_context = await handle_active_open_report(parsed, user_text, logs)
        final_text = f"{answer}\n\nA2A log:\n- " + "\n- ".join(logs)
        return build_text_message(final_text)

    customer_id: Optional[int] = parsed.get("customer_id")
    plan: List[Dict[str, Any]] = []

    if customer_id:
        plan.append({"tool": "get_customer", "args": {"customer_id": customer_id}, "parallel_key": None})

    if parsed.get("new_email") and customer_id:
        plan.append({"tool": "update_customer", "args": {"customer_id": customer_id, "data": {"email": parsed["new_email"]}}, "parallel_key": "customer_profile"})
    if parsed.get("wants_ticket_history") and customer_id:
        plan.append({"tool": "get_customer_history", "args": {"customer_id": customer_id}, "parallel_key": "customer_profile"})

    idx = 0
    while idx < len(plan):
        step = plan[idx]
        if step.get("parallel_key"):
            key = step["parallel_key"]
            group: List[Dict[str, Any]] = []
            while idx < len(plan) and plan[idx].get("parallel_key") == key:
                group.append(plan[idx])
                idx += 1
            results = await asyncio.gather(*(call_data_tool(item["tool"], item["args"], logs) for item in group))
            data_results.extend(results)
        else:
            result = await call_data_tool(step["tool"], step["args"], logs)
            data_results.append(result)
            idx += 1

    # Optional urgent ticket creation
    ticket_result: Dict[str, Any] | None = None
    if parsed.get("is_urgent") and customer_id:
        ticket_result = await call_data_tool(
            "create_ticket",
            {"customer_id": customer_id, "issue": user_text, "priority": "high"},
            logs,
        )
        data_results.append(ticket_result)

    support_reply = ""
    billing_reply = ""

    if parsed.get("needs_upgrade_help"):
        support_context = {
            "request": user_text,
            "parsed_flags": parsed,
            "data_results": data_results,
        }
        support_reply = await call_support(support_context, logs)
    elif parsed.get("has_billing_issue") or parsed.get("is_urgent"):
        billing_context = {
            "request": user_text,
            "parsed_flags": parsed,
            "data_results": data_results,
            "ticket_created": ticket_result is not None,
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

        if parsed.get("new_email") or parsed.get("wants_ticket_history"):
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
