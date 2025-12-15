from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

SUPPORT_SYSTEM_PROMPT = (
    "You are a friendly customer support representative. Use the provided account context as background, "
    "but speak directly to the customer in clear, empathetic language. Briefly summarize what you know about "
    "their situation and offer 2–3 practical next steps. Do not mention internal routing, agents, or raw JSON."
)


def _extract_payload(prompt: str) -> Tuple[str, Dict[str, Any]]:
    try:
        payload = json.loads(prompt)
        request = payload.get("request", "your request")
        data_context = payload.get("data_context", {}) if isinstance(payload, dict) else {}
        return str(request), data_context if isinstance(data_context, dict) else {}
    except json.JSONDecodeError:
        return prompt or "your request", {}


def _extract_customer_details(data_context: Dict[str, Any]) -> Dict[str, Any]:
    for item in data_context.get("tool_calls", []):
        if item.get("tool") == "get_customer":
            result = item.get("result", {})
            return result.get("result", result) if isinstance(result, dict) else result
    return {}


def _extract_recent_history(data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    for item in data_context.get("tool_calls", []):
        if item.get("tool") == "get_customer_history":
            result = item.get("result", {})
            history = result.get("result", result) if isinstance(result, dict) else result
            return history if isinstance(history, list) else []
    return []


def _build_suggestions(history: List[Dict[str, Any]]) -> List[str]:
    suggestions: List[str] = []
    if history:
        suggestions.append("I can follow up on any of the open tickets listed above.")
    suggestions.append("If anything looks off, reply here and I'll take action right away.")
    return suggestions[:3]


def _build_upgrade_suggestions() -> List[str]:
    return [
        "Confirm which plan you want and whether you prefer monthly or annual billing.",
        "I’ll review pricing and when the upgrade should take effect, then enable the new features.",
        "You'll get a confirmation email once the upgrade is applied.",
    ]


def _extract_open_ticket_report(data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    report = data_context.get("data_context", {}).get("active_customers_with_open_tickets", [])
    return report if isinstance(report, list) else []


def _strip_instruction_preamble(text: str) -> str:
    markers = [
        "You are a friendly customer support representative",
        "Do not mention internal routing",
    ]
    header, sep, rest = text.partition("\n\n")
    if sep and any(marker in header for marker in markers):
        return rest or header
    return text


async def support_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    request_text, data_context = _extract_payload(prompt)
    customer = _extract_customer_details(data_context)
    history = _extract_recent_history(data_context)

    intro = "Hi there, thanks for reaching out."
    if customer:
        intro = f"Hi {customer.get('name', 'there')}, I pulled up your account details."

    context_lines: List[str] = []
    if customer:
        status = customer.get("status")
        email = customer.get("email")
        context_lines.append(f"Account status: {status or 'active'}. Email on file: {email or 'not provided'}.")
    if history:
        open_items = [h for h in history if h.get("status") == "open"]
        if open_items:
            latest = open_items[0]
            context_lines.append(f"Latest open ticket #{latest.get('id')}: {latest.get('issue')} (status {latest.get('status')}).")

    suggestions = _build_suggestions(history)

    reply_lines = [intro]
    if context_lines:
        reply_lines.extend(context_lines)
    lower_request = request_text.lower()
    upgrade_request = "upgrade" in lower_request or "upgrad" in lower_request
    open_ticket_report = _extract_open_ticket_report(data_context)

    reply_lines.append(f"Regarding your request: {request_text}")
    if open_ticket_report:
        reply_lines.append("Active customers with open tickets:")
        for entry in open_ticket_report:
            customer = entry.get("customer", {}) if isinstance(entry, dict) else {}
            tickets = entry.get("open_tickets", []) if isinstance(entry, dict) else []
            name = customer.get("name", "Unknown")
            cid = customer.get("id", "?")
            reply_lines.append(f"- {name} (ID {cid}): {len(tickets)} open or in-progress tickets")

    if upgrade_request:
        upgrade_suggestions = _build_upgrade_suggestions()
        for suggestion in upgrade_suggestions:
            reply_lines.append(f"- {suggestion}")
        reply_lines.append("Please share your account name, preferred plan, and any verification details you need me to confirm.")
    else:
        for suggestion in suggestions:
            reply_lines.append(f"- {suggestion}")
        reply_lines.append("I'll stay on this until you're satisfied. Reply with any details you'd like me to handle now.")

    reply_text = "\n".join([line for line in reply_lines if line])
    billing_markers = ["refund", "charge", "billing", "payment", "invoice"]
    escalate = any(marker in lower_request for marker in billing_markers)

    response_payload = {
        "reply": _strip_instruction_preamble(reply_text),
        "escalate_to_billing": escalate,
        "billing_issue": request_text if escalate else "",
    }
    return build_text_message(json.dumps(response_payload))


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Support Agent",
        description="Handles non-billing support cases and provides troubleshooting guidance.",
        url="http://localhost:8012",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="support-general",
                name="General Support",
                description="Answer product and troubleshooting questions",
                tags=["support", "triage"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["Help reset password", "Walk me through troubleshooting"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        provider=AgentProvider(organization="Assignment 5", url="http://localhost:8012"),
        documentationUrl="https://example.com/docs/support",
        preferredTransport="JSONRPC",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Support Agent")
    handler = SimpleAgentRequestHandler("support", support_skill)
    register_agent_routes(app, build_agent_card(), handler)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8012)
