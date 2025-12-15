import json
from typing import Any, Dict

from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message


def _parse_payload(prompt: str) -> tuple[str, Dict[str, Any], str]:
    try:
        payload = json.loads(prompt)
        request = payload.get("request", "Billing question")
        data_context = payload.get("data_context", {}) if isinstance(payload, dict) else {}
        billing_issue = payload.get("billing_issue", "") if isinstance(payload, dict) else ""
        return str(request), data_context if isinstance(data_context, dict) else {}, billing_issue
    except json.JSONDecodeError:
        return prompt or "Billing question", {}, ""


def _extract_customer_info(data_context: Dict[str, Any]) -> Dict[str, Any]:
    for item in data_context.get("tool_calls", []):
        if item.get("tool") == "get_customer":
            result = item.get("result", {})
            return result.get("result", result) if isinstance(result, dict) else result
    return {}


def _strip_instruction_preamble(text: str) -> str:
    markers = [
        "Billing Agent:",
        "Do not mention internal routing",
    ]
    header, sep, rest = text.partition("\n\n")
    if sep and any(marker in header for marker in markers):
        return rest or header
    return text


async def billing_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    request, data_context, billing_issue = _parse_payload(prompt)
    customer = _extract_customer_info(data_context)

    lines = ["Billing support on it."]
    if customer:
        lines.append(
            f"Account {customer.get('id')} ({customer.get('email', 'no email on file')}) noted."
        )
    if billing_issue:
        lines.append(f"Issue details: {billing_issue}")
    lines.append(f"Request: {request}")
    lines.append("Next steps: we'll verify the transactions, apply necessary refunds, and confirm once resolved.")

    reply_text = " ".join(lines)
    return build_text_message(_strip_instruction_preamble(reply_text))


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Billing Agent",
        description="Specialist for billing questions and escalation.",
        url="http://localhost:8013",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="billing",
                name="Billing",
                description="Handle billing disputes and refunds",
                tags=["billing", "payments"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["Refund request", "Invoice copy"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        provider=AgentProvider(organization="Assignment 5", url="http://localhost:8013"),
        documentationUrl="https://example.com/docs/billing",
        preferredTransport="JSONRPC",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Billing Agent")
    handler = SimpleAgentRequestHandler("billing", billing_skill)
    register_agent_routes(app, build_agent_card(), handler)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8013)
