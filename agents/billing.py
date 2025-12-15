import json
from typing import Any, Dict, List

from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message


def _parse_payload(prompt: str) -> tuple[str, Dict[str, Any], List[Dict[str, Any]], bool]:
    try:
        payload = json.loads(prompt)
        request = payload.get("request", "Billing question")
        flags = payload.get("parsed_flags", {}) if isinstance(payload, dict) else {}
        data_results = payload.get("data_results", []) if isinstance(payload, dict) else []
        ticket_created = bool(payload.get("ticket_created"))
        return str(request), flags if isinstance(flags, dict) else {}, data_results if isinstance(data_results, list) else [], ticket_created
    except json.JSONDecodeError:
        return prompt or "Billing question", {}, [], False


def _extract_customer_info(data_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in data_results:
        if item.get("tool") == "get_customer":
            result = item.get("result", {})
            return result.get("result", result) if isinstance(result, dict) else result
    return {}


async def billing_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    request, flags, data_results, ticket_created = _parse_payload(prompt)
    customer = _extract_customer_info(data_results)

    lines = ["Billing support on it."]
    if customer:
        lines.append(
            f"Account {customer.get('id')} ({customer.get('email', 'no email on file')}) noted."
        )
    if ticket_created:
        lines.append("Created a high-priority ticket so our team can process this immediately.")
    if flags.get("is_urgent"):
        lines.append("We've marked this as urgent and will review recent charges for duplicates.")
    lines.append(f"Request: {request}")
    lines.append("Next steps: we'll verify the transactions, apply necessary refunds, and confirm once resolved.")

    return build_text_message(" ".join(lines))


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
