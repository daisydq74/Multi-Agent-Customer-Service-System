import json

from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message


def _split_context(prompt: str) -> tuple[str, str]:
    if "Data context:" in prompt:
        lead, data = prompt.split("Data context:", 1)
        return lead.strip() or "Billing question", data.strip()
    return prompt, ""


def _context_summary(raw: str) -> str:
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            result = payload.get("result")
            if isinstance(result, dict):
                status = result.get("status")
                email = result.get("email")
                return f"Account status {status or 'unknown'}; email on file {email or 'unspecified'}."
            if isinstance(result, list) and result:
                latest = result[0]
                if isinstance(latest, dict):
                    return (
                        f"Recent ticket #{latest.get('id')} ({latest.get('status')}): {latest.get('issue')}."
                    )
    except Exception:
        return raw[:200]
    return raw[:200]


async def billing_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    user_request, data_context = _split_context(prompt)
    context_line = _context_summary(data_context)

    lines = [
        "Billing Agent: I can help with invoices, refunds, and payment issues.",
        f"Request: {user_request}",
    ]
    if context_line:
        lines.append(f"Account details: {context_line}")
    lines.append(
        "Next steps: I'll review the account, verify recent charges, and process any needed adjustments or refunds."
    )
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
