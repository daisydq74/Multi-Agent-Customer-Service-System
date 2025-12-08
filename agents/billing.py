from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message


async def billing_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    text = (
        "Billing Agent: I can help with invoices, refunds, and payment issues. "
        f"Request: {prompt}"
    )
    return build_text_message(text)


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
