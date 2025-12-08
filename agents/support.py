import json
from typing import Any, Dict

from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.llm import get_default_llm
from shared.message_utils import build_text_message


class SupportAgent:
    def __init__(self) -> None:
        self.llm = get_default_llm()

    async def respond(self, user_message: str, context: Dict[str, Any]) -> str:
        system = (
            "You are a helpful customer support agent. Ground every response in the provided context and tool results. "
            "Be concise, empathetic, and avoid mentioning internal tooling. "
            "If you cannot answer, ask one clarifying question."
        )
        user = json.dumps(
            {
                "user_message": user_message,
                "context": context,
            },
            ensure_ascii=False,
        )
        print(f"[Support] Calling LLM model={self.llm.model}")
        return self.llm.complete(system=system, user=user)


support_agent = SupportAgent()


async def support_skill(message: Message) -> Message:
    payload_text = message.parts[0].text if message.parts else ""
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = {"user_message": payload_text, "data_result": None}

    user_message = payload.get("user_message", "")
    response = await support_agent.respond(user_message, payload)
    return build_text_message(response)


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Support Agent",
        description="LLM-driven support agent that crafts final customer responses.",
        url="http://localhost:8012",
        version="2.0.0",
        skills=[
            AgentSkill(
                id="support-general",
                name="General Support",
                description="Answer product and troubleshooting questions",
                tags=["support", "llm"],
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
