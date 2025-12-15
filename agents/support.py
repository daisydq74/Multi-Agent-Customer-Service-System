from __future__ import annotations

from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

SUPPORT_SYSTEM_PROMPT = (
    "You are a friendly customer support representative. Use any provided account context as background, "
    "but speak directly to the customer in clear, empathetic language. Briefly summarize what you know about "
    "their situation and offer 2–3 practical next steps. Do not mention internal routing, agents, or raw JSON."
)


def _extract_context_and_request(prompt: str) -> tuple[str, str]:
    """Split incoming text into optional data context and the user's request."""

    if "Data context:" in prompt:
        lead, data = prompt.split("Data context:", 1)
        request = lead.strip() if lead.strip() else "your request"
        return data.strip(), request
    return "", prompt.strip() or "your request"


def _summarize_data_context(data: str) -> str:
    if not data:
        return ""
    try:
        import json

        payload = json.loads(data)
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, list) and result:
            latest = result[0]
            if isinstance(latest, dict):
                ticket_id = latest.get("id")
                status = latest.get("status")
                issue = latest.get("issue") or "recent activity"
                return f"Latest ticket #{ticket_id} ({status}): {issue}."
        if isinstance(result, dict):
            name = result.get("name")
            status = result.get("status")
            return f"Account {name or 'record'} is currently {status or 'noted'}."
    except Exception:
        return data[:200]
    return data[:200]


def _build_suggestions(prompt: str) -> list[str]:
    lower = prompt.lower()
    suggestions = []
    if "login" in lower or "password" in lower:
        suggestions.append("Try resetting your password and confirm you can sign in from a trusted browser.")
        suggestions.append("If the issue persists, send us the exact error message so we can investigate quickly.")
    elif "ticket" in lower or "issue" in lower:
        suggestions.append("We'll open a support ticket and keep you updated via email.")
        suggestions.append("Feel free to reply with any screenshots or timestamps to speed things up.")
    elif "history" in lower or "follow" in lower:
        suggestions.append("We reviewed your recent activity and will keep monitoring for any new updates.")
        suggestions.append("If anything changes, let us know and we can adjust the plan together.")
    else:
        suggestions.append("Let me know any specifics you want us to double-check.")
        suggestions.append("We can schedule a quick follow-up if you'd like more help.")
    suggestions.append("If you need urgent assistance, reply here and we'll prioritize your request.")
    return suggestions


async def support_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    data_context, user_request = _extract_context_and_request(prompt)

    intro = "Hi there, thanks for reaching out."
    if data_context:
        intro = "Hi there, I took a look at the recent notes on your account."

    context_line = ""
    if "login" in prompt.lower():
        context_line = "It looks like you're having trouble signing in."
    elif "ticket" in prompt.lower() or "issue" in prompt.lower():
        context_line = "I see you're dealing with an issue you'd like us to track."
    elif data_context:
        context_line = _summarize_data_context(data_context)

    suggestions = _build_suggestions(prompt)

    reply_lines = [
        SUPPORT_SYSTEM_PROMPT,
        "",
        intro,
        context_line,
        f"Here's what I'd suggest based on {user_request}:",
    ]
    for item in suggestions[:3]:
        reply_lines.append(f"- {item}")

    reply_lines.append("We're here to help—just reply to this message if you'd like me to take action now.")
    reply_text = "\n".join(line for line in reply_lines if line)
    return build_text_message(reply_text)


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
