import os
from typing import List, TypedDict

import httpx
from fastapi import FastAPI
from langgraph.graph import StateGraph, END

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message, MessageSendParams, Role, Task
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

DATA_AGENT_RPC = os.getenv("DATA_AGENT_RPC", "http://localhost:8011/rpc")
SUPPORT_AGENT_RPC = os.getenv("SUPPORT_AGENT_RPC", "http://localhost:8012/rpc")
BILLING_AGENT_RPC = os.getenv("BILLING_AGENT_RPC", "http://localhost:8013/rpc")


class RouterState(TypedDict):
    messages: List[str]
    route: str
    specialist_responses: List[str]


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


def build_graph():
    graph = StateGraph(RouterState)

    def classify(state: RouterState) -> RouterState:
        user_input = state["messages"][-1].lower()
        if "billing" in user_input or "refund" in user_input or "payment" in user_input:
            route = "billing"
        elif "history" in user_input or "customer" in user_input:
            route = "data_then_support"
        else:
            route = "support"
        state["route"] = route
        return state

    async def call_specialist(state: RouterState) -> RouterState:
        text = state["messages"][-1]
        responses: List[str] = []
        if state["route"] == "data_then_support":
            data_reply = await send_agent_message(DATA_AGENT_RPC, text)
            support_prompt = f"Data context: {data_reply}. Now craft guidance for the user."
            support_reply = await send_agent_message(SUPPORT_AGENT_RPC, support_prompt)
            responses.extend([data_reply, support_reply])
        elif state["route"] == "billing":
            billing_reply = await send_agent_message(BILLING_AGENT_RPC, text)
            responses.append(billing_reply)
        else:
            support_reply = await send_agent_message(SUPPORT_AGENT_RPC, text)
            responses.append(support_reply)
        state["specialist_responses"] = responses
        return state

    def summarize(state: RouterState) -> RouterState:
        combined = " \n".join(state.get("specialist_responses", []))
        state["messages"].append(f"Router summary: {combined}")
        return state

    graph.add_node("classify", classify)
    graph.add_node("call_specialist", call_specialist)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "call_specialist")
    graph.add_edge("call_specialist", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


workflow = build_graph()


async def router_skill(message: Message) -> Message:
    initial_state: RouterState = {"messages": [message.parts[0].text if message.parts else ""], "route": "support", "specialist_responses": []}
    final_state = await workflow.ainvoke(initial_state)
    summary_text = final_state["messages"][-1]
    return build_text_message(summary_text)


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Router Agent",
        description="Orchestrates task routing across specialist A2A agents using LangGraph.",
        url="http://localhost:8010",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="router",
                name="Router",
                description="Routes user intents to specialist agents and aggregates responses",
                tags=["router", "langgraph"],
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
