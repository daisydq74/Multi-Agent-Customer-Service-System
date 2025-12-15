"""
How to test (local):
1) Start the MCP server and specialists: 
   - MCP server on 8000
   - Data agent on 8011
   - Support agent on 8012
   - Billing agent on 8013
   - Router agent on 8010
2) Run uvicorn for this router (see __main__ below).
3) Send user text via the router's JSON-RPC endpoint; routing will be decided by the LLM planner.
"""

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

import httpx
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message, MessageSendParams, Role, Task
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.message_utils import build_text_message

DATA_AGENT_RPC = os.getenv("DATA_AGENT_RPC", "http://127.0.0.1:8011/rpc")
SUPPORT_AGENT_RPC = os.getenv("SUPPORT_AGENT_RPC", "http://127.0.0.1:8012/rpc")
BILLING_AGENT_RPC = os.getenv("BILLING_AGENT_RPC", "http://127.0.0.1:8013/rpc")
ROUTER_LLM_MODEL = os.getenv("ROUTER_LLM_MODEL", "gpt-4o-mini")
MAX_PLAN_STEPS = 5


class PlanStep(TypedDict):
    agent: Literal["data", "support", "billing"]
    payload: Dict[str, Any]


class Plan(TypedDict):
    steps: List[PlanStep]
    final_answer_strategy: Literal["last_step_text", "compose"]


class RouterState(TypedDict, total=False):
    user_text: str
    parsed: Dict[str, Any]
    logs: List[str]
    plan: Plan
    step_index: int
    data_context: Dict[str, Any]
    support_payload: Dict[str, Any]
    billing_reply: Optional[str]
    final_answer: Optional[str]


def parse_request(text: str) -> Dict[str, Any]:
    customer_match = re.search(r"(?:customer\s*id|customer|id)\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    return {
        "customer_id": int(customer_match.group(1)) if customer_match else None,
        "email": email_match.group(0) if email_match else None,
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


def _parse_json_payload(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _summarize_result(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("summary"):
        return str(result.get("summary"))
    return str({k: v for k, v in result.items() if k != "tool_calls"})


async def call_data_agent(payload: Dict[str, Any], logs: List[str]) -> Dict[str, Any]:
    logs.append("Router -> Data: context sent")
    reply = await send_agent_message(DATA_AGENT_RPC, json.dumps(payload))
    parsed = _parse_json_payload(reply)
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


async def _plan_with_llm(user_text: str, parsed: Dict[str, Any]) -> Optional[Plan]:
    client = AsyncOpenAI()
    instructions = (
        "You are a planner that decides which specialist agents to call for a customer message. "
        "Agents: \n"
        "- data: fetches relevant customer data and context; expects JSON with request, customer_id, email.\n"
        "- support: crafts customer-facing responses; expects JSON with request, customer_id, email, data_context.\n"
        "- billing: handles billing replies; expects JSON with request, data_context, billing_issue.\n"
        "Return ONLY strict JSON following this schema:\n"
        "{\"steps\":[{\"agent\":\"data|support|billing\",\"payload\":{...}}],\"final_answer_strategy\":\"last_step_text|compose\"}. "
        "Steps may be in any order; omit agents that aren't needed. Avoid markdown."
    )
    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request": user_text,
                    "parsed": parsed,
                }
            ),
        },
    ]
    try:
        response = await client.chat.completions.create(
            model=ROUTER_LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            return None
        return json.loads(content)
    except Exception:
        return None


def _fallback_plan(user_text: str, parsed: Dict[str, Any]) -> Plan:
    base_payload = {
        "request": user_text,
        "customer_id": parsed.get("customer_id"),
        "email": parsed.get("email"),
    }
    return {
        "steps": [
            {"agent": "data", "payload": base_payload},
            {"agent": "support", "payload": {**base_payload, "data_context": {}}},
        ],
        "final_answer_strategy": "last_step_text",
    }


def _validate_plan(plan: Optional[Plan]) -> Optional[Plan]:
    if not isinstance(plan, dict):
        return None
    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return None
    cleaned_steps: List[PlanStep] = []
    for raw_step in steps[:MAX_PLAN_STEPS]:
        agent = raw_step.get("agent") if isinstance(raw_step, dict) else None
        payload = raw_step.get("payload") if isinstance(raw_step, dict) else None
        if agent not in {"data", "support", "billing"} or not isinstance(payload, dict):
            continue
        cleaned_steps.append({"agent": agent, "payload": payload})
    if not cleaned_steps:
        return None
    strategy = plan.get("final_answer_strategy")
    if strategy not in {"last_step_text", "compose"}:
        strategy = "last_step_text"
    return {"steps": cleaned_steps, "final_answer_strategy": strategy}


async def _plan_node(state: RouterState) -> RouterState:
    parsed = state.get("parsed", {})
    user_text = state["user_text"]
    llm_plan = await _plan_with_llm(user_text, parsed)
    validated = _validate_plan(llm_plan)
    if not validated:
        validated = _fallback_plan(user_text, parsed)
    return {"plan": validated, "step_index": 0}


async def _run_step_node(state: RouterState) -> RouterState:
    plan = state["plan"]
    idx = state.get("step_index", 0)
    logs = state.get("logs", [])
    if idx >= len(plan["steps"]):
        return {}
    step = plan["steps"][idx]
    agent = step["agent"]
    payload = step["payload"]
    if agent == "data":
        data_context = await call_data_agent(payload, logs)
        return {"data_context": data_context}
    if agent == "support":
        # include latest data_context if not already in payload
        payload = {**payload}
        if "data_context" not in payload and state.get("data_context") is not None:
            payload["data_context"] = state.get("data_context")
        elif "data_context" not in payload:
            payload["data_context"] = {}
        support_reply = await call_support(payload, logs)
        parsed_reply = _parse_json_payload(support_reply) or {"reply": support_reply}
        return {"support_payload": parsed_reply}
    if agent == "billing":
        billing_payload = {**payload}
        if "data_context" not in billing_payload and state.get("data_context") is not None:
            billing_payload["data_context"] = state.get("data_context")
        elif "data_context" not in billing_payload:
            billing_payload["data_context"] = {}
        billing_reply = await call_billing(billing_payload, logs)
        return {"billing_reply": billing_reply}
    return {}


async def _advance_node(state: RouterState) -> RouterState:
    return {"step_index": state.get("step_index", 0) + 1}


def _should_continue(state: RouterState) -> str:
    plan = state.get("plan", {"steps": []})
    idx = state.get("step_index", 0)
    if idx < len(plan.get("steps", [])):
        return "continue"
    return "done"


async def _compose_fallback(state: RouterState) -> str:
    if state.get("support_payload") and state["support_payload"].get("reply"):
        return state["support_payload"]["reply"]
    if state.get("billing_reply"):
        return state["billing_reply"] or ""
    if state.get("data_context"):
        return _summarize_result(state["data_context"])
    return "I'm sorry, I was unable to produce a response."


async def _finalize_node(state: RouterState) -> RouterState:
    plan = state.get("plan") or _fallback_plan(state.get("user_text", ""), state.get("parsed", {}))
    strategy = plan.get("final_answer_strategy", "last_step_text")
    last_agent = plan.get("steps", [{}])[-1].get("agent") if plan.get("steps") else None
    final_answer: Optional[str] = None
    if strategy == "last_step_text":
        if last_agent == "support" and state.get("support_payload"):
            final_answer = state["support_payload"].get("reply")
        elif last_agent == "billing" and state.get("billing_reply") is not None:
            final_answer = state.get("billing_reply")
        elif last_agent == "data" and state.get("data_context"):
            final_answer = _summarize_result(state["data_context"])
    if not final_answer:
        final_answer = await _compose_fallback(state)
    return {"final_answer": final_answer}


router_graph = StateGraph(RouterState)
router_graph.add_node("plan", _plan_node)
router_graph.add_node("run_step", _run_step_node)
router_graph.add_node("advance", _advance_node)
router_graph.add_node("finalize", _finalize_node)

router_graph.add_edge(START, "plan")
router_graph.add_edge("plan", "run_step")
router_graph.add_edge("run_step", "advance")
router_graph.add_conditional_edges("advance", _should_continue, {"continue": "run_step", "done": "finalize"})
router_graph.add_edge("finalize", END)

compiled_router_graph = router_graph.compile()


async def router_skill(message: Message) -> Message:
    user_text = message.parts[0].text if message.parts else ""
    parsed = parse_request(user_text)
    logs: List[str] = []
    initial_state: RouterState = {
        "user_text": user_text,
        "parsed": parsed,
        "logs": logs,
    }
    final_state = await compiled_router_graph.ainvoke(initial_state)
    answer = final_state.get("final_answer", "")
    if os.getenv("DEBUG_A2A_LOGS") == "1":
        answer = f"{answer}\n\nA2A log:\n- " + "\n- ".join(logs)
    return build_text_message(answer)


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Router Agent",
        description="Orchestrates task routing across specialist A2A agents using LLM-driven planning and LangGraph execution.",
        url="http://localhost:8010",
        version="2.0.0",
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
