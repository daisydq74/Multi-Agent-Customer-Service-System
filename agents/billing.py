from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI

from langgraph_sdk.types import AgentCard, AgentCapabilities, AgentProvider, AgentSkill, Message
from shared.a2a_handler import SimpleAgentRequestHandler, register_agent_routes
from shared.llm import call_llm_json
from shared.message_utils import build_text_message

DEBUG_LOGS = os.getenv("DEBUG_A2A_LOGS") == "1"
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MAX_TOOL_CALLS = 8
MAX_PARALLEL_FANOUT = 12

TOOL_CATALOG = [
    {"name": "get_customer", "description": "Fetch a single customer", "args": {"customer_id": "integer"}},
    {"name": "get_customer_history", "description": "Fetch customer interaction history", "args": {"customer_id": "integer"}},
    {"name": "create_ticket", "description": "Create a billing-related ticket", "args": {"customer_id": "integer", "issue": "string", "priority": "string"}},
]


async def call_mcp(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MCP_URL}/tools/call", json={"name": tool, "arguments": arguments})
        response.raise_for_status()
        return response.json()["result"]


def _parse_payload(prompt: str) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        payload = json.loads(prompt)
        return payload if isinstance(payload, dict) else {}, None
    except json.JSONDecodeError as exc:  # noqa: PERF203
        return {}, f"Invalid structured request: {exc}"


def _validate_tool_call(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = entry.get("tool_name") or entry.get("tool")
    if name not in {t["name"] for t in TOOL_CATALOG}:
        return None
    args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
    return {"tool_name": name, "args": args}


def _validate_llm_plan(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    tool_calls: List[Union[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]] = []
    used_calls = 0
    for entry in raw.get("tool_calls", []):
        if used_calls >= MAX_TOOL_CALLS:
            break
        if isinstance(entry, dict) and isinstance(entry.get("parallel"), list):
            parallel_calls: List[Dict[str, Any]] = []
            for child in entry["parallel"]:
                if used_calls >= MAX_TOOL_CALLS or len(parallel_calls) >= MAX_PARALLEL_FANOUT:
                    break
                validated = _validate_tool_call(child) if isinstance(child, dict) else None
                if validated:
                    parallel_calls.append(validated)
                    used_calls += 1
            if parallel_calls:
                tool_calls.append({"parallel": parallel_calls})
            continue
        validated = _validate_tool_call(entry) if isinstance(entry, dict) else None
        if validated:
            tool_calls.append(validated)
            used_calls += 1
    need_clarification = raw.get("need_clarification") if isinstance(raw.get("need_clarification"), str) else ""
    final_reply = raw.get("final_reply") if isinstance(raw.get("final_reply"), str) else ""
    return {
        "tool_calls": tool_calls,
        "need_clarification": need_clarification.strip(),
        "final_reply": final_reply,
    }


async def _run_tool(name: str, arguments: Dict[str, Any], logs: List[str]) -> Dict[str, Any]:
    logs.append(f"Agent -> MCP: {name}({arguments})")
    try:
        result = await call_mcp(name, arguments)
        logs.append(f"MCP -> Agent: success {name}")
        return result
    except Exception as exc:  # noqa: BLE001
        logs.append(f"MCP -> Agent: failure {name}: {exc}")
        return {"error": str(exc)}


async def _execute_plan(tool_calls: List[Union[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]], logs: List[str]) -> List[Dict[str, Any]]:
    executed: List[Dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict) and isinstance(call.get("parallel"), list):
            logs.append("Agent: executing parallel tool group")
            results = await asyncio.gather(*[_run_tool(item["tool_name"], item.get("args", {}), logs) for item in call["parallel"]])
            for item, result in zip(call["parallel"], results):
                executed.append({"tool": item["tool_name"], "args": item.get("args", {}), "result": result})
            continue
        if isinstance(call, dict) and call.get("tool_name"):
            result = await _run_tool(call["tool_name"], call.get("args", {}), logs)
            executed.append({"tool": call["tool_name"], "args": call.get("args", {}), "result": result})
    return executed


def _legacy_billing_reply(request: str, data_context: Dict[str, Any], billing_issue: str) -> Dict[str, Any]:
    lines = ["Billing support on it."]
    customer = data_context.get("customer") if isinstance(data_context, dict) else {}
    if isinstance(customer, dict) and customer:
        lines.append(f"Account {customer.get('id')} ({customer.get('email', 'no email on file')}) noted.")
    if billing_issue:
        lines.append(f"Issue details: {billing_issue}")
    lines.append(f"Request: {request}")
    lines.append("Next steps: we'll verify the transactions, apply necessary refunds, and confirm once resolved.")
    return {"reply": " ".join(lines)}


BILLING_SYSTEM_PROMPT = """
You are the Billing Agent. Provide concise billing answers or trigger MCP tools when helpful.
Tools you may call via MCP (name -> args):
%s

Output STRICT JSON with keys:
- tool_calls: list of {"tool_name": string, "args": object} or {"parallel": [same]}
- final_reply: concise user-facing billing update
- need_clarification: optional question if transaction identifiers are missing
Rules:
- Prefer calling tools over guessing; never fabricate results.
- Max 8 tool calls per request, max 12 items per parallel group.
- If you lack critical identifiers, set need_clarification.
""" % json.dumps(TOOL_CATALOG, ensure_ascii=False)


async def billing_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    payload, error = _parse_payload(prompt)
    request_text = payload.get("request", "Billing question") if isinstance(payload, dict) else prompt
    data_context = payload.get("data_context", {}) if isinstance(payload, dict) else {}
    billing_issue = payload.get("billing_issue", "") if isinstance(payload, dict) else ""
    logs: List[str] = []

    if error:
        reply_payload = _legacy_billing_reply(request_text, data_context, billing_issue)
        reply_payload["handled"] = False
        reply_payload["error"] = error
        return build_text_message(json.dumps(reply_payload))

    llm_plan = await call_llm_json(
        BILLING_SYSTEM_PROMPT,
        {
            "request": request_text,
            "data_context": data_context,
            "billing_issue": billing_issue,
            "hints": {"customer_id": payload.get("customer_id"), "email": payload.get("email")},
        },
    )
    validated_plan = _validate_llm_plan(llm_plan)

    if validated_plan:
        logs.append(f"LLM -> Agent: planned tool_calls={validated_plan['tool_calls']}")
        executed = await _execute_plan(validated_plan["tool_calls"], logs)
        reply_text = validated_plan.get("final_reply") or _legacy_billing_reply(request_text, data_context, billing_issue).get("reply", request_text)
        response_payload = {
            "handled": True,
            "reply": reply_text,
            "tool_calls_executed": executed,
            "need_clarification": validated_plan.get("need_clarification", ""),
        }
    else:
        logs.append("LLM -> Agent: invalid output, using legacy path")
        response_payload = _legacy_billing_reply(request_text, data_context, billing_issue)
        response_payload["handled"] = True

    if DEBUG_LOGS:
        response_payload["logs"] = logs
    return build_text_message(json.dumps(response_payload))


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
