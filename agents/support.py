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

# Support generally writes responses, but can request structured actions via MCP when helpful.
TOOL_CATALOG = [
    {"name": "create_ticket", "description": "Create a support ticket", "args": {"customer_id": "integer", "issue": "string", "priority": "string"}},
    {"name": "update_customer", "description": "Update customer fields", "args": {"customer_id": "integer", "data": "object"}},
    {"name": "get_customer_history", "description": "Fetch history for a customer", "args": {"customer_id": "integer"}},
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


def _collect_tool_results(data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize tool result lists from multiple agent response shapes."""

    results: List[Dict[str, Any]] = []
    if not isinstance(data_context, dict):
        return results

    candidates: List[Any] = []
    for key in ["tool_calls", "tool_calls_executed", "tool_results"]:
        if isinstance(data_context.get(key), list):
            candidates.extend(data_context.get(key, []))

    nested_context = data_context.get("data_context") if isinstance(data_context.get("data_context"), dict) else {}
    for key in ["tool_calls", "tool_calls_executed", "tool_results"]:
        if isinstance(nested_context.get(key), list):
            candidates.extend(nested_context.get(key, []))

    for item in candidates:
        if isinstance(item, dict):
            results.append(item)
    return results


def _extract_customer_details(data_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data_context, dict):
        return {}

    nested_context = data_context.get("data_context") if isinstance(data_context.get("data_context"), dict) else {}
    if isinstance(nested_context.get("customer"), dict):
        return nested_context.get("customer", {})

    for item in _collect_tool_results(data_context):
        if item.get("tool") in {"get_customer", "update_customer"}:
            result = item.get("result", {})
            parsed = result.get("result", result) if isinstance(result, dict) else result
            if isinstance(parsed, dict):
                return parsed
    return {}


def _extract_recent_history(data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(data_context, dict):
        return []

    nested_context = data_context.get("data_context") if isinstance(data_context.get("data_context"), dict) else {}
    if isinstance(nested_context.get("history"), list):
        return nested_context.get("history", [])

    for item in _collect_tool_results(data_context):
        if item.get("tool") == "get_customer_history":
            result = item.get("result", {})
            history = result.get("result", result) if isinstance(result, dict) else result
            return history if isinstance(history, list) else []
    return []


def _extract_open_ticket_report(data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull a structured report of active customers with open tickets when available."""

    if not isinstance(data_context, dict):
        return []

    nested_context = data_context.get("data_context") if isinstance(data_context.get("data_context"), dict) else {}
    report = nested_context.get("active_customers_with_open_tickets")
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]
    return []


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
    escalate_to_billing = bool(raw.get("escalate_to_billing"))
    return {
        "tool_calls": tool_calls,
        "need_clarification": need_clarification.strip(),
        "final_reply": final_reply,
        "escalate_to_billing": escalate_to_billing,
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


def _legacy_reply(request_text: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
    customer = _extract_customer_details(data_context)
    history = _extract_recent_history(data_context)
    open_ticket_report = _extract_open_ticket_report(data_context)

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
            formatted = ", ".join(
                f"#{item.get('id')}: {item.get('issue')} (status {item.get('status')})" for item in open_items[:5]
            )
            context_lines.append(f"Open tickets: {formatted}.")

    if open_ticket_report:
        context_lines.append(f"Found {len(open_ticket_report)} active customers with open tickets.")
        for entry in open_ticket_report[:5]:
            customer_obj = entry.get("customer", {}) if isinstance(entry.get("customer"), dict) else {}
            tickets = entry.get("open_tickets", []) if isinstance(entry.get("open_tickets"), list) else []
            summary = ", ".join(
                f"#{t.get('id')}: {t.get('issue')} (status {t.get('status')})" for t in tickets[:3] if isinstance(t, dict)
            )
            label = customer_obj.get("name") or f"Customer {customer_obj.get('id', 'unknown')}"
            context_lines.append(f"- {label} has {len(tickets)} open ticket(s): {summary if summary else 'details unavailable'}.")

    suggestions: List[str] = []
    if history:
        suggestions.append("I can follow up on any of the open tickets listed above.")
    suggestions.append("If anything looks off, reply here and I'll take action right away.")

    reply_lines = [intro]
    if context_lines:
        reply_lines.extend(context_lines)
    reply_lines.append(f"Regarding your request: {request_text}")
    for suggestion in suggestions[:3]:
        reply_lines.append(f"- {suggestion}")
    reply_lines.append("I'll stay on this until you're satisfied. Reply with any details you'd like me to handle now.")
    reply_text = "\n".join([line for line in reply_lines if line])

    billing_markers = ["refund", "charge", "billing", "payment", "invoice"]
    escalate = any(marker in request_text.lower() for marker in billing_markers)

    return {
        "reply": reply_text,
        "escalate_to_billing": escalate,
        "billing_issue": request_text if escalate else "",
    }


SUPPORT_SYSTEM_PROMPT = """
You are the Support Agent. Craft concise, empathetic replies using provided context.
Tools you may call via MCP (name -> args):
%s

Output STRICT JSON with keys:
- tool_calls: list of {"tool_name": string, "args": object} or {"parallel": [same]}
- final_reply: concise user-facing text grounded in tool results
- need_clarification: optional question when blocking details are missing
- escalate_to_billing: boolean when billing specialists should join
Rules:
- Prefer calling tools instead of inventing data; never fabricate results.
- Max 8 tool calls per request, max 12 items per parallel group.
- Keep responses short and actionable.
""" % json.dumps(TOOL_CATALOG, ensure_ascii=False)


async def support_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    payload, error = _parse_payload(prompt)
    data_context = payload.get("data_context", {}) if isinstance(payload, dict) else {}
    request_text = payload.get("request", "your request") if isinstance(payload, dict) else prompt
    logs: List[str] = []

    if error:
        reply_payload = _legacy_reply(request_text, data_context)
        reply_payload["handled"] = False
        reply_payload["error"] = error
        return build_text_message(json.dumps(reply_payload))

    llm_plan = await call_llm_json(
        SUPPORT_SYSTEM_PROMPT,
        {
            "request": request_text,
            "data_context": data_context,
            "hints": {"customer_id": payload.get("customer_id"), "email": payload.get("email")},
        },
    )
    validated_plan = _validate_llm_plan(llm_plan)

    if validated_plan:
        logs.append(f"LLM -> Agent: planned tool_calls={validated_plan['tool_calls']}")
        executed = await _execute_plan(validated_plan["tool_calls"], logs)
        reply_text = validated_plan.get("final_reply") or request_text
        response_payload = {
            "handled": True,
            "reply": reply_text,
            "tool_calls_executed": executed,
            "need_clarification": validated_plan.get("need_clarification", ""),
            "escalate_to_billing": validated_plan.get("escalate_to_billing", False),
            "billing_issue": request_text if validated_plan.get("escalate_to_billing", False) else "",
        }
    else:
        logs.append("LLM -> Agent: invalid output, using legacy path")
        response_payload = _legacy_reply(request_text, data_context)
        response_payload["handled"] = True

    if DEBUG_LOGS:
        response_payload["logs"] = logs
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
