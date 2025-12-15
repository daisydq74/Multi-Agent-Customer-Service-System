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

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
DEBUG_LOGS = os.getenv("DEBUG_A2A_LOGS") == "1"
MAX_TOOL_CALLS = 8
MAX_PARALLEL_FANOUT = 12

TOOL_CATALOG = [
    {"name": "get_customer", "description": "Fetch a single customer by id.", "args": {"customer_id": "int"}},
    {
        "name": "list_customers",
        "description": "List customers filtered by status and limit.",
        "args": {"status": "string", "limit": "integer"},
    },
    {
        "name": "update_customer",
        "description": "Update customer fields (name, email, status).",
        "args": {"customer_id": "integer", "data": "object"},
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket for a customer.",
        "args": {"customer_id": "integer", "issue": "string", "priority": "string"},
    },
    {
        "name": "get_customer_history",
        "description": "Return customer interaction history.",
        "args": {"customer_id": "integer"},
    },
]


async def call_mcp(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MCP_URL}/tools/call", json={"name": tool, "arguments": arguments})
        response.raise_for_status()
        return response.json()["result"]


def _parse_prompt(prompt: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(prompt)
        return payload if isinstance(payload, dict) else {}, None
    except json.JSONDecodeError as exc:  # noqa: PERF203
        return None, f"Invalid structured request: {exc}"


def _extract_hints(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request": payload.get("request", ""),
        "customer_id": payload.get("customer_id"),
        "email": payload.get("email"),
    }


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
    data_context = raw.get("data_context") if isinstance(raw.get("data_context"), dict) else {}
    need_clarification = raw.get("need_clarification") if isinstance(raw.get("need_clarification"), str) else ""
    final_reply = raw.get("final_reply") if isinstance(raw.get("final_reply"), str) else ""
    return {
        "tool_calls": tool_calls,
        "data_context": data_context,
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


async def _deterministic_fallback(payload: Dict[str, Any], logs: List[str]) -> Dict[str, Any]:
    request_text: str = payload.get("request", "")
    customer_id = payload.get("customer_id")
    email = payload.get("email")

    tool_calls: List[Dict[str, Any]] = []
    summaries: List[str] = []
    data_context: Dict[str, Any] = {}

    async def run_tool(name: str, arguments: Dict[str, Any]) -> Any:
        result = await _run_tool(name, arguments, logs)
        tool_calls.append({"tool": name, "args": arguments, "result": result})
        return result

    if customer_id and email:
        update_result = await run_tool("update_customer", {"customer_id": customer_id, "data": {"email": email}})
        history_result = await run_tool("get_customer_history", {"customer_id": customer_id})
        data_context = {
            "customer_id": customer_id,
            "email": email,
            "updated": update_result,
            "history": history_result,
        }
        summaries.append(f"Updated email and retrieved history for customer {customer_id}")
    elif customer_id:
        customer_result = await run_tool("get_customer", {"customer_id": customer_id})
        data_context = {"customer": customer_result}
        summaries.append(f"Fetched customer record for {customer_id}")
    else:
        customers_result = await run_tool("list_customers", {"status": "active", "limit": 50})
        customers = customers_result.get("result", []) if isinstance(customers_result, dict) else customers_result
        customers = customers if isinstance(customers, list) else []
        open_ticket_context: List[Dict[str, Any]] = []
        for customer in customers:
            cid = customer.get("id") if isinstance(customer, dict) else None
            if cid is None:
                continue
            history_result = await run_tool("get_customer_history", {"customer_id": cid})
            records = history_result.get("result", []) if isinstance(history_result, dict) else history_result
            records = records if isinstance(records, list) else []
            open_items = [r for r in records if isinstance(r, dict) and r.get("status") in {"open", "in_progress"}]
            if open_items:
                open_ticket_context.append({"customer": customer, "open_tickets": open_items})
        data_context = {"active_customers_with_open_tickets": open_ticket_context}
        summaries.append(f"Compiled report for {len(open_ticket_context)} active customers with open tickets")

    return {
        "handled": True,
        "tool_calls_executed": tool_calls,
        "summary": "; ".join(summaries) if summaries else "Tool execution completed",
        "customer_id": customer_id,
        "email": email,
        "request": request_text,
        "data_context": data_context,
    }


DATA_SYSTEM_PROMPT = """
You are the Data Agent. Decide which MCP tools to call to satisfy the user's request.
- Tools available (name -> args):
%s

Output STRICT JSON with keys:
- tool_calls: list of {"tool_name": string, "args": object} or {"parallel": [same]}
- data_context: optional object with structured notes about expected results
- final_reply: optional short summary grounded in tool outputs
- need_clarification: optional question if critical info is missing
Rules:
- Prefer calling tools; never fabricate results.
- Max 8 tool calls per request. Max 12 items inside any parallel group.
- Keep the user's request text verbatim in reasoning; do not rewrite it.
""" % json.dumps(TOOL_CATALOG, ensure_ascii=False)


async def data_skill(message: Message) -> Message:
    prompt = message.parts[0].text if message.parts else ""
    parsed_payload, error = _parse_prompt(prompt)
    logs: List[str] = []

    if error or parsed_payload is None:
        error_payload = {
            "handled": False,
            "reason": error or "Invalid structured request: expected JSON with request, customer_id, and email.",
        }
        return build_text_message(json.dumps(error_payload))

    hints = _extract_hints(parsed_payload)
    llm_plan = await call_llm_json(DATA_SYSTEM_PROMPT, {"request": hints.get("request"), "hints": hints})
    validated_plan = _validate_llm_plan(llm_plan)

    if validated_plan:
        logs.append(f"LLM -> Agent: planned tool_calls={validated_plan['tool_calls']}")
        executed = await _execute_plan(validated_plan["tool_calls"], logs)
        data_context = validated_plan.get("data_context", {}) or {}
        if executed:
            data_context = {**data_context, "tool_results": executed}
        response_payload = {
            "handled": True,
            "tool_calls_executed": executed,
            "data_context": data_context,
            "summary": validated_plan.get("final_reply", ""),
            "need_clarification": validated_plan.get("need_clarification", ""),
        }
    else:
        logs.append("LLM -> Agent: invalid or missing plan, using fallback")
        response_payload = await _deterministic_fallback(parsed_payload, logs)

    if DEBUG_LOGS:
        response_payload["logs"] = logs
    return build_text_message(json.dumps(response_payload))


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Customer Data Agent",
        description="Executes MCP database tools for customer records and tickets.",
        url="http://localhost:8011",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="customer-data",
                name="Customer Database Tools",
                description="Calls MCP tools to get and update customer data",
                tags=["mcp", "database"],
                inputModes=["text"],
                outputModes=["text"],
                examples=["List customers", "Get history for customer"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        provider=AgentProvider(organization="Assignment 5", url="http://localhost:8011"),
        documentationUrl="https://example.com/docs/data",
        preferredTransport="JSONRPC",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Customer Data Agent")
    handler = SimpleAgentRequestHandler("data", data_skill)
    register_agent_routes(app, build_agent_card(), handler)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8011)
