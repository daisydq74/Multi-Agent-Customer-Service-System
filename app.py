import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mcp_server.db import (
    create_ticket_record,
    fetch_customer,
    fetch_customers,
    fetch_history,
    update_customer_record,
)

app = FastAPI(title="Assignment 5 MCP Server", version="1.0.0")
event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()


class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]


@app.post("/tools/list")
async def list_tools() -> Dict[str, Any]:
    tools = [
        {
            "name": "get_customer",
            "description": "Fetch a single customer by id.",
            "input_schema": {
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
            },
        },
        {
            "name": "list_customers",
            "description": "List customers optionally filtered by status.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        },
        {
            "name": "update_customer",
            "description": "Update customer fields (name, email, status).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "data": {"type": "object"},
                },
                "required": ["customer_id", "data"],
            },
        },
        {
            "name": "create_ticket",
            "description": "Create a support ticket for a customer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "issue": {"type": "string"},
                    "priority": {"type": "string"},
                },
                "required": ["customer_id", "issue", "priority"],
            },
        },
        {
            "name": "get_customer_history",
            "description": "Return customer interaction history.",
            "input_schema": {
                "type": "object",
                "properties": {"customer_id": {"type": "integer"}},
                "required": ["customer_id"],
            },
        },
    ]
    return {"tools": tools}


async def _enqueue_event(event: Dict[str, Any]) -> None:
    await event_queue.put(event)


@app.post("/tools/call")
async def call_tool(payload: ToolCallRequest) -> Dict[str, Any]:
    name = payload.name
    args = payload.arguments

    if name == "get_customer":
        customer = await asyncio.to_thread(fetch_customer, int(args.get("customer_id")))
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        await _enqueue_event({"type": "audit", "tool": name, "customer_id": customer["id"]})
        return {"result": customer}

    if name == "list_customers":
        status = args.get("status")
        limit = int(args.get("limit", 20))
        customers = await asyncio.to_thread(fetch_customers, status, limit)
        await _enqueue_event({"type": "audit", "tool": name, "count": len(customers)})
        return {"result": customers}

    if name == "update_customer":
        updated = await asyncio.to_thread(
            update_customer_record, int(args.get("customer_id")), args.get("data", {})
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Customer not found")
        await _enqueue_event({"type": "update", "tool": name, "customer_id": updated["id"]})
        return {"result": updated}

    if name == "create_ticket":
        ticket = await asyncio.to_thread(
            create_ticket_record,
            int(args.get("customer_id")),
            str(args.get("issue")),
            str(args.get("priority")),
        )
        await _enqueue_event({"type": "ticket", "tool": name, "ticket_id": ticket["id"]})
        return {"result": ticket}

    if name == "get_customer_history":
        history = await asyncio.to_thread(fetch_history, int(args.get("customer_id")))
        await _enqueue_event({"type": "history", "tool": name, "count": len(history)})
        return {"result": history}

    raise HTTPException(status_code=404, detail=f"Unknown tool {name}")


@app.get("/events/stream")
async def stream_events():
    async def event_generator():
        while True:
            event = await event_queue.get()
            yield {"event": "update", "data": event}

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
