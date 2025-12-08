"""FastAPI server exposing agent-to-agent HTTP endpoints."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request

from src.agents.data import CustomerDataAgent, data_agent
from src.agents.router import RouterAgent
from src.agents.support import SupportAgent, support_agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

BASE_URL = os.getenv("A2A_BASE_URL", "http://127.0.0.1:8000")

router = RouterAgent(base_url=BASE_URL)
data = data_agent
support = support_agent

app = FastAPI(title="Multi-Agent Customer Service System")


def _card(agent: Any, path: str) -> Dict[str, Any]:
    card = dict(agent.card)
    card.setdefault("endpoint", f"{BASE_URL}{path}")
    return card


async def _handle_rpc(agent: Any, request: Request) -> Dict[str, Any]:
    body = await request.json()
    req_id = body.get("id")
    if body.get("method") != "message/send":
        raise HTTPException(status_code=400, detail="Only message/send supported")
    params = body.get("params") or {}
    try:
        result = await agent.handle_message(params)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as exc:  # pragma: no cover - ensure HTTP layer stability
        logger.error("Agent error for %s: %s", getattr(agent, "name", agent), exc)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc)},
        }


@app.get("/a2a/router")
async def router_card() -> Dict[str, Any]:
    return _card(router, "/a2a/router")


@app.post("/a2a/router")
async def router_rpc(request: Request) -> Dict[str, Any]:
    return await _handle_rpc(router, request)


@app.get("/a2a/data")
async def data_card() -> Dict[str, Any]:
    return _card(data, "/a2a/data")


@app.post("/a2a/data")
async def data_rpc(request: Request) -> Dict[str, Any]:
    return await _handle_rpc(data, request)


@app.get("/a2a/support")
async def support_card() -> Dict[str, Any]:
    return _card(support, "/a2a/support")


@app.post("/a2a/support")
async def support_rpc(request: Request) -> Dict[str, Any]:
    return await _handle_rpc(support, request)
