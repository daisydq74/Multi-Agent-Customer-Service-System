"""Helper to send JSON-RPC agent-to-agent calls with logging."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)


async def send_json_rpc(from_agent: str, to_agent: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a JSON-RPC payload to another agent and log request/response."""
    pretty_request = json.dumps(payload, indent=2)
    logger.info("[A2A] from=%s to=%s request=%s", from_agent, to_agent, pretty_request)
    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    pretty_response = json.dumps(data, indent=2)
    logger.info("[A2A] from=%s to=%s response=%s", from_agent, to_agent, pretty_response)
    return data
