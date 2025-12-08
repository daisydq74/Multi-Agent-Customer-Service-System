"""Run end-to-end A2A + MCP demo flows."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn

from scripts.init_db import init_database
from src.mcp_client import shared_mcp_client

LOG_PATH = Path("demos/output/run.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def _send_to_router(base_url: str, message: str) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {"message": {"role": "user", "content": message}, "context": {}},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{base_url}/a2a/router", json=payload, timeout=60.0)
        try:
            data = resp.json()
        except Exception:
            data = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {"code": -32002, "message": f"Invalid response: {resp.text}"},
            }
        return data


def _start_api_server(host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config("src.a2a_http:app", host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)
    logger.info("A2A server started at http://%s:%s", host, port)
    return server


async def run_demo() -> None:
    db_path = os.getenv("DB_PATH", "./support.db")
    if not Path(db_path).exists():
        logger.info("Database not found; initializing")
        init_database(db_path)

    base_url = os.getenv("A2A_BASE_URL", "http://127.0.0.1:8000")
    os.environ.setdefault("A2A_BASE_URL", base_url)
    os.environ.setdefault("DB_PATH", db_path)

    server = _start_api_server("127.0.0.1", int(base_url.rsplit(":", 1)[-1]))

    queries = [
        "Get customer information for ID 5",
        "I'm customer 12345 and need help upgrading my account",
        "Show me all active customers who have open tickets",
        "I've been charged twice, please refund immediately!",
        "Update my email to new@email.com and show my ticket history",
    ]

    for query in queries:
        logger.info("Sending query: %s", query)
        result = await _send_to_router(base_url, query)
        if "error" in result:
            logger.warning("Router returned error: %s", result["error"])
            print(json.dumps(result, indent=2))
            print("-" * 40)
            continue
        logger.info("Result: %s", json.dumps(result, indent=2))
        print("-" * 40)
        print(f"Query: {query}")
        print(json.dumps(result, indent=2))

    await shared_mcp_client.close()
    server.should_exit = True
    logger.info("Demo complete. Logs written to %s", LOG_PATH)


if __name__ == "__main__":
    asyncio.run(run_demo())
