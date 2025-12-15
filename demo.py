import asyncio
import os
from typing import List, Tuple

import httpx
import uvicorn

from agents.billing import app as billing_app
from agents.data import app as data_app
from agents.router import app as router_app
from agents.support import app as support_app
from langgraph_sdk.types import Message, MessageSendParams, Role
from mcp_server.app import app as mcp_app
from shared.message_utils import build_text_message

ROUTER_RPC = os.getenv("ROUTER_RPC", "http://localhost:8010/rpc")


def build_request(prompt: str) -> dict:
    base_message = build_text_message(prompt, role=Role.user)
    params = MessageSendParams(
        message=Message(messageId=base_message.messageId, role=Role.user, parts=base_message.parts)
    )
    return {"jsonrpc": "2.0", "id": "demo", "method": "message/send", "params": params.model_dump()}


def print_response(prompt: str, result: dict | None) -> None:
    print(f"=== Prompt: {prompt} ===")
    if not result:
        print("No result returned")
    else:
        history = result.get("history") or []
        final_text = ""
        if history:
            reply = history[-1]
            parts = reply.get("parts") or []
            if parts and isinstance(parts[0], dict):
                final_text = parts[0].get("text", "")
        print(f"Final Answer: {final_text or 'No reply received'}")
    print()


async def start_server(app, port: int, name: str) -> Tuple[uvicorn.Server, asyncio.Task[None]]:
    """Start a uvicorn server in the background and wait for it to be ready."""

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]

    task: asyncio.Task[None] = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    print(f"{name} started on port {port}")
    return server, task


async def stop_servers(servers: List[Tuple[uvicorn.Server, asyncio.Task[None]]]) -> None:
    for server, task in servers:
        server.should_exit = True
    await asyncio.gather(*(task for _, task in servers), return_exceptions=True)


async def run_demo_queries() -> None:
    prompts = [
        "Get customer information for ID 5",
        "I'm customer 12345 and need help upgrading my account",
        "Show me all active customers who have open tickets",
        "I'm customer 5 and I've been charged twice, please refund immediately!",
        "I'm customer 5, Update my email to new@email.com and show my ticket history",
    ]

    async with httpx.AsyncClient() as client:
        for prompt in prompts:
            request_body = build_request(prompt)
            response = await client.post(ROUTER_RPC, json=request_body)
            response.raise_for_status()
            result = response.json().get("result")
            print_response(prompt, result)


async def main():
    servers: List[Tuple[uvicorn.Server, asyncio.Task[None]]] = []
    try:
        servers.append(await start_server(mcp_app, 8000, "MCP Server"))
        servers.append(await start_server(data_app, 8011, "Data Agent"))
        servers.append(await start_server(support_app, 8012, "Support Agent"))
        servers.append(await start_server(billing_app, 8013, "Billing Agent"))
        servers.append(await start_server(router_app, 8010, "Router Agent"))

        await run_demo_queries()
    finally:
        await stop_servers(servers)


if __name__ == "__main__":
    asyncio.run(main())
