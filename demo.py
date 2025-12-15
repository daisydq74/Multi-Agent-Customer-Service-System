import asyncio
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

AGENT_PORTS = {
    "mcp": 8000,
    "data": 8011,
    "support": 8012,
    "billing": 8013,
    "router": 8010,
}


def build_request(prompt: str) -> dict:
    base_message = build_text_message(prompt, role=Role.user)
    params = MessageSendParams(
        message=Message(messageId=base_message.messageId, role=Role.user, parts=base_message.parts)
    )
    return {"jsonrpc": "2.0", "id": "demo", "method": "message/send", "params": params.model_dump()}


async def wait_for_health(port: int, name: str) -> None:
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient() as client:
        for _ in range(60):
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"{name} healthy at {url}")
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError(f"{name} failed to become healthy on {url}")


async def start_server(app, port: int, name: str) -> Tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]
    task: asyncio.Task[None] = asyncio.create_task(server.serve())
    await wait_for_health(port, name)
    return server, task


async def stop_servers(servers: List[Tuple[uvicorn.Server, asyncio.Task[None]]]) -> None:
    for server, _ in servers:
        server.should_exit = True
    await asyncio.gather(*(task for _, task in servers), return_exceptions=True)


async def run_prompt(prompt: str) -> None:
    request_body = build_request(prompt)
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"http://localhost:{AGENT_PORTS['router']}/rpc", json=request_body)
        response.raise_for_status()
        result = response.json().get("result")
    print(f"\n=== Prompt: {prompt} ===")
    if not result:
        print("No result returned")
        return
    history = result.get("history", []) if isinstance(result, dict) else []
    final_message = history[-1]["parts"][0]["text"] if history and "parts" in history[-1] else result
    print(final_message)


async def run_demo_queries() -> None:
    prompts = [
        "Get customer information for ID 5",
        "I'm customer 5 and need help upgrading my account",
        "Show me all active customers who have open tickets",
        "I've been charged twice, please refund immediately! I'm customer 1",
        "Update my email to new@email.com and show my ticket history for customer 1",
    ]
    for prompt in prompts:
        await run_prompt(prompt)


async def main() -> None:
    servers: List[Tuple[uvicorn.Server, asyncio.Task[None]]] = []
    try:
        servers.append(await start_server(mcp_app, AGENT_PORTS["mcp"], "MCP Server"))
        servers.append(await start_server(data_app, AGENT_PORTS["data"], "Data Agent"))
        servers.append(await start_server(support_app, AGENT_PORTS["support"], "Support Agent"))
        servers.append(await start_server(billing_app, AGENT_PORTS["billing"], "Billing Agent"))
        servers.append(await start_server(router_app, AGENT_PORTS["router"], "Router Agent"))

        await run_demo_queries()
    finally:
        await stop_servers(servers)


if __name__ == "__main__":
    asyncio.run(main())
