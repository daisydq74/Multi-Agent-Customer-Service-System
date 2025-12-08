import asyncio
import os
import subprocess
import sys
import time
from typing import List, Tuple

import httpx

from langgraph_sdk.types import Message, MessageSendParams, Role
from shared.message_utils import build_text_message

ROUTER_RPC = os.getenv("ROUTER_RPC", "http://localhost:8010/rpc")

Service = Tuple[str, List[str], int]

SERVICES: List[Service] = [
    ("mcp_server", [sys.executable, "-m", "uvicorn", "mcp_server.app:app", "--port", "8000"], 8000),
    ("router", [sys.executable, "-m", "uvicorn", "agents.router:app", "--port", "8010"], 8010),
    ("data", [sys.executable, "-m", "uvicorn", "agents.data:app", "--port", "8011"], 8011),
    ("support", [sys.executable, "-m", "uvicorn", "agents.support:app", "--port", "8012"], 8012),
    ("billing", [sys.executable, "-m", "uvicorn", "agents.billing:app", "--port", "8013"], 8013),
]


def build_request(prompt: str) -> dict:
    base_message = build_text_message(prompt, role=Role.user)
    params = MessageSendParams(
        message=Message(messageId=base_message.messageId, role=Role.user, parts=base_message.parts)
    )
    return {"jsonrpc": "2.0", "id": "demo", "method": "message/send", "params": params.model_dump()}


def print_response(scenario: str, prompt: str, result: dict | None) -> None:
    print(f"=== Scenario: {scenario} ===")
    print(f"Prompt: {prompt}")
    if not result:
        print("No result returned")
    else:
        print("Router task result:", result)
    print()


def start_service(name: str, cmd: List[str]) -> subprocess.Popen[bytes]:
    print(f"[demo] Starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


async def wait_for_service(client: httpx.AsyncClient, name: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{port}/health"
    while time.monotonic() < deadline:
        try:
            response = await client.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[demo] {name} healthy at {url}")
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Service {name} at {url} did not become healthy within {timeout} seconds")


async def main():
    processes: List[Tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for name, cmd, _ in SERVICES:
            processes.append((name, start_service(name, cmd)))

        async with httpx.AsyncClient() as client:
            for name, _, port in SERVICES:
                await wait_for_service(client, name, port)

            print("[demo] All services are healthy. Running demo scenarios...\n")

            test_scenarios = [
                ("Simple Query", "Get customer information for ID 5"),
                ("Coordinated Query", "I'm customer 12345 and need help upgrading my account"),
                ("Complex Query", "Show me all active customers who have open tickets"),
                ("Escalation", "I've been charged twice, please refund immediately!"),
                ("Multi-Intent", "Update my email to new@email.com and show my ticket history"),
            ]

            for scenario, prompt in test_scenarios:
                request_body = build_request(prompt)
                response = await client.post(ROUTER_RPC, json=request_body)
                response.raise_for_status()
                result = response.json().get("result")
                print_response(scenario, prompt, result)
    finally:
        print("[demo] Shutting down services...")
        for name, proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            print(f"[demo] Stopped {name}")


if __name__ == "__main__":
    asyncio.run(main())
