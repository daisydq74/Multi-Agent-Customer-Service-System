import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import httpx

from langgraph_sdk.types import Message, MessageSendParams, Role
from shared.message_utils import build_text_message

ROUTER_RPC = os.getenv("ROUTER_RPC", "http://127.0.0.1:8010/rpc")

Service = Tuple[str, List[str], int]

# Use "python agents/*.py" because your scripts already start uvicorn internally
SERVICES: List[Service] = [
    ("mcp_server", [sys.executable, "-m", "uvicorn", "mcp_server.app:app", "--host", "127.0.0.1", "--port", "8000"], 8000),
    ("data", [sys.executable, "-u", "agents/data.py"], 8011),
    ("support", [sys.executable, "-u", "agents/support.py"], 8012),
    ("billing", [sys.executable, "-u", "agents/billing.py"], 8013),
    ("router", [sys.executable, "-u", "agents/router.py"], 8010),
]


def build_request(prompt: str) -> dict:
    base_message = build_text_message(prompt, role=Role.user)
    params = MessageSendParams(
        message=Message(messageId=base_message.messageId, role=Role.user, parts=base_message.parts)
    )
    return {"jsonrpc": "2.0", "id": "demo", "method": "message/send", "params": params.model_dump()}


def print_response(scenario: str, prompt: str, result: dict | None) -> None:
    print(f"\n=== Scenario: {scenario} ===")
    print(f"Prompt: {prompt}")
    if result is None:
        print("No result returned")
    else:
        print("Router task result:", result)


def _repo_env() -> dict:
    """Env for child processes: ensure local packages import + MCP url available."""
    repo_root = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MCP_SERVER_URL", "http://127.0.0.1:8000")
    env.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    # IMPORTANT: OPENAI_API_KEY must be provided by user shell. Do not hardcode.
    if not env.get("OPENAI_API_KEY"):
        print("[demo] WARNING: OPENAI_API_KEY is not set. Agents that require LLM will fail.")
    return env


def start_service(name: str, cmd: List[str], env: dict) -> subprocess.Popen:
    print(f"[demo] Starting {name}: {' '.join(cmd)}")
    # start_new_session=True lets us terminate process groups more reliably on mac/linux
    return subprocess.Popen(cmd, env=env, start_new_session=True)


async def wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # socket connect check is more reliable than /health (which may not exist)
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            await asyncio.sleep(0.25)
    raise RuntimeError(f"Port {host}:{port} did not open within {timeout} seconds")


async def main():
    env = _repo_env()
    processes: List[Tuple[str, subprocess.Popen]] = []

    try:
        # Start services
        for name, cmd, _ in SERVICES:
            proc = start_service(name, cmd, env)
            processes.append((name, proc))

        # Wait for ports
        for name, _, port in SERVICES:
            await wait_for_port("127.0.0.1", port, timeout=25.0)
            print(f"[demo] {name} port ready at 127.0.0.1:{port}")

        print("\n[demo] All ports are ready. Running demo scenarios...\n")

        test_scenarios = [
            ("Simple Query", "Get customer information for ID 5"),
            ("Coordinated Query", "I'm customer 12345 and need help upgrading my account"),
            ("Complex Query", "Show me all active customers who have open tickets"),
            ("Escalation", "I've been charged twice, please refund immediately!"),
            ("Multi-Intent", "Update my email to new@email.com and show my ticket history"),
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            for scenario, prompt in test_scenarios:
                request_body = build_request(prompt)
                resp = await client.post(ROUTER_RPC, json=request_body)
                resp.raise_for_status()
                result = resp.json().get("result")
                print_response(scenario, prompt, result)

        print("\n[demo] Done.")

    finally:
        print("\n[demo] Shutting down services...")
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
