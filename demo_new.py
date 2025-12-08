import argparse
import asyncio
import json
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


def _repo_env(verbose: bool) -> dict:
    """Env for child processes: ensure local packages import + MCP url available."""
    repo_root = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MCP_SERVER_URL", "http://127.0.0.1:8000")
    env.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    if verbose and not env.get("OPENAI_API_KEY"):
        print("[demo] WARNING: OPENAI_API_KEY is not set. Agents that require LLM will fail.")
    return env


def start_service(name: str, cmd: List[str], env: dict, verbose: bool) -> subprocess.Popen:
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    if verbose:
        print(f"[demo] Starting {name}: {' '.join(cmd)}")
    # start_new_session=True lets us terminate process groups more reliably on mac/linux
    return subprocess.Popen(cmd, env=env, start_new_session=True, stdout=stdout, stderr=stderr)


async def wait_for_service(host: str, port: int, client: httpx.AsyncClient, verbose: bool, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://{host}:{port}/health"

    while time.monotonic() < deadline:
        try:
            response = await client.get(health_url, timeout=5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass

        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            await asyncio.sleep(0.25)

    raise RuntimeError(f"Service at {host}:{port} did not become ready")


def extract_answer_text(result: dict | None) -> str:
    if not result:
        return "No answer returned."

    try:
        text = result["status"]["message"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return "Unable to read response from router."

    stripped = (text or "").strip()
    if not stripped:
        return "No answer returned."

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(parsed, dict):
        for key in ("response", "message", "answer"):
            if isinstance(parsed.get(key), str):
                return parsed[key]
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return str(parsed)


async def run_scenarios(verbose: bool) -> None:
    env = _repo_env(verbose)
    processes: List[Tuple[str, subprocess.Popen]] = []
    startup_error: str | None = None

    try:
        for name, cmd, _ in SERVICES:
            proc = start_service(name, cmd, env, verbose)
            processes.append((name, proc))

        async with httpx.AsyncClient(timeout=60.0) as client:
            for _, _, port in SERVICES:
                try:
                    await wait_for_service("127.0.0.1", port, client, verbose)
                except Exception as exc:  # pragma: no cover - startup guard
                    startup_error = f"Service on port {port} did not start: {exc}"
                    break

            test_scenarios = [
                ("Simple Query", "Get customer information for ID 5"),
                ("Coordinated Query", "I'm customer 12345 and need help upgrading my account"),
                ("Complex Query", "Show me all active customers who have open tickets"),
                ("Escalation", "I've been charged twice, please refund immediately!"),
                ("Multi-Intent", "Update my email to new@email.com and show my ticket history"),
            ]

            for _, prompt in test_scenarios:
                print(f"Q: {prompt}")
                if startup_error:
                    print(f"A: {startup_error}\n")
                    continue

                request_body = build_request(prompt)
                try:
                    resp = await client.post(ROUTER_RPC, json=request_body)
                    if resp.status_code != 200:
                        answer = f"Router returned status {resp.status_code}."
                    else:
                        try:
                            result = resp.json().get("result")
                            answer = extract_answer_text(result)
                        except ValueError:
                            answer = "Router response was not valid JSON."
                except httpx.HTTPError as exc:
                    answer = f"Request failed: {exc}"

                print(f"A: {answer}\n")

    finally:
        for name, proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if verbose:
                print(f"[demo] Stopped {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demo scenarios against local agents.")
    parser.add_argument("--verbose", action="store_true", help="Show service logs and lifecycle events.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_scenarios(verbose=args.verbose))


if __name__ == "__main__":
    main()
