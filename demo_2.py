import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import List, Tuple, Optional

import httpx
from langgraph_sdk.types import Message, MessageSendParams, Role

from shared.message_utils import build_text_message

ROUTER_RPC = os.getenv("ROUTER_RPC", "http://127.0.0.1:8010/rpc")
VERBOSE = os.getenv("DEMO_VERBOSE", "0") == "1"

Service = Tuple[str, List[str], int]

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
        message=Message(
            messageId=base_message.messageId,
            role=Role.user,
            parts=base_message.parts,
        )
    )
    return {
        "jsonrpc": "2.0",
        "id": "demo",
        "method": "message/send",
        "params": params.model_dump(),
    }


def start_service(name: str, cmd: List[str]) -> subprocess.Popen[bytes]:
    if VERBOSE:
        print(f"[demo] Starting {name}: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        stdout=None if VERBOSE else subprocess.DEVNULL,
        stderr=None if VERBOSE else subprocess.DEVNULL,
    )


def wait_for_port(host: str, port: int, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Port not ready: {host}:{port}")


def _extract_agent_text(result: Optional[dict]) -> str:
    if not result:
        return "No result returned"

    try:
        msg = (result.get("status") or {}).get("message") or {}
        parts = msg.get("parts") or []
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            return str(parts[0]["text"]).strip()
    except Exception:
        pass

    # fallback：history
    try:
        hist = result.get("history") or []
        for item in reversed(hist):
            if item.get("role") == "agent":
                parts = item.get("parts") or []
                if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                    return str(parts[0]["text"]).strip()
    except Exception:
        pass

    return "No answer text found"


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _try_parse_json(text: str):
    # 1) ```json ... ```
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    try:
        return json.loads(text)
    except Exception:
        return None


def _to_natural_answer(text: str) -> str:
    obj = _try_parse_json(text)
    if not isinstance(obj, dict):
        return text

    customer = obj.get("customer")
    if isinstance(customer, dict):
        lines = []
        cid = customer.get("id")
        name = customer.get("name")
        status = customer.get("status")
        email = customer.get("email")
        created = customer.get("created_at")
        if cid is not None or name:
            lines.append(f"Customer {cid}: {name} ({status})".strip())
        if email:
            lines.append(f"Email: {email}")
        if created:
            lines.append(f"Created: {created}")
        return "\n".join(lines).strip() or text

    return json.dumps(obj, indent=2, ensure_ascii=False)


def print_qa(prompt: str, answer: str) -> None:
    print(f"Q: {prompt}")
    print(f"A: {answer}")
    print()


async def main():
    processes: List[Tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for name, cmd, _port in SERVICES:
            processes.append((name, start_service(name, cmd)))

        for _name, _cmd, port in SERVICES:
            wait_for_port("127.0.0.1", port)

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            test_scenarios = [
                ("Get customer information for ID 5"),
                ("I'm customer 12345 and need help upgrading my account"),
                ("Show me all active customers who have open tickets"),
                ("I've been charged twice, please refund immediately!"),
                ("Update my email to new@email.com and show my ticket history"),
            ]

            for prompt in test_scenarios:
                req = build_request(prompt)
                try:
                    resp = await client.post(ROUTER_RPC, json=req)
                    resp.raise_for_status()
                    result = resp.json().get("result")
                    raw = _extract_agent_text(result)
                    answer = _to_natural_answer(raw)
                    print_qa(prompt, answer)
                except Exception as e:
                    print_qa(prompt, f"[ERROR] {type(e).__name__}: {e}")

    finally:
        for _name, proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
