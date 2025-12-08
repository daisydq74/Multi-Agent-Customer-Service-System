import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from langgraph_sdk.types import Message, MessageSendParams, Role

from shared.message_utils import build_text_message

ROUTER_RPC = os.getenv("ROUTER_RPC", "http://127.0.0.1:8010/rpc")

Service = Tuple[str, List[str], int]

# Use "python agents/*.py" because those scripts already start uvicorn internally.
SERVICES: List[Service] = [
    (
        "mcp_server",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mcp_server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        8000,
    ),
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


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    return t


def _try_parse_json(text: str) -> Any | None:
    t = _strip_code_fences(text)

    try:
        return json.loads(t)
    except Exception:
        pass

    start_candidates = [t.find("{"), t.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)

    end_obj = t.rfind("}")
    end_arr = t.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        return None

    snippet = t[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _format_from_struct(obj: Any) -> str:
    if isinstance(obj, dict):
        if obj.get("error"):
            return f"Error: {obj.get('error')}"

        customer = obj.get("customer")
        if isinstance(customer, dict) and customer.get("id") is not None:
            cid = customer.get("id")
            name = customer.get("name") or "Unknown"
            email = customer.get("email") or "N/A"
            status = customer.get("status") or "unknown"
            created = customer.get("created_at") or "N/A"

            bits = [
                f"Customer #{cid}: {name} ({status}).",
                f"Email: {email}.",
                f"Created: {created}.",
            ]
            history = obj.get("history")
            if isinstance(history, list) and history:
                bits.append(f"Tickets: {len(history)} found.")
            return " ".join(bits)

        raw_tool = obj.get("raw_tool_result")
        if isinstance(raw_tool, dict) and raw_tool.get("needs_more_info") and raw_tool.get("question"):
            return str(raw_tool["question"]).strip()

        if isinstance(obj.get("answer"), str) and obj["answer"].strip():
            return obj["answer"].strip()

    return json.dumps(obj, ensure_ascii=False)


def extract_answer_from_router_result(result: Dict[str, Any] | None) -> str:
    if not result:
        return "No answer returned."

    status = result.get("status") or {}
    msg = status.get("message")
    if isinstance(msg, dict):
        parts = msg.get("parts") or []
        if parts and isinstance(parts[0], dict) and isinstance(parts[0].get("text"), str):
            text = parts[0]["text"].strip()
            parsed = _try_parse_json(text)
            return _format_from_struct(parsed) if parsed is not None else text

    history = result.get("history") or []
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            parts = last.get("parts") or []
            if parts and isinstance(parts[0], dict) and isinstance(parts[0].get("text"), str):
                text = parts[0]["text"].strip()
                parsed = _try_parse_json(text)
                return _format_from_struct(parsed) if parsed is not None else text

    return "No readable answer found in result."


def start_service(name: str, cmd: List[str], log_dir: Path) -> subprocess.Popen[bytes]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = open(log_path, "ab", buffering=0)

    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_port(host: str, port: int, timeout_s: float = 25.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"Service on {host}:{port} did not become ready within {timeout_s} seconds")


async def main() -> None:
    log_dir = Path(os.getenv("DEMO_LOG_DIR", ".demo_logs"))

    processes: List[Tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for name, cmd, _ in SERVICES:
            processes.append((name, start_service(name, cmd, log_dir)))

        for _, _, port in SERVICES:
            wait_for_port("127.0.0.1", port)

        scenarios = [
            "Get customer information for ID 5",
            "I'm customer 12345 and need help upgrading my account",
            "Show me all active customers who have open tickets",
            "I've been charged twice, please refund immediately!",
            "Update my email to new@email.com and show my ticket history",
        ]

        timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i, prompt in enumerate(scenarios, start=1):
                req = build_request(prompt)
                try:
                    resp = await client.post(ROUTER_RPC, json=req)
                    resp.raise_for_status()
                    result = resp.json().get("result")
                    answer = extract_answer_from_router_result(result)
                except Exception as e:
                    answer = f"Error: {type(e).__name__}: {e}"

                print(f"Q{i}: {prompt}")
                print(f"A{i}: {answer}")
                print()
    finally:
        for _, proc in reversed(processes):
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


if __name__ == "__main__":
    asyncio.run(main())
