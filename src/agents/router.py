"""RouterAgent orchestrates requests across specialists."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from .. import llm_backend
from ..a2a_client import send_json_rpc

logger = logging.getLogger(__name__)


def _extract_customer_id(text: str) -> Optional[int]:
    patterns = [
        r"\b(?:customer\s*)?id\b\s*[:#]?\s*(\d+)\b",
        r"\bID\b\s*[:#]?\s*(\d+)\b",
        r"\b(\d{3,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_email(text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


class RouterAgent:
    """Agent coordinating plan creation and delegation."""

    id: str = "router-agent"
    name: str = "RouterAgent"
    version: str = "0.1.0"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.data_endpoint = f"{self.base_url}/a2a/data"
        self.support_endpoint = f"{self.base_url}/a2a/support"

    @property
    def card(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "Routes customer intents to specialist agents",
            "capabilities": ["message/send"],
            "endpoints": {
                "message/send": f"{self.base_url}/a2a/router",
            },
        }

    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        user_message: str = params.get("message", {}).get("content", "")
        plan = self._build_plan(user_message)
        logger.info("Router plan=%s", plan)

        data_context: List[Dict[str, Any]] = []

        async def call_data(operation: str, args: Dict[str, Any]) -> Any:
            payload = self._rpc_payload(user_message, {"operation": operation, "args": args})
            response = await send_json_rpc("Router", "CustomerData", self.data_endpoint, payload)
            if "error" in response:
                logger.error("Data agent error: %s", response["error"])
                return {"error": response["error"]}
            return response.get("result", {}).get("context", {}).get("result")

        # Route based on action
        if plan["action"] == "get_customer":
            customer_id = plan.get("customer_id") or _extract_customer_id(user_message)
            if customer_id is not None:
                customer = await call_data("get_customer", {"customer_id": customer_id})
                data_context.append({"customer": customer})
                if plan.get("need_history"):
                    history = await call_data("get_history", {"customer_id": customer_id})
                    data_context.append({"history": history})
            else:
                return {"message": "Please provide a customer ID to continue."}

        elif plan["action"] == "upgrade_help":
            customer_id = plan.get("customer_id") or _extract_customer_id(user_message)
            if customer_id:
                customer = await call_data("get_customer", {"customer_id": customer_id})
                history = await call_data("get_history", {"customer_id": customer_id})
                data_context.append({"customer": customer, "history": history})
            else:
                data_context.append({"info": "No customer id provided for upgrade"})

        elif plan["action"] == "active_open_tickets":
            customers = await call_data("list_customers", {"status": "active", "limit": 200}) or []
            active_with_open: List[Dict[str, Any]] = []
            for cust in customers:
                history = await call_data("get_history", {"customer_id": cust.get("id")}) or []
                open_tickets = [t for t in history if t.get("status") in {"open", "in_progress"}]
                if open_tickets:
                    active_with_open.append({"customer": cust, "tickets": open_tickets})
            data_context.append({"active_open_customers": active_with_open})

        elif plan["action"] == "refund_urgent":
            customer_id = plan.get("customer_id") or _extract_customer_id(user_message)
            ticket = None
            if customer_id is not None:
                ticket = await call_data(
                    "create_ticket",
                    {
                        "customer_id": customer_id,
                        "issue": "Urgent billing refund requested (charged twice)",
                        "priority": "high",
                    },
                )
                history = await call_data("get_history", {"customer_id": customer_id})
                data_context.append({"ticket": ticket, "history": history})
            else:
                data_context.append({"error": "customer_id missing for refund"})
            plan["urgency"] = "high"

        elif plan["action"] == "update_email_and_history":
            customer_id = plan.get("customer_id") or _extract_customer_id(user_message)
            email = plan.get("update", {}).get("email") or _extract_email(user_message)
            update_result = None
            if customer_id and email:
                update_result = await call_data(
                    "update_customer", {"customer_id": customer_id, "data": {"email": email}}
                )
                history = await call_data("get_history", {"customer_id": customer_id})
                data_context.append({"updated": update_result, "history": history})
            else:
                data_context.append({"error": "customer id or email missing"})

        support_payload = self._rpc_payload(
            user_message,
            {
                "plan": plan,
                "data": data_context,
                "notes": params.get("context", {}).get("notes"),
            },
        )
        support_reply = await send_json_rpc("Router", "Support", self.support_endpoint, support_payload)
        if "error" in support_reply:
            logger.error("Support agent error: %s", support_reply["error"])
            return {"error": support_reply["error"]}
        return support_reply.get("result") or {}

    def _rpc_payload(self, message_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {"role": "user", "content": message_content},
                "context": context,
            },
        }

    def _build_plan(self, user_message: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a planning agent. Output ONLY JSON. Schema: {"  # noqa: E501
            "\n  action: one of ['get_customer','upgrade_help','active_open_tickets','refund_urgent','update_email_and_history'],"
            "\n  customer_id: optional int,"
            "\n  update: optional object (e.g. {\"email\": value}),"
            "\n  urgency: low|medium|high,"
            "\n  need_history: bool,"
            "\n  notes: string"
            "\n}. Do not add commentary."
        )
        raw_plan = llm_backend.generate_text(system_prompt=system_prompt, user_prompt=user_message)
        try:
            plan = json.loads(raw_plan)
        except Exception:
            plan = {}

        plan.setdefault("action", self._infer_action(user_message))
        plan.setdefault("customer_id", _extract_customer_id(user_message))
        plan.setdefault("urgency", "medium")
        plan.setdefault("need_history", False)
        plan.setdefault("notes", "")

        if "charged twice" in user_message.lower() or "refund" in user_message.lower():
            plan["action"] = "refund_urgent"
            plan["urgency"] = "high"
        if "upgrade" in user_message.lower():
            plan["action"] = plan.get("action") or "upgrade_help"
        if "update" in user_message.lower() and "email" in user_message.lower():
            plan["action"] = "update_email_and_history"
            plan.setdefault("update", {})
            email = _extract_email(user_message)
            if email:
                plan["update"]["email"] = email
        if plan.get("customer_id") is None:
            extracted = _extract_customer_id(user_message)
            if extracted is not None:
                plan["customer_id"] = extracted
        elif isinstance(plan.get("customer_id"), str) and plan.get("customer_id").isdigit():
            plan["customer_id"] = int(plan["customer_id"])
        return plan

    def _infer_action(self, user_message: str) -> str:
        text = user_message.lower()
        if "upgrade" in text:
            return "upgrade_help"
        if "charged" in text or "refund" in text:
            return "refund_urgent"
        if "active" in text and "ticket" in text:
            return "active_open_tickets"
        if "update" in text and "email" in text:
            return "update_email_and_history"
        return "get_customer"


router_agent = RouterAgent(base_url="http://127.0.0.1:8000")
