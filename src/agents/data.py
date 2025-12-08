"""CustomerDataAgent interacts with MCP tools only."""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..mcp_client import shared_mcp_client

logger = logging.getLogger(__name__)


class CustomerDataAgent:
    """Agent responsible for data retrieval and updates via MCP."""

    id: str = "customer-data-agent"
    name: str = "CustomerDataAgent"
    version: str = "0.1.0"

    @property
    def card(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "Retrieves and updates customer data using MCP tools",
            "capabilities": ["message/send"],
        }

    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON-RPC message/send calls."""
        context = params.get("context", {})
        operation = context.get("operation")
        args: Dict[str, Any] = context.get("args", {})
        if not operation:
            raise ValueError("operation is required in context")

        logger.info("DataAgent handling operation=%s args=%s", operation, args)
        result = None

        if operation == "get_customer":
            result = await shared_mcp_client.call_tool("get_customer", {"customer_id": args.get("customer_id")})
        elif operation == "list_customers":
            result = await shared_mcp_client.call_tool(
                "list_customers", {"status": args.get("status"), "limit": args.get("limit", 20)}
            )
        elif operation == "update_customer":
            result = await shared_mcp_client.call_tool(
                "update_customer", {"customer_id": args.get("customer_id"), "data": args.get("data", {})}
            )
        elif operation == "create_ticket":
            result = await shared_mcp_client.call_tool(
                "create_ticket",
                {
                    "customer_id": args.get("customer_id"),
                    "issue": args.get("issue"),
                    "priority": args.get("priority", "medium"),
                },
            )
        elif operation == "get_history":
            result = await shared_mcp_client.call_tool(
                "get_customer_history", {"customer_id": args.get("customer_id")}
            )
        else:
            raise ValueError(f"Unsupported operation {operation}")

        logger.info("DataAgent result=%s", result)
        return {
            "message": {"role": "assistant", "content": f"Operation {operation} completed"},
            "context": {"operation": operation, "result": result},
        }


data_agent = CustomerDataAgent()
