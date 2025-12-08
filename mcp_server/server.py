"""FastMCP server exposing customer support tools."""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from . import db

logger = logging.getLogger(__name__)
mcp = FastMCP("customer-support-mcp")


@mcp.tool()
def get_customer(customer_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a customer by ID."""
    customer = db.get_customer(customer_id)
    result = db.serialize_customer(customer) if customer else None
    logger.info("MCP get_customer id=%s -> %s", customer_id, result)
    return result


@mcp.tool()
def list_customers(status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """List customers filtered by status."""
    customers = db.list_customers(status=status, limit=limit)
    result = [db.serialize_customer(c) for c in customers]
    logger.info("MCP list_customers status=%s limit=%s -> %d", status, limit, len(result))
    return result


@mcp.tool()
def update_customer(customer_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a customer record with provided fields."""
    updated = db.update_customer(customer_id, data)
    result = db.serialize_customer(updated) if updated else None
    logger.info("MCP update_customer id=%s fields=%s -> %s", customer_id, list(data.keys()), result)
    return result


@mcp.tool()
def create_ticket(customer_id: int, issue: str, priority: str = "medium") -> Dict[str, Any]:
    """Create a new support ticket."""
    ticket = db.create_ticket(customer_id, issue, priority)
    result = db.serialize_ticket(ticket)
    logger.info(
        "MCP create_ticket customer_id=%s priority=%s -> ticket_id=%s",
        customer_id,
        priority,
        ticket.id,
    )
    return result


@mcp.tool()
def get_customer_history(customer_id: int) -> List[Dict[str, Any]]:
    """Return ticket history for a customer."""
    tickets = db.get_customer_history(customer_id)
    result = [db.serialize_ticket(t) for t in tickets]
    logger.info("MCP get_customer_history id=%s -> %d tickets", customer_id, len(result))
    return result


def run() -> None:
    """Run the MCP server using stdio transport."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting MCP server")
    mcp.run()


if __name__ == "__main__":
    run()
