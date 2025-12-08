"""SQLite helpers for MCP server tools."""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Customer:
    id: int
    name: str
    email: Optional[str]
    phone: Optional[str]
    status: str


@dataclass
class Ticket:
    id: int
    customer_id: int
    issue: str
    status: str
    priority: str
    created_at: str


def get_db_path() -> str:
    """Return configured database path."""
    return os.getenv("DB_PATH", "./support.db")


def connect_db() -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    logger.debug("Connected to database at %s", db_path)
    return conn


def get_customer(customer_id: int) -> Optional[Customer]:
    """Retrieve a customer by ID."""
    with connect_db() as conn:
        row = conn.execute(
            "SELECT id, name, email, phone, status FROM customers WHERE id = ?",
            (customer_id,),
        ).fetchone()
        if not row:
            return None
        return Customer(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            phone=row["phone"],
            status=row["status"],
        )


def list_customers(status: Optional[str], limit: int = 20) -> List[Customer]:
    """List customers filtered by status and limit."""
    status_filter = status or "%"
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, name, email, phone, status
            FROM customers
            WHERE status LIKE ?
            ORDER BY id
            LIMIT ?
            """,
            (status_filter, limit),
        ).fetchall()
        return [
            Customer(
                id=row["id"],
                name=row["name"],
                email=row["email"],
                phone=row["phone"],
                status=row["status"],
            )
            for row in rows
        ]


def update_customer(customer_id: int, data: Dict[str, Any]) -> Optional[Customer]:
    """Update customer fields and return updated record."""
    allowed_fields = {"name", "email", "phone", "status"}
    updates = {k: v for k, v in data.items() if k in allowed_fields and v is not None}
    if not updates:
        raise ValueError("No valid fields provided for update")

    set_clause = ", ".join(f"{field} = ?" for field in updates)
    params = list(updates.values()) + [customer_id]

    with connect_db() as conn:
        conn.execute(f"UPDATE customers SET {set_clause} WHERE id = ?", params)
        conn.commit()
    return get_customer(customer_id)


def create_ticket(customer_id: int, issue: str, priority: str = "medium") -> Ticket:
    """Create a support ticket."""
    if priority not in {"low", "medium", "high"}:
        raise ValueError("priority must be low, medium, or high")
    with connect_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tickets (customer_id, issue, priority)
            VALUES (?, ?, ?)
            """,
            (customer_id, issue, priority),
        )
        conn.commit()
        ticket_id = cursor.lastrowid
        row = conn.execute(
            "SELECT id, customer_id, issue, status, priority, created_at FROM tickets WHERE id = ?",
            (ticket_id,),
        ).fetchone()
        if not row:
            raise RuntimeError("Failed to fetch created ticket")
        return Ticket(
            id=row["id"],
            customer_id=row["customer_id"],
            issue=row["issue"],
            status=row["status"],
            priority=row["priority"],
            created_at=row["created_at"],
        )


def get_customer_history(customer_id: int) -> List[Ticket]:
    """Return all tickets for a given customer."""
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, customer_id, issue, status, priority, created_at
            FROM tickets
            WHERE customer_id = ?
            ORDER BY created_at DESC
            """,
            (customer_id,),
        ).fetchall()
        return [
            Ticket(
                id=row["id"],
                customer_id=row["customer_id"],
                issue=row["issue"],
                status=row["status"],
                priority=row["priority"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


def serialize_customer(customer: Customer) -> Dict[str, Any]:
    """Convert Customer dataclass to dict."""
    return {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "status": customer.status,
    }


def serialize_ticket(ticket: Ticket) -> Dict[str, Any]:
    """Convert Ticket dataclass to dict."""
    return {
        "id": ticket.id,
        "customer_id": ticket.customer_id,
        "issue": ticket.issue,
        "status": ticket.status,
        "priority": ticket.priority,
        "created_at": ticket.created_at,
    }
