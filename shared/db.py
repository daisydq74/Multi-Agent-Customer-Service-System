import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

DB_PATH = Path(os.getenv("A2A_DB_PATH", "./database.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    issue TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    channel TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
);
"""

SAMPLE_CUSTOMERS = [
    ("Ana Customer", "ana@example.com", "active"),
    ("Brian Blocked", "brian@example.com", "delinquent"),
    ("Cara Care", "cara@example.com", "vip"),
]

SAMPLE_INTERACTIONS = [
    (1, "email", "Welcome email sent"),
    (1, "phone", "Reported login issue"),
    (2, "chat", "Billing dispute opened"),
    (3, "email", "Requested feature roadmap"),
]

async def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        cursor = await db.execute("SELECT COUNT(*) FROM customers")
        count = (await cursor.fetchone())[0]
        if count == 0:
            await db.executemany(
                "INSERT INTO customers(name, email, status) VALUES(?,?,?)",
                SAMPLE_CUSTOMERS,
            )
            await db.executemany(
                "INSERT INTO interactions(customer_id, channel, notes) VALUES(?,?,?)",
                SAMPLE_INTERACTIONS,
            )
            await db.commit()

async def get_db_connection(db_path: Path = DB_PATH):
    await init_db(db_path)
    return await aiosqlite.connect(db_path)

async def fetch_customer(customer_id: int) -> Optional[Dict[str, Any]]:
    async with await get_db_connection() as db:
        cursor = await db.execute(
            "SELECT id, name, email, status, created_at FROM customers WHERE id=?",
            (customer_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "status": row[3],
            "created_at": row[4],
        }

async def fetch_customers(status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    async with await get_db_connection() as db:
        if status:
            cursor = await db.execute(
                "SELECT id, name, email, status, created_at FROM customers WHERE status=? LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT id, name, email, status, created_at FROM customers LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "name": r[1], "email": r[2], "status": r[3], "created_at": r[4]}
            for r in rows
        ]

async def update_customer_record(customer_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    async with await get_db_connection() as db:
        existing = await fetch_customer(customer_id)
        if not existing:
            return None
        for key, value in data.items():
            if key not in {"name", "email", "status"}:
                continue
            await db.execute(f"UPDATE customers SET {key}=? WHERE id=?", (value, customer_id))
        await db.commit()
    return await fetch_customer(customer_id)

async def create_ticket_record(customer_id: int, issue: str, priority: str) -> Dict[str, Any]:
    async with await get_db_connection() as db:
        cursor = await db.execute(
            "INSERT INTO tickets(customer_id, issue, priority, status) VALUES(?,?,?,?)",
            (customer_id, issue, priority, "open"),
        )
        await db.commit()
        ticket_id = cursor.lastrowid
        cursor = await db.execute(
            "SELECT id, customer_id, issue, priority, status, created_at FROM tickets WHERE id=?",
            (ticket_id,),
        )
        row = await cursor.fetchone()
        return {
            "id": row[0],
            "customer_id": row[1],
            "issue": row[2],
            "priority": row[3],
            "status": row[4],
            "created_at": row[5],
        }

async def fetch_history(customer_id: int) -> List[Dict[str, Any]]:
    async with await get_db_connection() as db:
        cursor = await db.execute(
            "SELECT id, channel, notes, created_at FROM interactions WHERE customer_id=? ORDER BY created_at DESC",
            (customer_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "channel": r[1], "notes": r[2], "created_at": r[3]} for r in rows
        ]

async def add_history_record(customer_id: int, notes: str, channel: str = "agent") -> Dict[str, Any]:
    async with await get_db_connection() as db:
        cursor = await db.execute(
            "INSERT INTO interactions(customer_id, channel, notes) VALUES(?,?,?)",
            (customer_id, channel, notes),
        )
        await db.commit()
        record_id = cursor.lastrowid
        cursor = await db.execute(
            "SELECT id, channel, notes, created_at FROM interactions WHERE id=?",
            (record_id,),
        )
        row = await cursor.fetchone()
        return {"id": row[0], "channel": row[1], "notes": row[2], "created_at": row[3]}

__all__ = [
    "init_db",
    "fetch_customer",
    "fetch_customers",
    "update_customer_record",
    "create_ticket_record",
    "fetch_history",
    "add_history_record",
    "get_db_connection",
    "DB_PATH",
]
