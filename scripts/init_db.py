"""Initialize the SQLite database with sample data."""
from __future__ import annotations

import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from database_setup import DatabaseSetup

logger = logging.getLogger(__name__)


def init_database(db_path: str = "support.db") -> None:
    """Create the database and seed sample data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    db_file = Path(db_path)
    logger.info("Initializing database at %s", db_file.resolve())
    setup = DatabaseSetup(db_path)
    setup.connect()
    setup.create_tables()
    setup.create_triggers()
    setup.insert_sample_data()
    setup.close()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM customers WHERE id = ?", (12345,))
    if cursor.fetchone() is None:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO customers (id, name, email, phone, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                12345,
                "Test Customer 12345",
                "test12345@example.com",
                "000-000-0000",
                "active",
                now,
                now,
            ),
        )
        logger.info("Inserted seed customer 12345")

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE customer_id = ?", (12345,))
    ticket_count = cursor.fetchone()[0]
    if ticket_count == 0:
        cursor.executemany(
            """
            INSERT INTO tickets (customer_id, issue, status, priority)
            VALUES (?, ?, ?, ?)
            """,
            [
                (12345, "Assistance needed with account upgrade", "open", "medium"),
                (12345, "Follow-up on previous upgrade request", "in_progress", "low"),
            ],
        )
        logger.info("Inserted seed tickets for customer 12345")

    conn.commit()
    conn.close()
    logger.info("Database initialized with sample data")


if __name__ == "__main__":
    init_database()
