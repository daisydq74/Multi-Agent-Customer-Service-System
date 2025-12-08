"""Initialize the SQLite database with sample data."""
from __future__ import annotations

import logging
from pathlib import Path

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
    logger.info("Database initialized with sample data")


if __name__ == "__main__":
    init_database()
