# db/init_db.py

from __future__ import annotations
import asyncio
import sys
from pathlib import Path

import aiosqlite

# Allow running directly as `python db/init_db.py` (adds project root to path).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def create_tables(db_path: Path | None = None) -> None:
    """
    Initialise the database by executing schema.sql.
    Idempotent: CREATE TABLE IF NOT EXISTS means safe to re-run.
    Called by main.py before the pipeline starts.
    """
    path = db_path or settings.db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    async with aiosqlite.connect(str(path)) as conn:
        await conn.executescript(sql)
        await conn.commit()


async def reset_database(db_path: Path | None = None) -> None:
    """
    Delete all existing audit data and recreate the schema from scratch.
    Called at the start of every new Streamlit run so each audit shows
    only its own results — no carryover from previous runs.
    """
    path = Path(db_path or settings.db_path)
    if path.exists():
        path.unlink()
    await create_tables(db_path)


if __name__ == "__main__":
    asyncio.run(create_tables())
    print(f"Database initialised at {settings.db_path}")
