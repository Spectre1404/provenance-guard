"""SQLite-backed structured audit log and content store for Provenance Guard.

Two tables:
  - content: one row per submission (current state, incl. status for appeals)
  - audit_log: append-only structured event log (classifications + appeals)

The audit_log is the canonical record graders rely on; it is append-only so an
appeal never overwrites the original decision — it adds a new event.
"""

import json
import sqlite3
from contextlib import contextmanager

DB_PATH = "provenance.db"


@contextmanager
def get_conn():
    """Yield a SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS content (
                content_id   TEXT PRIMARY KEY,
                creator_id   TEXT NOT NULL,
                text         TEXT NOT NULL,
                attribution  TEXT,
                confidence   REAL,
                status       TEXT NOT NULL DEFAULT 'classified',
                created_at   TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id   TEXT NOT NULL,
                event_type   TEXT NOT NULL,   -- 'classification' | 'appeal'
                timestamp    TEXT NOT NULL,
                payload      TEXT NOT NULL    -- JSON blob of the full event
            )
            """
        )


def save_content(content_id, creator_id, text, attribution, confidence,
                 status, created_at):
    """Insert a new classified content row."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO content
                (content_id, creator_id, text, attribution, confidence,
                 status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (content_id, creator_id, text, attribution, confidence,
             status, created_at),
        )


def get_content(content_id):
    """Return a content row as a dict, or None if not found."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
        return dict(row) if row else None


def update_status(content_id, status):
    """Update the status of a content row (e.g. -> 'under_review')."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE content SET status = ? WHERE content_id = ?",
            (status, content_id),
        )


def write_log(content_id, event_type, timestamp, payload):
    """Append a structured event to the audit log. `payload` is a dict."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO audit_log (content_id, event_type, timestamp, payload)
            VALUES (?, ?, ?, ?)
            """,
            (content_id, event_type, timestamp, json.dumps(payload)),
        )


def get_log(limit=50):
    """Return the most recent audit-log entries as a list of dicts.

    Each entry's JSON payload is parsed back into a nested dict so the /log
    endpoint returns clean structured JSON.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT content_id, event_type, timestamp, payload
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    entries = []
    for row in rows:
        entry = {
            "content_id": row["content_id"],
            "event_type": row["event_type"],
            "timestamp": row["timestamp"],
        }
        entry.update(json.loads(row["payload"]))
        entries.append(entry)
    return entries
