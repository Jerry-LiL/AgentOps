"""Append-only run log in SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


class Tracer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                event TEXT NOT NULL,
                server TEXT,
                tool TEXT,
                detail TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id, id)")
        self._conn.commit()

    def record(
        self,
        run_id: str,
        event: str,
        *,
        server: str | None = None,
        tool: str | None = None,
        detail: dict | None = None,
    ) -> None:
        payload = json.dumps(detail, default=str) if detail is not None else None
        stamp = datetime.now(UTC).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events (run_id, ts, event, server, tool, detail)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, stamp, event, server, tool, payload),
            )
            self._conn.commit()

    def list_events(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, ts, event, server, tool, detail
                FROM events
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        events = []
        for row in rows:
            detail = json.loads(row["detail"]) if row["detail"] else None
            events.append(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "event": row["event"],
                    "server": row["server"],
                    "tool": row["tool"],
                    "detail": detail,
                }
            )
        return events

    def close(self) -> None:
        with self._lock:
            self._conn.close()
