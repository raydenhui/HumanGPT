"""SQLite persistence layer.

Thin wrapper around the stdlib ``sqlite3`` module. Schema is created and
migrated on boot; all writes go through a single connection with a write
lock so operator actions and endpoint releases never interleave badly.
Deliberately no ORM: this tool's data model is three small tables, and raw
SQL keeps the surface auditable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id             TEXT PRIMARY KEY,
    endpoint       TEXT NOT NULL,
    model          TEXT NOT NULL,
    body           TEXT NOT NULL,
    parsed         TEXT NOT NULL,
    state          TEXT NOT NULL,
    created_at     REAL NOT NULL,
    answered_at    REAL,
    answered_by    TEXT,
    claimed_by     TEXT,
    claimed_at     REAL,
    interrupted_at REAL,
    stream_mode    TEXT NOT NULL DEFAULT 'word-chunk',
    timeout_at     REAL
);

CREATE TABLE IF NOT EXISTS responses (
    id             TEXT PRIMARY KEY,
    request_id     TEXT NOT NULL REFERENCES requests(id),
    body           TEXT NOT NULL,
    finish_reason  TEXT NOT NULL,
    is_tool_call   INTEGER NOT NULL DEFAULT 0,
    tool_name      TEXT,
    tool_arguments TEXT,
    stream_mode    TEXT NOT NULL DEFAULT 'word-chunk',
    created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS templates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    body       TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models_meta (
    model_id    TEXT PRIMARY KEY,
    description TEXT,
    pricing     TEXT,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_state ON requests(state);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_responses_request ON responses(request_id);
"""


@dataclass(frozen=True, slots=True)
class RequestRow:
    id: str
    endpoint: str
    model: str
    body: str
    parsed: str
    state: str
    created_at: float
    answered_at: float | None
    answered_by: str | None
    claimed_by: str | None
    claimed_at: float | None
    interrupted_at: float | None
    stream_mode: str
    timeout_at: float | None

    @property
    def body_json(self) -> dict[str, Any]:
        return json.loads(self.body)

    @property
    def parsed_json(self) -> dict[str, Any]:
        return json.loads(self.parsed)


@dataclass(frozen=True, slots=True)
class ResponseRow:
    id: str
    request_id: str
    body: str
    finish_reason: str
    is_tool_call: bool
    tool_name: str | None
    tool_arguments: str | None
    stream_mode: str
    created_at: float

    @property
    def body_json(self) -> dict[str, Any]:
        return json.loads(self.body)


@dataclass(frozen=True, slots=True)
class TemplateRow:
    id: int
    name: str
    body: str
    created_at: float
    updated_at: float


def _rows_to_dicts(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


class Database:
    """Owns the SQLite connection and all SQL. Single app instance, single worker."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        sqlite3.enable_callback_tracebacks(True)
        self._write_lock = __import__("threading").Lock()
        self._execute_script(SCHEMA_SQL)

    # ------------------------------------------------------------------ core

    def _execute_script(self, script: str) -> None:
        with self._write_lock:
            self._conn.executescript(script)

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            return self._conn.execute(sql, params)

    def _executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> None:
        with self._write_lock:
            self._conn.executemany(sql, seq)

    def commit(self) -> None:
        with self._write_lock:
            self._conn.commit()

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._execute("BEGIN")
        try:
            yield
        except BaseException:
            self._execute("ROLLBACK")
            raise
        else:
            self._execute("COMMIT")

    # ------------------------------------------------------------ requests

    def create_request(self, row: RequestRow) -> None:
        self._execute(
            """INSERT INTO requests
                 (id, endpoint, model, body, parsed, state, created_at,
                  answered_at, answered_by, claimed_by, claimed_at,
                  interrupted_at, stream_mode, timeout_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.id,
                row.endpoint,
                row.model,
                row.body,
                row.parsed,
                row.state,
                row.created_at,
                row.answered_at,
                row.answered_by,
                row.claimed_by,
                row.claimed_at,
                row.interrupted_at,
                row.stream_mode,
                row.timeout_at,
            ),
        )
        self.commit()

    def count_by_state(self, states: Sequence[str] | None = None) -> dict[str, int]:
        """Row counts per state (single aggregated query, for the UI badges)."""
        sql = "SELECT state, COUNT(*) AS n FROM requests"
        params: list[Any] = []
        if states:
            sql += " WHERE state IN ({})".format(",".join("?" * len(states)))
            params.extend(states)
        sql += " GROUP BY state"
        cur = self._execute(sql, params)
        return {r["state"]: r["n"] for r in cur.fetchall()}

    def get_request(self, request_id: str) -> RequestRow | None:
        cur = self._execute("SELECT * FROM requests WHERE id = ?", (request_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._to_request(row)

    def list_requests(
        self,
        states: Sequence[str] | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[RequestRow]:
        sql = "SELECT * FROM requests"
        params: list[Any] = []
        if states:
            sql += " WHERE state IN ({})".format(",".join("?" * len(states)))
            params.extend(states)
        sql += " ORDER BY created_at ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur = self._execute(sql, params)
        return [self._to_request(r) for r in cur.fetchall()]

    def update_request_state(
        self,
        request_id: str,
        state: str,
        *,
        answered_at: float | None = None,
        answered_by: str | None = None,
        interrupted_at: float | None = None,
        only_if_pending: bool = False,
    ) -> bool:
        """Transition state; returns True when a row changed (regardless of prior state).

        ``only_if_pending`` adds ``AND state='pending'`` so the atomic
        pending→answered race (two operators submitting simultaneously) is
        decided by the database: exactly one wins.
        """
        where = "WHERE id = ?" if not only_if_pending else "WHERE id = ? AND state = 'pending'"
        cur = self._execute(
            f"""UPDATE requests SET state = ?,
                 answered_at = COALESCE(?, answered_at),
                 answered_by = COALESCE(?, answered_by),
                 interrupted_at = COALESCE(?, interrupted_at)
               {where}""",
            (state, answered_at, answered_by, interrupted_at, request_id),
        )
        self.commit()
        return cur.rowcount > 0

    def claim_request(self, request_id: str, operator: str) -> bool:
        """Soft claim: no lock, anyone may steal. Returns True if claimed."""
        now = time.time()
        cur = self._execute(
            "UPDATE requests SET claimed_by = ?, claimed_at = ? WHERE id = ? AND state = 'pending'",
            (operator, now, request_id),
        )
        self.commit()
        return cur.rowcount > 0

    def unclaim_request(self, request_id: str, operator: str) -> bool:
        cur = self._execute(
            "UPDATE requests SET claimed_by = NULL, claimed_at = NULL WHERE id = ? AND claimed_by = ?",
            (request_id, operator),
        )
        self.commit()
        return cur.rowcount > 0

    def mark_stale_pending_interrupted(self, before: float) -> int:
        """Startup sweep: any request left 'pending' (in-flight at a crash) becomes 'interrupted'."""
        cur = self._execute(
            "UPDATE requests SET state = 'interrupted', interrupted_at = ? "
            "WHERE state = 'pending' AND created_at < ?",
            (time.time(), before),
        )
        self.commit()
        return cur.rowcount

    def discard_request(self, request_id: str) -> bool:
        cur = self._execute(
            "UPDATE requests SET state = 'discarded' WHERE id = ? AND state = 'pending'",
            (request_id,),
        )
        self.commit()
        return cur.rowcount > 0

    # ----------------------------------------------------------- responses

    def create_response(self, row: ResponseRow) -> None:
        self._execute(
            """INSERT INTO responses
                 (id, request_id, body, finish_reason, is_tool_call,
                  tool_name, tool_arguments, stream_mode, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                row.id,
                row.request_id,
                row.body,
                row.finish_reason,
                int(row.is_tool_call),
                row.tool_name,
                row.tool_arguments,
                row.stream_mode,
                row.created_at,
            ),
        )
        self.commit()

    def get_response(self, response_id: str) -> ResponseRow | None:
        cur = self._execute("SELECT * FROM responses WHERE id = ?", (response_id,))
        row = cur.fetchone()
        return self._to_response(row) if row else None

    def get_response_for_request(self, request_id: str) -> ResponseRow | None:
        cur = self._execute("SELECT * FROM responses WHERE request_id = ? LIMIT 1", (request_id,))
        row = cur.fetchone()
        return self._to_response(row) if row else None

    # ----------------------------------------------------------- templates

    def create_template(self, name: str, body: str) -> TemplateRow:
        now = time.time()
        self._execute(
            "INSERT INTO templates (name, body, created_at, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET body = excluded.body, updated_at = excluded.updated_at",
            (name, body, now, now),
        )
        self.commit()
        return self.get_template(name)  # type: ignore[return-value]

    def get_template(self, name: str) -> TemplateRow | None:
        cur = self._execute("SELECT * FROM templates WHERE name = ?", (name,))
        row = cur.fetchone()
        return self._to_template(row) if row else None

    def list_templates(self) -> list[TemplateRow]:
        cur = self._execute("SELECT * FROM templates ORDER BY name ASC")
        return [self._to_template(r) for r in cur.fetchall()]

    def delete_template(self, name: str) -> bool:
        cur = self._execute("DELETE FROM templates WHERE name = ?", (name,))
        self.commit()
        return cur.rowcount > 0

    # ------------------------------------------------- models_meta + settings

    def upsert_model_meta(self, model_id: str, description: str | None, pricing: str | None) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO models_meta (model_id, description, pricing, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET description = excluded.description, "
            "pricing = excluded.pricing, updated_at = excluded.updated_at",
            (model_id, description, pricing, now),
        )
        self.commit()

    def delete_model_meta(self, model_id: str) -> bool:
        cur = self._execute("DELETE FROM models_meta WHERE model_id = ?", (model_id,))
        self.commit()
        return cur.rowcount > 0

    def list_models_meta(self) -> list[dict[str, Any]]:
        cur = self._execute("SELECT * FROM models_meta")
        out = []
        for r in cur.fetchall():
            out.append(
                {
                    "model_id": r["model_id"],
                    "description": r["description"],
                    "pricing": r["pricing"],
                    "updated_at": r["updated_at"],
                }
            )
        return out

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        cur = self._execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, now),
        )
        self.commit()

    def delete_setting(self, key: str) -> None:
        self._execute("DELETE FROM settings WHERE key = ?", (key,))
        self.commit()

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_request(row: sqlite3.Row) -> RequestRow:
        return RequestRow(
            id=row["id"],
            endpoint=row["endpoint"],
            model=row["model"],
            body=row["body"],
            parsed=row["parsed"],
            state=row["state"],
            created_at=row["created_at"],
            answered_at=row["answered_at"],
            answered_by=row["answered_by"],
            claimed_by=row["claimed_by"],
            claimed_at=row["claimed_at"],
            interrupted_at=row["interrupted_at"],
            stream_mode=row["stream_mode"],
            timeout_at=row["timeout_at"],
        )

    @staticmethod
    def _to_response(row: sqlite3.Row) -> ResponseRow:
        return ResponseRow(
            id=row["id"],
            request_id=row["request_id"],
            body=row["body"],
            finish_reason=row["finish_reason"],
            is_tool_call=bool(row["is_tool_call"]),
            tool_name=row["tool_name"],
            tool_arguments=row["tool_arguments"],
            stream_mode=row["stream_mode"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_template(row: sqlite3.Row) -> TemplateRow:
        return TemplateRow(
            id=row["id"],
            name=row["name"],
            body=row["body"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
