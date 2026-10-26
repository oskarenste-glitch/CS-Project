"""A tiny document database on top of SQLite.

Each collection (members, expenses, tasks, shopping, recipes, config) holds
JSON documents with an "id" field. The file wg.sqlite sits next to the app
(or wherever the environment variable WG_DB points).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid

DB_PATH = os.environ.get("WG_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "wg.sqlite"))
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("CREATE TABLE IF NOT EXISTS docs (coll TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (coll, id))")
    return c


def list_docs(coll: str) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT data FROM docs WHERE coll = ? ORDER BY rowid", (coll,)).fetchall()
    return [json.loads(r[0]) for r in rows]


def get(coll: str, id: str) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT data FROM docs WHERE coll = ? AND id = ?", (coll, id)).fetchone()
    return json.loads(row[0]) if row else None


def put(coll: str, doc: dict) -> dict:
    if not doc.get("id"):
        doc = {**doc, "id": new_id(coll[0])}
    with _lock, _conn() as c:
        c.execute("INSERT INTO docs (coll, id, data) VALUES (?, ?, ?) "
                  "ON CONFLICT(coll, id) DO UPDATE SET data = excluded.data",
                  (coll, doc["id"], json.dumps(doc, ensure_ascii=False)))
    return doc


def delete(coll: str, id: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM docs WHERE coll = ? AND id = ?", (coll, id))


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
