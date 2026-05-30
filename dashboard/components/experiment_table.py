"""Experiment table loader."""

from __future__ import annotations

import sqlite3
from typing import Any


def load_experiments(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, parent_policy_id, status, score_before, score_after, accepted, created_at
        FROM experiments
        ORDER BY rowid DESC
        """
    ).fetchall()
    columns = ["id", "parent_policy_id", "status", "score_before", "score_after", "accepted", "created_at"]
    return [dict(zip(columns, row, strict=True)) for row in rows]

