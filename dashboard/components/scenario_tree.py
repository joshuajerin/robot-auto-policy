"""Scenario tree loader."""

from __future__ import annotations

import sqlite3
from typing import Any


def load_scenarios(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT scenario_id, parent_scenario_id, task_family, difficulty, status, created_at
        FROM scenarios
        ORDER BY difficulty, scenario_id
        """
    ).fetchall()
    columns = ["scenario_id", "parent_scenario_id", "task_family", "difficulty", "status", "created_at"]
    return [dict(zip(columns, row, strict=True)) for row in rows]

