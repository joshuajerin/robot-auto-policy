"""Policy metric loader."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def load_policies(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT policy_id, parent_policy_id, checkpoint_path, score, metrics_json, accepted, created_at
        FROM policies
        ORDER BY score DESC
        """
    ).fetchall()
    policies: list[dict[str, Any]] = []
    for row in rows:
        metrics = json.loads(row[4])
        policies.append(
            {
                "policy_id": row[0],
                "parent_policy_id": row[1],
                "checkpoint_path": row[2],
                "score": row[3],
                "accepted": bool(row[5]),
                "created_at": row[6],
                "command_tracking": metrics.get("command_tracking"),
                "survival_no_fall": metrics.get("survival_no_fall"),
                "generated_scenario_success": metrics.get("generated_scenario_success"),
            }
        )
    return policies

