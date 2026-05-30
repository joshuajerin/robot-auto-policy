"""SQLite research memory for experiment lineage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from core.schemas import FailureReport, PatchSpec, ScenarioSpec, ScoreBreakdown


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  parent_policy_id TEXT,
  patch_json TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  modal_job_id TEXT,
  score_before REAL,
  score_after REAL,
  accepted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS policies (
  policy_id TEXT PRIMARY KEY,
  parent_policy_id TEXT,
  checkpoint_path TEXT,
  score REAL NOT NULL,
  metrics_json TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  accepted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scenarios (
  scenario_id TEXT PRIMARY KEY,
  parent_scenario_id TEXT,
  task_family TEXT NOT NULL,
  scenario_spec_json TEXT NOT NULL,
  difficulty REAL NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scenario_evals (
  scenario_id TEXT NOT NULL,
  policy_id TEXT NOT NULL,
  success_rate REAL NOT NULL,
  score REAL NOT NULL,
  failure_modes_json TEXT NOT NULL,
  rollout_video_path TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (scenario_id, policy_id)
);

CREATE TABLE IF NOT EXISTS failure_reports (
  experiment_id TEXT NOT NULL,
  policy_id TEXT NOT NULL,
  failure_report_json TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class ExperimentDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init()

    def init(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def insert_experiment(
        self,
        experiment_id: str,
        parent_policy_id: str | None,
        patch: PatchSpec,
        status: str,
        score_before: float | None = None,
        score_after: float | None = None,
        accepted: bool = False,
        modal_job_id: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiments
            (id, parent_policy_id, patch_json, hypothesis, status, modal_job_id, score_before, score_after, accepted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                parent_policy_id,
                json.dumps(patch.to_dict(), sort_keys=True),
                patch.hypothesis,
                status,
                modal_job_id,
                score_before,
                score_after,
                int(accepted),
            ),
        )
        self.conn.commit()

    def insert_policy(
        self,
        policy_id: str,
        parent_policy_id: str | None,
        checkpoint_path: str | None,
        metrics: ScoreBreakdown,
        accepted: bool,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO policies
            (policy_id, parent_policy_id, checkpoint_path, score, metrics_json, accepted)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                policy_id,
                parent_policy_id,
                checkpoint_path,
                metrics.total_score,
                json.dumps(metrics.to_dict(), sort_keys=True),
                int(accepted),
            ),
        )
        self.conn.commit()

    def insert_scenarios(self, scenarios: Iterable[ScenarioSpec]) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO scenarios
            (scenario_id, parent_scenario_id, task_family, scenario_spec_json, difficulty, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    scenario.scenario_id,
                    scenario.parent_scenario_id,
                    scenario.task_family,
                    json.dumps(scenario.to_dict(), sort_keys=True),
                    scenario.difficulty,
                    scenario.status,
                )
                for scenario in scenarios
            ],
        )
        self.conn.commit()

    def insert_scenario_eval(
        self,
        scenario_id: str,
        policy_id: str,
        success_rate: float,
        score: float,
        failure_modes: list[str],
        rollout_video_path: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO scenario_evals
            (scenario_id, policy_id, success_rate, score, failure_modes_json, rollout_video_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (scenario_id, policy_id, success_rate, score, json.dumps(failure_modes), rollout_video_path),
        )
        self.conn.commit()

    def insert_failure_report(self, experiment_id: str, policy_id: str, report: FailureReport) -> None:
        self.conn.execute(
            """
            INSERT INTO failure_reports (experiment_id, policy_id, failure_report_json)
            VALUES (?, ?, ?)
            """,
            (experiment_id, policy_id, json.dumps(report.to_dict(), sort_keys=True)),
        )
        self.conn.commit()

    def recent_experiments(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def scenario_matrix(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT s.scenario_id, s.difficulty, s.status, e.policy_id, e.success_rate, e.score, e.failure_modes_json
            FROM scenarios s
            LEFT JOIN scenario_evals e ON s.scenario_id = e.scenario_id
            ORDER BY s.difficulty, s.scenario_id
            """
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent_failures(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM failure_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}

