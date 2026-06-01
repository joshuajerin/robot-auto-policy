"""Smoke-test the RoboGenesis AutoResearch orchestration loop.

This tool is intentionally lightweight: it can run the deterministic local
AutoResearch loop and inspect the resulting SQLite research DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.locomotion.scenario_generator import classify_frontier
from core.autoresearch_loop import run_dry_research_loop


def summarize_research_db(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        scenario_rows = list(
            con.execute(
                """
                SELECT scenario_id, parent_scenario_id, difficulty, status, scenario_spec_json
                FROM scenarios
                ORDER BY scenario_id
                """
            )
        )
        eval_rows = list(
            con.execute(
                """
                SELECT scenario_id, success_rate, score, failure_modes_json
                FROM scenario_evals
                ORDER BY scenario_id
                """
            )
        )
        experiment_rows = list(
            con.execute(
                """
                SELECT id, accepted, score_before, score_after
                FROM experiments
                ORDER BY id
                """
            )
        )
    finally:
        con.close()

    difficulties = [float(row["difficulty"]) for row in scenario_rows if row["difficulty"] is not None]
    terrain_types: set[str] = set()
    scenario_kinds: set[str] = set()
    parent_edges = 0
    for row in scenario_rows:
        spec = json.loads(row["scenario_spec_json"])
        terrain_type = spec.get("terrain", {}).get("type")
        if terrain_type:
            terrain_types.add(str(terrain_type))
        scenario_kinds.add(_scenario_kind(str(row["scenario_id"])))
        if row["parent_scenario_id"]:
            parent_edges += 1

    frontier_counts: dict[str, int] = {}
    for row in eval_rows:
        status = classify_frontier(float(row["success_rate"]))
        frontier_counts[status] = frontier_counts.get(status, 0) + 1

    return {
        "db_path": str(db_path),
        "counts": {
            "experiments": len(experiment_rows),
            "scenarios": len(scenario_rows),
            "scenario_evals": len(eval_rows),
            "accepted_experiments": sum(1 for row in experiment_rows if int(row["accepted"] or 0)),
            "rejected_experiments": sum(1 for row in experiment_rows if not int(row["accepted"] or 0)),
            "scenario_parent_edges": parent_edges,
        },
        "difficulty": {
            "min": min(difficulties) if difficulties else None,
            "max": max(difficulties) if difficulties else None,
        },
        "terrain_types": sorted(terrain_types),
        "scenario_kinds": sorted(scenario_kinds),
        "frontier_status_counts": dict(sorted(frontier_counts.items())),
        "accepted_experiment_ids": [
            str(row["id"]) for row in experiment_rows if int(row["accepted"] or 0)
        ],
        "latest_scenarios": _latest_scenarios(scenario_rows, limit=8),
    }


def validate_summary(
    summary: dict[str, Any],
    *,
    min_experiments: int,
    min_scenarios: int,
    min_max_difficulty: float,
    min_parent_edges: int,
) -> list[str]:
    blockers: list[str] = []
    counts = summary["counts"]
    difficulty = summary["difficulty"]

    if counts["experiments"] < min_experiments:
        blockers.append(f"expected at least {min_experiments} experiments")
    if counts["scenarios"] < min_scenarios:
        blockers.append(f"expected at least {min_scenarios} generated scenarios")
    if counts["scenario_evals"] < min_scenarios:
        blockers.append(f"expected at least {min_scenarios} scenario eval rows")
    if counts["accepted_experiments"] < 1:
        blockers.append("expected at least one accepted experiment")
    if counts["rejected_experiments"] < 1:
        blockers.append("expected at least one rejected experiment")
    if counts["scenario_parent_edges"] < min_parent_edges:
        blockers.append(f"expected at least {min_parent_edges} parent-child scenario edges")
    if difficulty["max"] is None or float(difficulty["max"]) < min_max_difficulty:
        blockers.append(f"expected max difficulty >= {min_max_difficulty}")

    terrain_types = set(summary["terrain_types"])
    for required in ("flat", "slope", "rough"):
        if required not in terrain_types:
            blockers.append(f"missing terrain type: {required}")

    frontier_statuses = set(summary["frontier_status_counts"])
    if "learning_frontier" not in frontier_statuses:
        blockers.append("missing learning_frontier scenario evaluations")
    if "too_hard" not in frontier_statuses:
        blockers.append("missing too_hard scenario evaluations")

    return blockers


def _latest_scenarios(rows: list[sqlite3.Row], *, limit: int) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for row in rows[-limit:]:
        spec = json.loads(row["scenario_spec_json"])
        latest.append(
            {
                "scenario_id": row["scenario_id"],
                "parent_scenario_id": row["parent_scenario_id"],
                "difficulty": row["difficulty"],
                "terrain": spec.get("terrain", {}),
                "disturbances": spec.get("disturbances", {}),
                "robot_variation": spec.get("robot_variation", {}),
            }
        )
    return latest


def _scenario_kind(scenario_id: str) -> str:
    head, _, tail = scenario_id.rpartition("_v")
    if head and tail.isdigit():
        return head
    return scenario_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/orchestration_smoke.db")
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--run-dry-loop", action="store_true")
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--fail-on-blockers", action="store_true")
    parser.add_argument("--min-experiments", type=int, default=6)
    parser.add_argument("--min-scenarios", type=int, default=20)
    parser.add_argument("--min-max-difficulty", type=float, default=0.75)
    parser.add_argument("--min-parent-edges", type=int, default=8)
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.run_dry_loop:
        if db_path.exists():
            if not args.reset_db:
                raise SystemExit(f"{db_path} exists; pass --reset-db to overwrite it")
            db_path.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        run_dry_research_loop(REPO_ROOT, db_path, args.experiments)

    summary = summarize_research_db(db_path)
    blockers = validate_summary(
        summary,
        min_experiments=args.min_experiments,
        min_scenarios=args.min_scenarios,
        min_max_difficulty=args.min_max_difficulty,
        min_parent_edges=args.min_parent_edges,
    )
    payload = {
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "autoresearch": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.fail_on_blockers and blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
