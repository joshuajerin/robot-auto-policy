"""Small Streamlit dashboard for local H1 AutoResearch lineage."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "artifacts" / "research.db"


def load_dashboard_state(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "db_path": str(db_path),
            "experiments": [],
            "policies": [],
            "scenarios": [],
            "failures": [],
            "videos": _discover_videos(REPO_ROOT),
        }

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return {
            "db_path": str(db_path),
            "experiments": _fetch_rows(
                con,
                """
                SELECT id, status, accepted, score_before, score_after, modal_job_id, created_at
                FROM experiments
                ORDER BY created_at DESC, id DESC
                LIMIT 200
                """,
            ),
            "policies": _fetch_rows(
                con,
                """
                SELECT policy_id, parent_policy_id, score, accepted, checkpoint_path, created_at
                FROM policies
                ORDER BY created_at DESC, score DESC
                LIMIT 100
                """,
            ),
            "scenarios": _fetch_rows(
                con,
                """
                SELECT scenario_id, parent_scenario_id, difficulty, status, created_at
                FROM scenarios
                ORDER BY created_at DESC, scenario_id DESC
                LIMIT 200
                """,
            ),
            "failures": _failure_rows(con),
            "videos": _discover_videos(REPO_ROOT),
        }
    finally:
        con.close()


def _fetch_rows(con: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in con.execute(query)]


def _failure_rows(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = _fetch_rows(
        con,
        """
        SELECT experiment_id, policy_id, failure_report_json, created_at
        FROM failure_reports
        ORDER BY created_at DESC
        LIMIT 100
        """,
    )
    for row in rows:
        try:
            report = json.loads(str(row.pop("failure_report_json")))
        except json.JSONDecodeError:
            report = {}
        row["primary_failure"] = report.get("primary_failure", "")
        row["secondary_failures"] = ", ".join(report.get("secondary_failures", []) or [])
    return rows


def _discover_videos(root: Path) -> list[dict[str, str]]:
    videos: list[dict[str, str]] = []
    for search_root in (root / "docs" / "media", root / "artifacts"):
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*.mp4")):
            videos.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                }
            )
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args, _ = parser.parse_known_args()

    import streamlit as st

    db_path = Path(args.db)
    state = load_dashboard_state(db_path)

    st.set_page_config(page_title="H1 AutoResearch", layout="wide")
    st.title("H1 AutoResearch")
    st.caption(f"SQLite lineage: {state['db_path']}")

    experiments = state["experiments"]
    accepted = sum(1 for row in experiments if int(row.get("accepted") or 0))
    running = sum(1 for row in experiments if row.get("status") == "running")
    completed = sum(1 for row in experiments if row.get("status") in {"completed", "accepted", "rejected"})

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Experiments", len(experiments))
    col_b.metric("Accepted", accepted)
    col_c.metric("Running", running)
    col_d.metric("Reviewed", completed)

    tab_runs, tab_policies, tab_scenarios, tab_failures, tab_videos = st.tabs(
        ["Runs", "Policies", "Scenarios", "Failures", "Videos"]
    )
    with tab_runs:
        st.dataframe(experiments, use_container_width=True, hide_index=True)
    with tab_policies:
        st.dataframe(state["policies"], use_container_width=True, hide_index=True)
    with tab_scenarios:
        st.dataframe(state["scenarios"], use_container_width=True, hide_index=True)
    with tab_failures:
        st.dataframe(state["failures"], use_container_width=True, hide_index=True)
    with tab_videos:
        for video in state["videos"]:
            st.subheader(video["relative_path"])
            st.video(video["path"])


if __name__ == "__main__":
    main()
