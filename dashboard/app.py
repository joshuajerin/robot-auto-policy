"""Streamlit dashboard for RoboGenesis research memory."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from dashboard.components.experiment_table import load_experiments
from dashboard.components.metric_charts import load_policies
from dashboard.components.scenario_tree import load_scenarios


def run_dashboard(db_path: Path) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install dashboard extras first: pip install -e '.[dashboard]'") from exc

    st.set_page_config(page_title="RoboGenesis", layout="wide")
    st.title("RoboGenesis AutoResearch")
    st.caption("Policy lineage, generated scenarios, patch history, and evaluation scores.")

    if not db_path.exists():
        st.warning(f"No research DB found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    policies = load_policies(conn)
    experiments = load_experiments(conn)
    scenarios = load_scenarios(conn)

    best = max(policies, key=lambda row: row.get("score", 0.0), default=None)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Policies", len(policies))
    col_b.metric("Experiments", len(experiments))
    col_c.metric("Scenarios", len(scenarios))

    if best:
        st.subheader("Current Best Policy")
        st.json(best)

    st.subheader("Experiment Leaderboard")
    st.dataframe(experiments, use_container_width=True)

    st.subheader("Scenario Tree")
    st.dataframe(scenarios, use_container_width=True)

    st.subheader("Policy Scores")
    st.dataframe(policies, use_container_width=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/research.db")
    args = parser.parse_args()
    run_dashboard(Path(args.db))


if __name__ == "__main__":
    main()

