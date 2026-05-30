from pathlib import Path

from core.autoresearch_loop import run_dry_research_loop
from tools.orchestration_smoke import summarize_research_db, validate_summary


def test_orchestration_smoke_validates_environment_progression(tmp_path) -> None:
    db_path = tmp_path / "research.db"

    run_dry_research_loop(Path.cwd(), db_path, experiments=10)
    summary = summarize_research_db(db_path)
    blockers = validate_summary(
        summary,
        min_experiments=6,
        min_scenarios=20,
        min_max_difficulty=0.75,
        min_parent_edges=8,
    )

    assert blockers == []
    assert summary["counts"]["experiments"] == 10
    assert summary["counts"]["scenario_parent_edges"] >= 8
    assert {"flat", "slope", "rough"}.issubset(set(summary["terrain_types"]))
    assert "learning_frontier" in summary["frontier_status_counts"]
    assert "too_hard" in summary["frontier_status_counts"]
