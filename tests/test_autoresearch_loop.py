from pathlib import Path

from core.autoresearch_loop import run_dry_research_loop


def test_dry_research_loop_records_experiments(tmp_path) -> None:
    summaries = run_dry_research_loop(
        repo_root=Path.cwd(),
        db_path=tmp_path / "research.db",
        experiments=2,
    )

    assert len(summaries) == 2
    assert summaries[0]["experiment_id"] == "exp_0001"
    assert "score_before" in summaries[0]
    assert "patch" in summaries[0]
