import json
import sqlite3

from core.artifact_ingest import ingest_artifact_dir


def test_ingest_artifact_dir_records_policy_and_failure(tmp_path) -> None:
    artifact_dir = tmp_path / "baseline_h1_seed_1"
    artifact_dir.mkdir()
    (artifact_dir / "experiment_spec.json").write_text(
        json.dumps({"train": {"max_iterations": 1000}}, sort_keys=True)
    )
    (artifact_dir / "eval_metrics.json").write_text(
        json.dumps(
            {
                "policy_id": "baseline_h1_seed_1",
                "command_tracking": 0.5,
                "survival_no_fall": 0.4,
                "stability": 0.4,
                "generated_scenario_success": 0.0,
                "gait_quality": 0.3,
                "energy_efficiency": 0.5,
                "smoothness": 0.5,
                "recovery_from_disturbance": 0.0,
                "base_success": 0.4,
                "eval_seed_count": 8,
            },
            sort_keys=True,
        )
    )
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "baseline_h1_seed_1",
                "rollout_video_files": ["logs/rsl_rl/run/videos/rollout.mp4"],
            },
            sort_keys=True,
        )
    )
    (artifact_dir / "model_1000.pt").write_bytes(b"checkpoint")
    video = artifact_dir / "logs" / "rsl_rl" / "run" / "videos" / "rollout.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    db_path = tmp_path / "research.db"
    summary = ingest_artifact_dir(artifact_dir, db_path=db_path)

    assert summary["experiment_id"] == "baseline_h1_seed_1"
    assert summary["rollout_video_path"] == str(video)
    assert summary["rollout_video_paths"] == [str(video)]
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM failure_reports").fetchone()[0] == 1
    conn.close()
