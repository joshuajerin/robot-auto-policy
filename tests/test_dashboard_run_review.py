import json
import runpy
import sqlite3
from pathlib import Path

from core.experiment_db import ExperimentDB
from core.schemas import FailureReport, PatchSpec
from dashboard.app import _default_review_index, _policy_reference_count, _run_label, _system_graph_dot
from dashboard.components.run_review import build_change_rows, load_run_reviews


def test_dashboard_app_imports_from_outside_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    runpy.run_path(str(app_path), run_name="dashboard_app_import_check")


def test_dashboard_run_selector_label_stays_compact() -> None:
    assert (
        _run_label(
            {
                "experiment_id": "autoresearch_h1_motor_rough_20260531_000911_seed-2400",
                "status": "running",
                "score_delta": None,
                "run_videos": [],
            }
        )
        == "autoresearch_h1_motor_rough_20260531_000911_seed-2400"
    )


def test_policy_reference_count_includes_parent_policy_refs() -> None:
    assert _policy_reference_count([], [{"parent_policy_id": "parent_policy_001"}]) == 1


def test_system_graph_dot_summarizes_autoresearch_loop() -> None:
    dot = _system_graph_dot(
        {
            "experiment_id": "autoresearch_h1_motor_rough_20260531_000911_seed-2400",
            "status": "running",
            "task": "Isaac-Velocity-Rough-H1-v0",
            "train": {"seed": 2400, "num_envs": 4096, "max_iterations": 300},
            "rationale": {"primary_failure": "oscillatory_actions"},
            "changed_sections": ["Actuators", "Rewards", "Terrain"],
            "change_rows": [{"parameter": "reward_weights.smoothness"}],
            "generated_scenarios": [{"scenario_id": "rough_heightfield_walk_v001"}],
            "run_videos": [
                {"exists": True, "path": "diagonal-policy-step-0.mp4"},
                {"exists": True, "path": "front-policy-step-0.mp4"},
                {"exists": True, "path": "side-policy-step-0.mp4"},
            ],
            "source_videos": [{"exists": True, "path": "source.mp4"}],
        }
    )

    assert "LLM planner\\nhypothesis + risk" in dot
    assert "Camera render\\n3 videos" in dot
    assert "Scenario frontier\\n1 tasks" in dot
    assert "oscillatory_actions" in dot
    assert "telemetry" not in dot.lower()


def test_load_run_reviews_merges_db_spec_and_videos(tmp_path) -> None:
    experiment_id = "autoresearch_h1_test_20260531_010000_seed-1"
    db_path = tmp_path / "research.db"
    db = ExperimentDB(db_path)
    patch = PatchSpec(
        experiment_name="add_push_recovery_curriculum",
        hypothesis="The policy falls after side pushes because recovery is underrepresented.",
        allowed_files=[
            "configs/locomotion/rewards.yaml",
            "configs/locomotion/domain_randomization.yaml",
        ],
        patch={
            "reward_weights.recovery": 0.35,
            "domain_randomization.push_impulse_probability": 0.05,
        },
        expected_effect="Better recovery from lateral disturbances without base-task regression.",
        risk="May overfit to pushes.",
        rollback="Restore recovery and push values.",
    )
    report = FailureReport(
        primary_failure="fails_on_push",
        secondary_failures=["fall_risk"],
        evidence={"aggregate": {"mean_action_jerk": 0.18}},
        likely_causes=["push recovery is underrepresented"],
        suggested_research_directions=["increase recovery reward"],
    )
    db.insert_experiment(
        experiment_id=experiment_id,
        parent_policy_id="baseline_0000",
        patch=patch,
        status="running",
        score_before=0.5,
    )
    db.insert_failure_report(experiment_id, "baseline_0000", report)
    db.close()

    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "modal_downloads" / experiment_id
    artifact_dir.mkdir(parents=True)
    telemetry_video = artifact_dir / "rollout_telemetry.mp4"
    telemetry_video.write_bytes(b"telemetry-video")
    run_video = artifact_dir / "camera_render.mp4"
    run_video.write_bytes(b"render-video")
    source_video = artifact_root / "multiview" / "side" / "side-policy-step-0.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"source-video")
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps({"rollout_video_files": ["rollout_telemetry.mp4", "camera_render.mp4"]}, sort_keys=True)
    )
    spec_path = artifact_root / "autoresearch_specs" / f"{experiment_id}.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "task": "Isaac-Velocity-Flat-H1-v0",
                "train": {"seed": 1, "num_envs": 512, "max_iterations": 10},
                "render": {"mode": "telemetry", "video_length": 60, "fps": 20},
                "autoresearch": {
                    "config_changes": {
                        "configs/locomotion/rewards.yaml": {
                            "reward_weights.recovery": {"old": 0.2, "new": 0.35}
                        }
                    },
                    "generated_scenarios": [
                        {
                            "scenario_id": "side_push_recovery_v001",
                            "difficulty": 0.38,
                            "status": "candidate",
                            "terrain": {"type": "flat"},
                            "disturbances": {"push_impulse_probability": 0.05},
                        }
                    ],
                    "quick_iteration": {"seed": 1, "num_envs": 512, "max_iterations": 10},
                    "multiview_context": {
                        "views": [{"view": "side", "local_video_path": str(source_video)}]
                    },
                },
            },
            sort_keys=True,
        )
    )

    conn = sqlite3.connect(db_path)
    reviews = load_run_reviews(conn, repo_root=tmp_path, artifact_root=artifact_root)
    conn.close()

    assert len(reviews) == 1
    review = reviews[0]
    assert review["experiment_id"] == experiment_id
    assert review["task"] == "Isaac-Velocity-Flat-H1-v0"
    assert review["rationale"]["primary_failure"] == "fails_on_push"
    assert review["rationale"]["hypothesis"] == patch.hypothesis
    assert review["changed_sections"] == ["Domain Randomization", "Rewards"]
    assert review["change_rows"][0]["parameter"] == "reward_weights.recovery"
    assert review["change_rows"][0]["old_value"] == 0.2
    assert review["change_rows"][0]["new_value"] == 0.35
    assert review["run_videos"][0]["path"] == str(run_video.resolve())
    assert review["source_videos"][0]["path"] == str(source_video.resolve())
    assert review["generated_scenarios"][0]["scenario_id"] == "side_push_recovery_v001"


def test_build_change_rows_falls_back_to_patch_values() -> None:
    rows = build_change_rows(
        {},
        {
            "allowed_files": ["configs/locomotion/ppo.yaml"],
            "patch": {"ppo.max_iterations": 300},
        },
    )

    assert rows == [
        {
            "section": "PPO",
            "parameter": "ppo.max_iterations",
            "old": "not set",
            "new": "300",
            "old_value": None,
            "new_value": 300,
            "file": "configs/locomotion/ppo.yaml",
        }
    ]


def test_load_run_reviews_reads_raindrop_video_manifest(tmp_path) -> None:
    experiment_id = "autoresearch_h1_video_20260531_010000_seed-7"
    db_path = tmp_path / "research.db"
    db = ExperimentDB(db_path)
    patch = PatchSpec(
        experiment_name="render_manifest_smoke",
        hypothesis="A synced Modal render should be visible in the dashboard.",
        allowed_files=["configs/locomotion/render.yaml"],
        patch={"render.video_length": 120},
        expected_effect="Dashboard shows the rollout video.",
        risk="None.",
        rollback="Restore previous render settings.",
    )
    db.insert_experiment(
        experiment_id=experiment_id,
        parent_policy_id="baseline_0000",
        patch=patch,
        status="complete",
        score_before=0.1,
        score_after=0.2,
    )
    db.close()

    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "modal_downloads" / experiment_id
    video = artifact_dir / "logs" / "rsl_rl" / "run" / "rollout.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    remote_video = f"/runs/experiments/{experiment_id}/logs/rsl_rl/run/rollout.mp4"
    (artifact_dir / "raindrop_trace.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "primary_video_path": remote_video,
                "tasks": [
                    {
                        "name": "render_isaac_camera_video",
                        "output": {"actual_video_paths": [remote_video]},
                    }
                ],
                "video_paths": [remote_video],
            },
            sort_keys=True,
        )
    )

    conn = sqlite3.connect(db_path)
    reviews = load_run_reviews(conn, repo_root=tmp_path, artifact_root=artifact_root)
    conn.close()

    assert len(reviews) == 1
    assert reviews[0]["run_videos"] == [
        {
            "kind": "run",
            "path": str(video.resolve()),
            "label": "Run render",
            "exists": True,
        }
    ]
    assert reviews[0]["artifact_paths"]["video_manifests"]["raindrop_trace.json"] == [
        str(artifact_dir / "raindrop_trace.json")
    ]


def test_run_review_defaults_to_first_local_video() -> None:
    reviews = [
        {"experiment_id": "latest", "status": "running", "score_delta": None, "run_videos": []},
        {
            "experiment_id": "with-missing-reference",
            "status": "complete",
            "score_delta": 0,
            "run_videos": [{"exists": False}],
        },
        {
            "experiment_id": "with-video",
            "status": "complete",
            "score_delta": 0.1,
            "run_videos": [{"exists": True}],
        },
    ]

    assert _default_review_index(reviews) == 2
    assert _run_label(reviews[2]) == "with-video"
