from adapters.base import ExperimentHistory
from adapters.manipulation import ManipulationAdapter
from adapters.manipulation.failure_diagnosis import diagnose_manipulation_failure
from adapters.manipulation.metrics import score_manipulation
from adapters.manipulation.scenario_generator import generate_manipulation_scenarios
from agents.planner import propose_patch
from core.patch_validator import apply_yaml_patch, validate_patch_spec


def test_manipulation_adapter_exposes_task_surface() -> None:
    adapter = ManipulationAdapter()

    task = adapter.default_task_spec()

    assert task.task_family == "manipulation"
    assert "configs/manipulation/rewards.yaml" in adapter.allowed_patch_paths()


def test_manipulation_scenarios_include_3d_objects() -> None:
    scenarios = generate_manipulation_scenarios(ExperimentHistory())

    assert scenarios[0].task_family == "manipulation"
    assert scenarios[0].objects
    assert scenarios[0].workspace["table_asset"].endswith("lab_table.usda")
    assert scenarios[0].dataset["asset_manifest"] == "assets/manipulation_objects/manifest.json"


def test_manipulation_planner_patch_validates_and_reads_config() -> None:
    context = {
        "task_spec": ManipulationAdapter().default_task_spec().to_dict(),
        "failure_reports": [{"failure_report_json": '{"primary_failure":"object_slip"}'}],
    }

    patch = propose_patch(context)
    result = validate_patch_spec(patch)
    changes = apply_yaml_patch(patch, repo_root=Path.cwd(), dry_run=True)

    assert result.ok, result.errors
    assert patch.allowed_files[0].startswith("configs/manipulation/")
    assert changes["configs/manipulation/rewards.yaml"]["reward_weights.object_stability"]["old"] == 0.45


def test_manipulation_score_and_failure_diagnosis() -> None:
    metrics = {
        "policy_id": "manipulation_test",
        "task_success_rate": 0.4,
        "task_progress": 0.5,
        "grasp_success_rate": 0.7,
        "object_slip_rate": 0.5,
        "placement_error_m": 0.04,
        "collision_rate": 0.05,
        "eval_seed_count": 8,
    }

    score = score_manipulation(metrics)
    report = diagnose_manipulation_failure([], metrics)

    assert score.policy_id == "manipulation_test"
    assert score.total_score > 0.0
    assert report.primary_failure == "object_slip"
from pathlib import Path
