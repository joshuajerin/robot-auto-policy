from pathlib import Path

from core.patch_validator import apply_yaml_patch, validate_patch_spec
from core.schemas import PatchSpec


def test_valid_patch_dry_run_reads_old_values() -> None:
    patch = PatchSpec(
        experiment_name="push_recovery",
        hypothesis="Pushes need curriculum.",
        allowed_files=[
            "configs/locomotion/rewards.yaml",
            "configs/locomotion/domain_randomization.yaml",
        ],
        patch={
            "reward_weights.recovery": 0.35,
            "domain_randomization.push_impulse_probability": 0.05,
        },
        expected_effect="Improve recovery.",
        risk="May overfit to pushes.",
        rollback="Restore previous values.",
    )

    result = validate_patch_spec(patch)
    assert result.ok, result.errors
    changes = apply_yaml_patch(patch, Path.cwd(), dry_run=True)
    assert changes["configs/locomotion/rewards.yaml"]["reward_weights.recovery"]["old"] == 0.2


def test_locked_eval_path_is_rejected() -> None:
    patch = PatchSpec(
        experiment_name="bad_eval_edit",
        hypothesis="Cheat the evaluator.",
        allowed_files=["eval/fixed_eval_seeds.json"],
        patch={"reward_weights.recovery": 0.35},
        expected_effect="Invalid.",
        risk="Invalid.",
        rollback="Invalid.",
    )

    result = validate_patch_spec(patch)
    assert not result.ok
    assert any("unknown or locked" in error for error in result.errors)


def test_unsafe_torque_limit_is_rejected() -> None:
    patch = PatchSpec(
        experiment_name="unsafe_torque",
        hypothesis="Increase torque.",
        allowed_files=["configs/locomotion/actuators.yaml"],
        patch={"actuators.torque_limit_scale": 1.2},
        expected_effect="Invalid.",
        risk="Unsafe.",
        rollback="Restore torque limit.",
    )

    result = validate_patch_spec(patch)
    assert not result.ok
    assert any("outside safe range" in error for error in result.errors)

