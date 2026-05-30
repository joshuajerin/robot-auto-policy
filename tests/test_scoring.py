from core.scoring import score_from_metrics, should_accept


def test_accepts_clear_improvement_without_regression() -> None:
    old = score_from_metrics(
        "old",
        {
            "command_tracking": 0.5,
            "survival_no_fall": 0.6,
            "stability": 0.5,
            "generated_scenario_success": 0.3,
            "gait_quality": 0.4,
            "energy_efficiency": 0.5,
            "smoothness": 0.4,
            "recovery_from_disturbance": 0.3,
            "base_success": 0.6,
            "eval_seed_count": 8,
        },
    )
    new = score_from_metrics(
        "new",
        {
            "command_tracking": 0.6,
            "survival_no_fall": 0.7,
            "stability": 0.6,
            "generated_scenario_success": 0.5,
            "gait_quality": 0.5,
            "energy_efficiency": 0.55,
            "smoothness": 0.5,
            "recovery_from_disturbance": 0.55,
            "base_success": 0.62,
            "eval_seed_count": 8,
        },
    )

    assert should_accept(old, new)


def test_rejects_base_task_regression() -> None:
    old = score_from_metrics("old", {"survival_no_fall": 0.8, "base_success": 0.8, "eval_seed_count": 8})
    new = score_from_metrics(
        "new",
        {
            "command_tracking": 1.0,
            "survival_no_fall": 0.9,
            "stability": 1.0,
            "generated_scenario_success": 1.0,
            "gait_quality": 1.0,
            "energy_efficiency": 1.0,
            "smoothness": 1.0,
            "recovery_from_disturbance": 1.0,
            "base_success": 0.7,
            "eval_seed_count": 8,
        },
    )

    assert not should_accept(old, new)

