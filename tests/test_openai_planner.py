from agents.openai_planner import propose_patch_with_openai


def test_openai_planner_falls_back_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    patch = propose_patch_with_openai({"failure_reports": []})

    assert patch.experiment_name == "increase_command_tracking_baseline"
    assert patch.patch == {"reward_weights.command_tracking": 1.15}

