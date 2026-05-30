from agents.subagents import run_parallel_correction_subagents
from core.schemas import FailureReport


def test_parallel_subagents_select_push_recovery_patch() -> None:
    report = FailureReport(
        primary_failure="fails_on_push",
        secondary_failures=[],
        evidence={"push_recovery_success": 0.2},
        likely_causes=["push recovery is underrepresented"],
        suggested_research_directions=["add push curriculum"],
    )

    plan = run_parallel_correction_subagents(failure_report=report, metrics={})

    assert plan.primary_patch is not None
    assert "reward_weights.recovery" in plan.primary_patch.patch
    assert plan.scenario_candidates


def test_parallel_subagents_return_valid_candidate_list() -> None:
    report = FailureReport(
        primary_failure="foot_slip",
        secondary_failures=[],
        evidence={"foot_slip_events_per_meter": 2.0},
        likely_causes=["contact behavior is not robust"],
        suggested_research_directions=["increase foot slip penalty"],
    )

    plan = run_parallel_correction_subagents(failure_report=report, metrics={})

    assert any(candidate.agent_name == "contact_specialist" for candidate in plan.patch_candidates)
    assert all(not candidate.validation_errors for candidate in plan.patch_candidates)
