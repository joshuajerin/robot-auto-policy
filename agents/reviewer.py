"""Reviewer checks suspicious improvements before policy promotion."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.schemas import ScoreBreakdown
from core.scoring import should_accept


@dataclass(frozen=True)
class ReviewDecision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def review_policy_candidate(old: ScoreBreakdown, new: ScoreBreakdown) -> ReviewDecision:
    reasons: list[str] = []
    if new.total_score < old.total_score + 0.03:
        reasons.append("total score did not improve by at least 0.03")
    if not new.safety_passed:
        reasons.append("safety checks failed")
    if new.base_success < old.base_success - 0.05:
        reasons.append("base task regressed by more than 5%")
    if new.generated_scenario_success < old.generated_scenario_success:
        reasons.append("generated scenario success regressed")
    if new.eval_seed_count < 8:
        reasons.append("not enough held-out evaluation seeds")
    if new.reward_hacking_detected:
        reasons.append("reward hacking was detected")

    accepted = should_accept(old, new)
    return ReviewDecision(accepted=accepted, reasons=[] if accepted else reasons)

