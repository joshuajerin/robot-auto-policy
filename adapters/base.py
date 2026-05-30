"""Task adapter protocol for generalized robot policy training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.schemas import FailureReport, ScenarioSpec, ScoreBreakdown, TaskSpec


@dataclass(frozen=True)
class Rollout:
    scenario_id: str
    video_path: str | None = None
    metrics_path: str | None = None
    frames_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentHistory:
    recent_experiments: list[dict[str, Any]] = field(default_factory=list)
    scenario_matrix: list[dict[str, Any]] = field(default_factory=list)
    failure_reports: list[dict[str, Any]] = field(default_factory=list)


class TaskAdapter(Protocol):
    task_family: str

    def default_task_spec(self) -> TaskSpec:
        ...

    def allowed_patch_paths(self) -> list[str]:
        ...

    def generate_scenarios(self, history: ExperimentHistory) -> list[ScenarioSpec]:
        ...

    def score(self, raw_metrics: dict[str, Any]) -> ScoreBreakdown:
        ...

    def diagnose(self, rollouts: list[Rollout], metrics: dict[str, Any]) -> FailureReport:
        ...

