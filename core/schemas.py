"""Typed contracts shared by agents, adapters, evaluators, and runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskFamily = Literal["locomotion", "manipulation", "navigation", "aerial", "locomanipulation"]
ScenarioStatus = Literal["candidate", "validated", "too_easy", "learning_frontier", "too_hard", "invalid"]
ExperimentStatus = Literal["proposed", "running", "accepted", "rejected", "failed"]


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    embodiment_type: str
    asset_path: str
    num_dofs: int
    controlled_joints: list[str]
    actuator_groups: dict[str, list[str]] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_family: TaskFamily
    objective: str
    base_env: str
    commands: dict[str, Any] = field(default_factory=dict)
    style_targets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    task_family: TaskFamily
    difficulty: float
    terrain: dict[str, Any] = field(default_factory=dict)
    disturbances: dict[str, Any] = field(default_factory=dict)
    robot_variation: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    status: ScenarioStatus = "candidate"
    parent_scenario_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchSpec:
    experiment_name: str
    hypothesis: str
    allowed_files: list[str]
    patch: dict[str, Any]
    expected_effect: str
    risk: str
    rollback: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreBreakdown:
    policy_id: str
    total_score: float
    command_tracking: float
    survival_no_fall: float
    stability: float
    generated_scenario_success: float
    gait_quality: float
    energy_efficiency: float
    smoothness: float
    recovery_from_disturbance: float
    safety_penalty: float = 0.0
    regression_penalty: float = 0.0
    safety_passed: bool = True
    base_success: float = 0.0
    eval_seed_count: int = 0
    reward_hacking_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureReport:
    primary_failure: str
    secondary_failures: list[str]
    evidence: dict[str, Any]
    likely_causes: list[str]
    suggested_research_directions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    policy_id: str
    status: ExperimentStatus
    metrics: ScoreBreakdown
    checkpoint_path: str | None = None
    rollout_video_path: str | None = None
    train_log_path: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

