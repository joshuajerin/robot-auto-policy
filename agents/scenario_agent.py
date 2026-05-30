"""Scenario agent wrapper."""

from __future__ import annotations

from adapters.base import ExperimentHistory, TaskAdapter
from core.schemas import ScenarioSpec


def generate_scenarios(adapter: TaskAdapter, history: ExperimentHistory) -> list[ScenarioSpec]:
    return adapter.generate_scenarios(history)

