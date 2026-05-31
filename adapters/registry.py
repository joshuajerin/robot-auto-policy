"""Task adapter registry."""

from __future__ import annotations

from adapters.base import TaskAdapter
from adapters.locomotion import LocomotionAdapter
from adapters.manipulation import ManipulationAdapter


_ADAPTERS: dict[str, type[TaskAdapter]] = {
    "locomotion": LocomotionAdapter,
    "manipulation": ManipulationAdapter,
}


def get_task_adapter(task_family: str) -> TaskAdapter:
    try:
        return _ADAPTERS[task_family]()
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unsupported task_family {task_family!r}; available: {available}") from exc


def available_task_families() -> list[str]:
    return sorted(_ADAPTERS)
