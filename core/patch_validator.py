"""Deterministic PatchSpec validation and YAML application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.schemas import PatchSpec


ALLOWED_PATCH_PATHS = {
    "configs/locomotion/rewards.yaml",
    "configs/locomotion/curriculum.yaml",
    "configs/locomotion/domain_randomization.yaml",
    "configs/locomotion/actuators.yaml",
    "configs/locomotion/ppo.yaml",
    "configs/locomotion/terrain.yaml",
}

LOCKED_PREFIXES = ("eval/", "modal_runner/", "artifacts/", "specs/")

PATCH_ROOT_TO_FILE = {
    "reward_weights": "configs/locomotion/rewards.yaml",
    "curriculum": "configs/locomotion/curriculum.yaml",
    "domain_randomization": "configs/locomotion/domain_randomization.yaml",
    "actuators": "configs/locomotion/actuators.yaml",
    "ppo": "configs/locomotion/ppo.yaml",
    "terrain": "configs/locomotion/terrain.yaml",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def validate_patch_spec(patch_spec: PatchSpec) -> ValidationResult:
    errors: list[str] = []

    requested_files = set(patch_spec.allowed_files)
    unknown_files = requested_files - ALLOWED_PATCH_PATHS
    if unknown_files:
        errors.append(f"unknown or locked allowed_files: {sorted(unknown_files)}")

    for path in requested_files:
        if path.startswith(LOCKED_PREFIXES) or ".." in Path(path).parts:
            errors.append(f"locked or unsafe path requested: {path}")

    if not patch_spec.patch:
        errors.append("patch must contain at least one parameter change")

    for key, value in patch_spec.patch.items():
        key_errors = _validate_patch_key_value(key, value, requested_files)
        errors.extend(key_errors)

    return ValidationResult(ok=not errors, errors=errors)


def target_file_for_key(key: str) -> str:
    root = key.split(".", 1)[0]
    if root not in PATCH_ROOT_TO_FILE:
        raise ValueError(f"unsupported patch root: {root}")
    return PATCH_ROOT_TO_FILE[root]


def apply_yaml_patch(patch_spec: PatchSpec, repo_root: Path, dry_run: bool = False) -> dict[str, dict[str, Any]]:
    """Apply a validated dotted-key patch to editable YAML configs."""

    validate_patch_spec(patch_spec).raise_for_errors()
    changes: dict[str, dict[str, Any]] = {}

    grouped: dict[str, dict[str, Any]] = {}
    for key, value in patch_spec.patch.items():
        grouped.setdefault(target_file_for_key(key), {})[key] = value

    for rel_path, file_patch in grouped.items():
        path = repo_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"config file does not exist: {rel_path}")
        data = yaml.safe_load(path.read_text()) or {}
        file_changes: dict[str, Any] = {}
        for dotted_key, new_value in file_patch.items():
            old_value = _set_dotted_value(data, dotted_key, new_value)
            file_changes[dotted_key] = {"old": old_value, "new": new_value}
        changes[rel_path] = file_changes
        if not dry_run:
            path.write_text(yaml.safe_dump(data, sort_keys=False))

    return changes


def _validate_patch_key_value(key: str, value: Any, requested_files: set[str]) -> list[str]:
    errors: list[str] = []
    if "." not in key:
        return [f"patch key must be dotted: {key}"]

    try:
        target_file = target_file_for_key(key)
    except ValueError as exc:
        return [str(exc)]

    if target_file not in requested_files:
        errors.append(f"patch key {key} targets {target_file}, which is not listed in allowed_files")

    if not _is_safe_scalar_or_list(value):
        errors.append(f"patch value for {key} must be a scalar or scalar list")

    range_error = _range_error(key, value)
    if range_error:
        errors.append(range_error)
    return errors


def _is_safe_scalar_or_list(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, list):
        return all(isinstance(item, (str, int, float, bool)) or item is None for item in value)
    return False


def _range_error(key: str, value: Any) -> str | None:
    if key.startswith("reward_weights."):
        return _numeric_range(key, value, 0.0, 2.0)
    if key.startswith("curriculum.roughness_"):
        return _numeric_range(key, value, 0.0, 0.25)
    if key.startswith("curriculum.push_probability_"):
        return _numeric_range(key, value, 0.0, 0.25)
    if key.startswith("curriculum.slope_"):
        return _numeric_range(key, value, -15.0, 15.0)
    if key.startswith("curriculum.command_velocity_"):
        return _numeric_range(key, value, 0.0, 3.0)
    if key == "domain_randomization.friction_range":
        return _numeric_pair_range(key, value, 0.2, 2.0)
    if key == "domain_randomization.motor_strength_scale":
        return _numeric_pair_range(key, value, 0.5, 1.25)
    if key == "domain_randomization.action_delay_steps":
        return _numeric_pair_range(key, value, 0, 6)
    if key == "domain_randomization.payload_mass_kg":
        return _numeric_pair_range(key, value, 0.0, 10.0)
    if key == "domain_randomization.push_force_range_n":
        return _numeric_pair_range(key, value, 0.0, 250.0)
    if key == "domain_randomization.push_impulse_probability":
        return _numeric_range(key, value, 0.0, 0.25)
    if key.startswith("actuators.") and key.endswith("_scale"):
        high = 1.0 if key == "actuators.torque_limit_scale" else 1.5
        return _numeric_range(key, value, 0.1, high)
    if key == "ppo.max_iterations":
        return _numeric_range(key, value, 1, 100_000)
    if key == "ppo.num_envs":
        return _numeric_range(key, value, 1, 16_384)
    if key == "ppo.learning_rate":
        return _numeric_range(key, value, 1e-6, 3e-3)
    if key in {"ppo.entropy_coef", "ppo.clip_param", "ppo.gamma", "ppo.lam"}:
        return _numeric_range(key, value, 0.0, 1.0)
    if key == "terrain.height_noise_m":
        return _numeric_range(key, value, 0.0, 0.25)
    if key == "terrain.slope_range_deg":
        return _numeric_pair_range(key, value, -20.0, 20.0)
    if key.startswith("terrain.") and key.endswith("_enabled"):
        return None if isinstance(value, bool) else f"{key} must be boolean"
    if key == "terrain.type":
        return None if value in {"flat", "rough", "stairs", "stepping_stones", "mixed"} else f"{key} has invalid terrain type"
    return f"unsupported patch parameter: {key}"


def _numeric_range(key: str, value: Any, low: float, high: float) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"{key} must be numeric"
    if not low <= float(value) <= high:
        return f"{key}={value} outside safe range [{low}, {high}]"
    return None


def _numeric_pair_range(key: str, value: Any, low: float, high: float) -> str | None:
    if not isinstance(value, list) or len(value) != 2:
        return f"{key} must be a two-value list"
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return f"{key} must contain numeric values"
    if value[0] > value[1]:
        return f"{key} lower bound must be <= upper bound"
    if float(value[0]) < low or float(value[1]) > high:
        return f"{key}={value} outside safe range [{low}, {high}]"
    return None


def _set_dotted_value(data: dict[str, Any], dotted_key: str, value: Any) -> Any:
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set nested key under non-object: {part}")
        cursor = next_value
    old_value = cursor.get(parts[-1])
    cursor[parts[-1]] = value
    return old_value

