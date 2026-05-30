"""Research motion-reference preparation for locomotion style conditioning."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_CONFIG = REPO_ROOT / "configs" / "motion_references" / "cmu_human_walk.yaml"


def prepare_motion_reference(
    *,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_REFERENCE_CONFIG,
    motion_id: str = "07_01",
    refresh: bool = False,
) -> dict[str, Any]:
    """Download a configured research mocap trial and write motion_context.json."""

    config = load_motion_reference_config(config_path)
    motion = select_motion(config, motion_id)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_motion"
    source_dir.mkdir(parents=True, exist_ok=True)

    asf_path = source_dir / f"{config['subject_id']}.asf"
    amc_path = source_dir / f"{motion['motion_id']}.amc"
    _download_if_needed(str(motion["asf_url"]), asf_path, refresh=refresh)
    _download_if_needed(str(motion["amc_url"]), amc_path, refresh=refresh)

    motion_metadata = inspect_amc_motion(amc_path, sample_rate_hz=float(config["sample_rate_hz"]))
    context = build_motion_context(
        config=config,
        motion=motion,
        asf_path=asf_path,
        amc_path=amc_path,
        motion_metadata=motion_metadata,
    )

    context_path = output / "motion_context.json"
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    return {
        "motion_context": str(context_path),
        "asf_path": str(asf_path),
        "amc_path": str(amc_path),
        "context": context,
    }


def load_motion_reference_config(config_path: str | Path = DEFAULT_REFERENCE_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"motion reference config must be a mapping: {path}")
    return data


def select_motion(config: dict[str, Any], motion_id: str) -> dict[str, Any]:
    for motion in config.get("motions", []):
        if str(motion.get("motion_id")) == motion_id:
            return dict(motion)
    available = [str(motion.get("motion_id")) for motion in config.get("motions", [])]
    raise ValueError(f"unknown motion_id {motion_id!r}; available motions: {available}")


def inspect_amc_motion(amc_path: str | Path, *, sample_rate_hz: float) -> dict[str, Any]:
    path = Path(amc_path)
    frame_count = count_amc_frames(path)
    return {
        "frame_count": frame_count,
        "sample_rate_hz": sample_rate_hz,
        "duration_seconds": round(frame_count / sample_rate_hz, 4) if sample_rate_hz > 0 else None,
        "size_bytes": path.stat().st_size,
    }


def count_amc_frames(amc_path: str | Path) -> int:
    """Count AMC frame markers without parsing joint channels."""

    count = 0
    for raw_line in Path(amc_path).read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if line and line.isdigit():
            count += 1
    return count


def build_motion_context(
    *,
    config: dict[str, Any],
    motion: dict[str, Any],
    asf_path: str | Path,
    amc_path: str | Path,
    motion_metadata: dict[str, Any],
) -> dict[str, Any]:
    style_context = dict(config.get("style_context", {}))
    style_context.setdefault("style", "research_mocap_walk")
    style_context["source_type"] = "research_motion_capture"
    style_context["source_motion_id"] = motion["motion_id"]

    return {
        "reference_id": config["reference_id"],
        "dataset_id": config["dataset_id"],
        "dataset_name": config["dataset_name"],
        "source_homepage": config["source_homepage"],
        "license_note": config.get("license_note", ""),
        "format": config.get("format", "asf_amc"),
        "subject_id": config["subject_id"],
        "motion_id": motion["motion_id"],
        "motion_description": motion.get("description", ""),
        "motion_files": {
            "skeleton_asf": str(asf_path),
            "motion_amc": str(amc_path),
            "asf_url": motion["asf_url"],
            "amc_url": motion["amc_url"],
        },
        "motion_metadata": motion_metadata,
        "style_context": style_context,
        "training_use": dict(config.get("training_use", {})),
        "usage_note": (
            "Phase 1 uses research mocap as style and reward-target context. "
            "The LLM may propose bounded reward/curriculum patches from this "
            "context, but it never directly controls torques."
        ),
    }


def _download_if_needed(url: str, destination: Path, *, refresh: bool) -> None:
    if destination.exists() and not refresh:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "RoboGenesis research prototype"})
    with urllib.request.urlopen(request, timeout=90) as response:
        destination.write_bytes(response.read())
