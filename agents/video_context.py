"""Version-1 video context extraction placeholder.

This module treats video as style context, not direct robot control. It returns
structured style targets that can condition planning and reporting.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def summarize_locomotion_video(
    video_path: str | Path,
    *,
    source_url: str | None = None,
    license_name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    metadata = probe_video(path)
    return {
        "style": "upright human walk",
        "cadence_hz": 1.75,
        "stride_symmetry": 0.9,
        "stance_swing_ratio": 0.62,
        "torso_lean": "slight_forward",
        "foot_clearance": "moderate",
        "arm_swing": "natural",
        "target_velocity_class": "normal_walk",
        "reward_bias": {
            "torso_upright": "high",
            "gait_symmetry": "high",
            "smoothness": "medium_high",
            "foot_slip_penalty": "high",
            "energy_penalty": "medium",
        },
        "source_video": str(path),
        "source_url": source_url,
        "license": license_name,
        "description": description,
        "video_metadata": metadata,
        "usage_note": (
            "V1 style context only. This is not direct motion retargeting or "
            "torque control; it biases reward/curriculum proposals toward a "
            "normal upright walking style."
        ),
    }


def probe_video(video_path: str | Path) -> dict[str, Any]:
    path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"size_bytes": path.stat().st_size}
    raw = json.loads(proc.stdout)
    stream = (raw.get("streams") or [{}])[0]
    fmt = raw.get("format") or {}
    return {
        "width": _int_or_none(stream.get("width")),
        "height": _int_or_none(stream.get("height")),
        "frame_rate": stream.get("r_frame_rate"),
        "stream_duration_seconds": _float_or_none(stream.get("duration")),
        "format_duration_seconds": _float_or_none(fmt.get("duration")),
        "size_bytes": _int_or_none(fmt.get("size")) or path.stat().st_size,
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
