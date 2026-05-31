"""Prepare user MP4 videos for humanoid motion-imitation training.

This module does not directly control robot torques. It creates an auditable
motion-imitation context: video metadata, sampled frames, a human-keypoint to H1
joint retarget map, and bounded reward/curriculum suggestions for training.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_SPEC = REPO_ROOT / "assets" / "h1_robot_spec.json"
DEFAULT_SKILL_SPEC = REPO_ROOT / "skills" / "humanoid_motion_imitation" / "skill.yaml"


HUMAN_TO_H1_RETARGET_MAP: dict[str, dict[str, Any]] = {
    "pelvis": {"robot_joints": ["torso"], "targets": ["root_height", "root_yaw", "root_velocity"]},
    "left_hip": {"robot_joints": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch"], "targets": ["hip_3d_angle"]},
    "left_knee": {"robot_joints": ["left_knee"], "targets": ["knee_flexion"]},
    "left_ankle": {"robot_joints": ["left_ankle"], "targets": ["ankle_pitch", "foot_clearance"]},
    "right_hip": {"robot_joints": ["right_hip_yaw", "right_hip_roll", "right_hip_pitch"], "targets": ["hip_3d_angle"]},
    "right_knee": {"robot_joints": ["right_knee"], "targets": ["knee_flexion"]},
    "right_ankle": {"robot_joints": ["right_ankle"], "targets": ["ankle_pitch", "foot_clearance"]},
    "left_shoulder": {
        "robot_joints": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw"],
        "targets": ["arm_swing_phase", "shoulder_angle"],
    },
    "left_elbow": {"robot_joints": ["left_elbow"], "targets": ["elbow_flexion"]},
    "right_shoulder": {
        "robot_joints": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw"],
        "targets": ["arm_swing_phase", "shoulder_angle"],
    },
    "right_elbow": {"robot_joints": ["right_elbow"], "targets": ["elbow_flexion"]},
}


def prepare_user_video_motion_skill(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    robot_spec_path: str | Path = DEFAULT_ROBOT_SPEC,
    skill_spec_path: str | Path = DEFAULT_SKILL_SPEC,
    sample_fps: float = 6.0,
    extract_frames: bool = False,
) -> dict[str, Any]:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(video)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "sampled_frames"
    metadata = inspect_video(video)
    robot_spec = json.loads(Path(robot_spec_path).read_text())
    retarget_map = build_h1_retarget_map(robot_spec)
    sampled_frames: list[str] = []
    if extract_frames:
        sampled_frames = extract_video_frames(video, frame_dir, sample_fps=sample_fps)

    context = {
        "skill_id": "humanoid_motion_imitation_v1",
        "source_type": "user_recorded_mp4",
        "video_path": str(video),
        "video_metadata": metadata,
        "sampled_frames_dir": str(frame_dir) if extract_frames else None,
        "sampled_frames": sampled_frames,
        "pose_estimation": {
            "required": True,
            "recommended_backends": ["mediapipe_pose", "openpose", "video_pose_3d"],
            "minimum_keypoints": [
                "pelvis",
                "left_hip",
                "left_knee",
                "left_ankle",
                "right_hip",
                "right_knee",
                "right_ankle",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            ],
            "note": "Use sampled frames plus pose estimation to produce time-indexed keypoints before full retargeting.",
        },
        "retarget_map": retarget_map,
        "style_context": build_style_context(metadata),
        "imitation_targets": build_imitation_targets(retarget_map),
        "training_hooks": build_training_hooks(),
        "safety": {
            "llm_direct_torque_control": False,
            "retargeted_motion_is_reference_only": True,
            "joint_limits_source": robot_spec.get("limits", {}).get("joint_position_limits", "from_asset"),
        },
        "skill_spec_path": str(skill_spec_path),
    }

    context_path = output / "motion_imitation_context.json"
    retarget_path = output / "h1_retarget_map.json"
    hooks_path = output / "training_hooks.json"
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    retarget_path.write_text(json.dumps(retarget_map, indent=2, sort_keys=True) + "\n")
    hooks_path.write_text(json.dumps(context["training_hooks"], indent=2, sort_keys=True) + "\n")
    return {
        "motion_imitation_context": str(context_path),
        "retarget_map": str(retarget_path),
        "training_hooks": str(hooks_path),
        "context": context,
    }


def inspect_video(video_path: str | Path) -> dict[str, Any]:
    video = Path(video_path)
    metadata: dict[str, Any] = {
        "size_bytes": video.stat().st_size,
        "duration_seconds": None,
        "width": None,
        "height": None,
        "fps": None,
        "probe": "unavailable",
    }
    if shutil.which("ffprobe") is None:
        return metadata
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(video),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        metadata["probe_error"] = proc.stderr.strip()
        return metadata
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return metadata
    stream = streams[0]
    metadata.update(
        {
            "duration_seconds": _coerce_float(stream.get("duration")),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": _parse_frame_rate(stream.get("r_frame_rate")),
            "probe": "ffprobe",
        }
    )
    return metadata


def extract_video_frames(video_path: str | Path, frame_dir: str | Path, *, sample_fps: float) -> list[str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for frame extraction")
    output = Path(frame_dir)
    output.mkdir(parents=True, exist_ok=True)
    pattern = output / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps}",
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(command, check=True)
    return [str(path) for path in sorted(output.glob("frame_*.jpg"))]


def build_h1_retarget_map(robot_spec: dict[str, Any]) -> dict[str, Any]:
    controlled = set(robot_spec.get("controlled_joints", []))
    map_entries: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for human_joint, mapping in HUMAN_TO_H1_RETARGET_MAP.items():
        robot_joints = [joint for joint in mapping["robot_joints"] if joint in controlled]
        if robot_joints:
            map_entries[human_joint] = {**mapping, "robot_joints": robot_joints}
        else:
            missing[human_joint] = list(mapping["robot_joints"])
    return {
        "robot_id": robot_spec.get("robot_id", "unitree_h1"),
        "embodiment_type": robot_spec.get("embodiment_type", "humanoid"),
        "mapping": map_entries,
        "missing_robot_joints": missing,
        "retargeting_mode": "reference_motion_not_direct_control",
    }


def build_imitation_targets(retarget_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "cadence_hz": {"source": "footfall_phase_from_video", "weight": 0.25},
        "pelvis_root_velocity": {"source": "pelvis_track", "weight": 0.2},
        "left_right_stride_symmetry": {"source": "ankle_tracks", "weight": 0.2},
        "joint_angle_reference": {
            "source": "retarget_map",
            "weight": 0.3,
            "mapped_human_joints": sorted(retarget_map["mapping"]),
        },
        "arm_swing_phase": {"source": "shoulder_elbow_tracks", "weight": 0.1},
        "foot_clearance": {"source": "ankle_vertical_tracks", "weight": 0.15},
    }


def build_training_hooks() -> dict[str, Any]:
    return {
        "editable_files": [
            "configs/locomotion/rewards.yaml",
            "configs/locomotion/curriculum.yaml",
            "configs/locomotion/domain_randomization.yaml",
        ],
        "suggested_patch": {
            "reward_weights.gait_symmetry": 0.35,
            "reward_weights.smoothness": 0.22,
            "reward_weights.foot_clearance": 0.28,
            "reward_weights.torso_upright": 0.55,
        },
        "training_mode": "style_reference_then_motion_imitation",
        "phase1": "Use extracted gait/style targets as reward conditioning.",
        "phase2": "Retarget keypoints into H1 joint reference trajectories.",
        "phase3": "Train with imitation or AMP-style discriminator rewards.",
    }


def build_style_context(video_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "style": "user_recorded_humanoid_motion",
        "source_type": "user_video_motion_imitation",
        "target_velocity_class": "estimated_from_pose_tracks",
        "cadence_hz": "estimated_from_footfalls",
        "stride_symmetry": "estimated_from_left_right_phase",
        "torso_lean": "estimated_from_pelvis_shoulder_track",
        "arm_swing": "estimated_from_shoulder_elbow_tracks",
        "video_duration_seconds": video_metadata.get("duration_seconds"),
        "reward_bias": {
            "gait_symmetry": "high",
            "smoothness": "high",
            "torso_upright": "medium_high",
            "foot_clearance": "medium",
        },
    }


def _parse_frame_rate(value: Any) -> float | None:
    if not value:
        return None
    text = str(value)
    if "/" not in text:
        return _coerce_float(text)
    numerator, denominator = text.split("/", 1)
    den = _coerce_float(denominator)
    if not den:
        return None
    return round(_coerce_float(numerator) / den, 4)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
