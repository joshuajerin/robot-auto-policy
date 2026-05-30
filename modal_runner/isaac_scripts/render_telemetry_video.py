"""Render rollout telemetry to MP4 without Isaac cameras, RTX, or Vulkan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="RoboGenesis H1 rollout telemetry")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
    else:
        trace = {
            "policy_id": "unknown",
            "frames": [],
            "render_note": f"Trace file was missing: {trace_path}",
        }
    metrics = json.loads(Path(args.metrics).read_text()) if args.metrics else trace.get("metrics", {})
    render_telemetry_video(trace=trace, metrics=metrics, output=Path(args.output), title=args.title, fps=args.fps)


def render_telemetry_video(*, trace: dict[str, Any], metrics: dict[str, Any], output: Path, title: str, fps: int) -> None:
    import imageio.v2 as imageio

    frames = trace.get("frames") or []
    if not frames:
        frames = _fallback_frames(trace)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=max(1, fps), codec="libx264")
    try:
        stride = max(1, len(frames) // 180)
        for frame_index, frame in enumerate(frames[::stride]):
            writer.append_data(draw_frame(frame, metrics=metrics, title=title, frame_index=frame_index))
    finally:
        writer.close()


def draw_frame(frame: dict[str, Any], *, metrics: dict[str, Any], title: str, frame_index: int) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joint_pos = _as_float_list(frame.get("joint_pos", []), length=19)
    command = _as_float_list(frame.get("velocity_command", []), length=3)
    base_vel = _as_float_list(frame.get("base_lin_vel", []), length=3)
    actions = _as_float_list(frame.get("actions", []), length=19)

    fig = plt.figure(figsize=(10, 5.625), dpi=120)
    ax = fig.add_axes((0.04, 0.08, 0.56, 0.82))
    ax_metrics = fig.add_axes((0.64, 0.12, 0.32, 0.76))

    _draw_h1_schematic(ax, joint_pos, frame_index)
    _draw_metrics_panel(ax_metrics, frame, metrics, command, base_vel, actions, title)

    fig.canvas.draw()
    rgba = fig.canvas.buffer_rgba()
    image = _rgba_to_rgb_array(rgba)
    plt.close(fig)
    return image


def _draw_h1_schematic(ax: Any, joint_pos: list[float], frame_index: int) -> None:
    ax.set_title("H1 telemetry pose proxy", fontsize=12, pad=8)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.15, 2.25)
    ax.set_aspect("equal")
    ax.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.set_facecolor("#fbfbfb")
    ax.axhline(0, color="#8a8a8a", linewidth=1.0)

    phase = frame_index * 0.11
    left = _leg_points(-0.18, joint_pos[1], joint_pos[2], joint_pos[3], phase)
    right = _leg_points(0.18, joint_pos[6], joint_pos[7], joint_pos[8], phase + math.pi)
    torso_bottom = (0.0, 1.05)
    torso_top = (0.0 + 0.08 * math.sin(joint_pos[10] if len(joint_pos) > 10 else 0.0), 1.72)
    head = (torso_top[0], 1.94)
    left_shoulder = (torso_top[0] - 0.24, 1.58)
    right_shoulder = (torso_top[0] + 0.24, 1.58)
    left_hand = (left_shoulder[0] - 0.18, 1.08 + 0.10 * math.sin(phase + math.pi))
    right_hand = (right_shoulder[0] + 0.18, 1.08 + 0.10 * math.sin(phase))

    _plot_limb(ax, [torso_bottom, torso_top, head], "#2f4858", linewidth=5)
    _plot_limb(ax, [torso_top, left_shoulder, left_hand], "#33658a")
    _plot_limb(ax, [torso_top, right_shoulder, right_hand], "#33658a")
    _plot_limb(ax, [torso_bottom, *left], "#f26419")
    _plot_limb(ax, [torso_bottom, *right], "#2a9d8f")

    for point in [torso_bottom, torso_top, head, left_shoulder, right_shoulder, left_hand, right_hand, *left, *right]:
        ax.scatter(point[0], point[1], s=28, color="#1f2933", zorder=5)

    ax.text(-1.25, 2.08, "Non-Vulkan diagnostic render", fontsize=9, color="#555")
    ax.text(-1.25, 1.96, "Uses rollout obs/actions, not Isaac cameras", fontsize=8, color="#777")


def _leg_points(x_hip: float, hip_roll: float, hip_pitch: float, knee: float, phase: float) -> list[tuple[float, float]]:
    hip = (x_hip, 1.05)
    swing = 0.10 * math.sin(phase)
    knee_x = x_hip + 0.18 * math.sin(hip_pitch) + swing
    knee_y = 0.58 - 0.06 * math.cos(knee)
    foot_x = knee_x + 0.18 * math.sin(hip_roll + knee) + swing
    foot_y = 0.07 + 0.04 * max(0.0, math.sin(phase))
    return [(knee_x, knee_y), (foot_x, foot_y)]


def _plot_limb(ax: Any, points: list[tuple[float, float]], color: str, linewidth: int = 4) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle="round", zorder=3)


def _draw_metrics_panel(
    ax: Any,
    frame: dict[str, Any],
    metrics: dict[str, Any],
    command: list[float],
    base_vel: list[float],
    actions: list[float],
    title: str,
) -> None:
    ax.axis("off")
    action_rms = math.sqrt(sum(value * value for value in actions) / max(1, len(actions)))
    rows = [
        ("policy", str(metrics.get("policy_id", ""))[:30]),
        ("step", frame.get("step")),
        ("reward", f"{float(frame.get('reward', 0.0)):.3f}"),
        ("done", frame.get("done")),
        ("cmd vx", f"{command[0]:.3f}"),
        ("base vx", f"{base_vel[0]:.3f}"),
        ("action rms", f"{action_rms:.3f}"),
        ("total score", f"{float(metrics.get('total_score', 0.0)):.3f}"),
        ("survival", f"{float(metrics.get('survival_no_fall', 0.0)):.3f}"),
        ("safety", metrics.get("safety_passed", "unknown")),
    ]
    ax.text(0.0, 1.02, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
    y = 0.92
    for label, value in rows:
        ax.text(0.0, y, f"{label}", fontsize=9, color="#666", transform=ax.transAxes)
        ax.text(0.38, y, f"{value}", fontsize=9, color="#111", transform=ax.transAxes)
        y -= 0.075
    if metrics.get("safety_reasons"):
        ax.text(0.0, y - 0.02, "safety reasons", fontsize=9, color="#a33", transform=ax.transAxes)
        ax.text(0.0, y - 0.09, "; ".join(metrics["safety_reasons"])[:52], fontsize=8, color="#a33", transform=ax.transAxes)
    if frame.get("render_note"):
        ax.text(0.0, 0.02, str(frame["render_note"])[:64], fontsize=8, color="#8a5a00", transform=ax.transAxes)


def _as_float_list(values: Any, *, length: int) -> list[float]:
    if not isinstance(values, list):
        values = []
    output: list[float] = []
    for value in values[:length]:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            output.append(0.0)
    while len(output) < length:
        output.append(0.0)
    return output


def _rgba_to_rgb_array(buffer: Any) -> Any:
    import numpy as np

    array = np.asarray(buffer)
    return array[:, :, :3].copy()


def _fallback_frames(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": index + 1,
            "reward": 0.0,
            "done": False,
            "velocity_command": [0.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "joint_pos": [0.0] * 19,
            "actions": [0.0] * 19,
            "render_note": trace.get("render_note", "No rollout frames were recorded."),
        }
        for index in range(60)
    ]


if __name__ == "__main__":
    main()
