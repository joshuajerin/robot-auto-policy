"""Render up to two Isaac Lab review videos and rollout diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_VIEWS: dict[str, dict[str, list[float]]] = {
    "front": {"eye": [4.0, 0.0, 1.7], "target": [0.0, 0.0, 0.95]},
    "side": {"eye": [0.0, -4.0, 1.7], "target": [0.0, 0.0, 0.95]},
    "diagonal": {"eye": [3.2, -3.2, 2.0], "target": [0.0, 0.0, 1.0]},
}

TABLETOP_VIEWS: dict[str, dict[str, list[float]]] = {
    "front": {"eye": [2.15, 0.0, 1.25], "target": [0.56, 0.0, 0.86]},
    "side": {"eye": [0.62, -2.05, 1.18], "target": [0.62, 0.0, 0.86]},
    "diagonal": {"eye": [1.95, -1.65, 1.35], "target": [0.56, 0.0, 0.9]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-id", default="h1_policy")
    parser.add_argument("--runner", default="rsl_rl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--video-length", type=int, default=240)
    parser.add_argument("--views", default="side,diagonal")
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--fixed-camera", action="store_true", default=False)
    parser.add_argument("--real-time", action="store_true", default=False)
    parser.add_argument("--config-patch", default="")
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
    simulation_app = app_launcher.app

    try:
        report = render_multiview(args_cli, simulation_app)
    finally:
        simulation_app.close()

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def render_multiview(args_cli: argparse.Namespace, simulation_app: Any) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    _register_robogenesis_tasks()
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_views = [name.strip() for name in args_cli.views.split(",") if name.strip()]
    view_presets = _view_presets_for_task(args_cli.task)
    unknown = [name for name in selected_views if name not in view_presets]
    if unknown:
        raise ValueError(f"Unknown view names: {unknown}. Available: {sorted(view_presets)}")

    view_reports: list[dict[str, Any]] = []
    for index, view_name in enumerate(selected_views):
        env = None
        view_dir = output_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        view_cfg = view_presets[view_name]
        try:
            env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
            env_cfg.seed = args_cli.seed
            agent_cfg = _load_rsl_rl_agent_cfg(args_cli)
            if args_cli.config_patch:
                from train_rsl_rl_policy import _load_patch_values, apply_training_patch

                apply_training_patch(env_cfg, agent_cfg, _load_patch_values(args_cli.config_patch))
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
            follow_camera = not args_cli.fixed_camera and not args_cli.task.startswith("RoboGenesis-H1-Tabletop")
            _set_camera(env.unwrapped, view_cfg, follow=follow_camera)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(view_dir),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
                name_prefix=f"{view_name}-policy",
            )
            env = RslRlVecEnvWrapper(env)

            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            print(f"[RoboGenesisRender] loading checkpoint: {args_cli.checkpoint}", flush=True)
            runner.load(args_cli.checkpoint)
            print("[RoboGenesisRender] checkpoint loaded", flush=True)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            diagnostics = RolloutDiagnostics(
                policy_id=args_cli.policy_id,
                view_name=view_name,
                seed=args_cli.seed,
                scenario_id=args_cli.scenario_id,
            )
            obs, _ = env.get_observations()
            dt = float(env.unwrapped.physics_dt)
            previous_actions = None
            print(
                f"[RoboGenesisRender] starting rollout view={view_name} scenario={args_cli.scenario_id or 'default'} "
                f"steps={args_cli.video_length}",
                flush=True,
            )

            for step in range(args_cli.video_length):
                start_time = time.time()
                _set_camera(env.unwrapped, view_cfg, follow=follow_camera)
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, rewards, dones, infos = env.step(actions)
                diagnostics.observe(
                    step=step,
                    obs=obs,
                    actions=actions,
                    previous_actions=previous_actions,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                )
                previous_actions = actions.detach().clone()
                if bool(dones.any().item()):
                    break
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
            print(
                f"[RoboGenesisRender] finished rollout view={view_name} frames={len(diagnostics.frames)} "
                f"done_step={diagnostics.done_step}",
                flush=True,
            )

            env.close()
            env = None
            videos = sorted(str(path) for path in view_dir.glob("*.mp4"))
            view_report = diagnostics.to_report()
            view_report.update(
                {
                    "camera": view_cfg,
                    "video_dir": str(view_dir),
                    "video_paths": videos,
                    "primary_video_path": videos[-1] if videos else None,
                }
            )
            view_reports.append(view_report)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    print(f"[WARN] failed to close {view_name} env: {type(exc).__name__}: {exc}", flush=True)

    report = {
        "policy_id": args_cli.policy_id,
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "scenario_id": args_cli.scenario_id or None,
        "seed": args_cli.seed,
        "video_length": args_cli.video_length,
        "views": view_reports,
        "diagnosis": diagnose_views(view_reports),
        "autoresearch_inputs": {
            "failure_taxonomy": [
                "fall_forward",
                "fall_backward",
                "fall_sideways",
                "foot_slip",
                "toe_drag",
                "torso_pitch_instability",
                "command_tracking_failure",
                "excessive_energy",
                "stuck_or_no_progress",
                "oscillatory_actions",
            ],
            "planner_hint": (
                "Use these limited review videos plus rollout diagnostics to propose one bounded "
                "scenario, reward, curriculum, actuator, or domain-randomization patch."
            ),
        },
    }
    (output_dir / "multiview_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _set_camera(env: Any, view_cfg: dict[str, list[float]], *, follow: bool) -> None:
    if not follow:
        env.sim.set_camera_view(eye=view_cfg["eye"], target=view_cfg["target"])
        return

    target = _robot_target(env) or view_cfg["target"]
    eye_offset = [float(eye) - float(center) for eye, center in zip(view_cfg["eye"], view_cfg["target"])]
    eye = [float(target[index]) + eye_offset[index] for index in range(3)]
    env.sim.set_camera_view(eye=eye, target=target)


def _view_presets_for_task(task_name: str) -> dict[str, dict[str, list[float]]]:
    if task_name.startswith("RoboGenesis-H1-Tabletop"):
        return TABLETOP_VIEWS
    return DEFAULT_VIEWS


def _robot_target(env: Any) -> list[float] | None:
    try:
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].detach().float().cpu().tolist()
        return [float(root_pos[0]), float(root_pos[1]), float(root_pos[2])]
    except Exception:
        return None


def _load_rsl_rl_agent_cfg(args_cli: argparse.Namespace) -> Any:
    script_dir = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl"
    if script_dir.exists():
        sys.path.insert(0, str(script_dir))
    import cli_args  # type: ignore

    cfg_args = argparse.Namespace(**vars(args_cli))
    defaults = {
        "experiment_name": None,
        "run_name": None,
        "resume": False,
        "load_run": ".*",
        "logger": None,
        "log_project_name": None,
    }
    for key, value in defaults.items():
        if not hasattr(cfg_args, key):
            setattr(cfg_args, key, value)
    return cli_args.parse_rsl_rl_cfg(args_cli.task, cfg_args)


def _register_robogenesis_tasks() -> None:
    try:
        import robogenesis_tasks  # noqa: F401
    except Exception:
        return


class RolloutDiagnostics:
    def __init__(self, *, policy_id: str, view_name: str, seed: int, scenario_id: str = ""):
        self.policy_id = policy_id
        self.view_name = view_name
        self.seed = seed
        self.scenario_id = scenario_id
        self.frames: list[dict[str, Any]] = []
        self.rewards: list[float] = []
        self.command_errors: list[float] = []
        self.torso_tilt_terms: list[float] = []
        self.action_l2_terms: list[float] = []
        self.action_jerk_terms: list[float] = []
        self.done_step: int | None = None

    def observe(
        self,
        *,
        step: int,
        obs: Any,
        actions: Any,
        previous_actions: Any | None,
        rewards: Any,
        dones: Any,
        infos: Any,
    ) -> None:
        del infos
        obs_row = obs[0].detach().float().cpu().tolist()
        action_row = actions[0].detach().float().cpu().tolist()
        reward = float(rewards[0].detach().float().cpu().item())
        done = bool(dones[0].detach().cpu().item())

        base_lin_vel = obs_row[0:3]
        base_ang_vel = obs_row[3:6]
        projected_gravity = obs_row[6:9]
        velocity_command = obs_row[9:12]
        command_error = math.sqrt(
            (float(base_lin_vel[0]) - float(velocity_command[0])) ** 2
            + (float(base_lin_vel[1]) - float(velocity_command[1])) ** 2
        )
        torso_tilt = math.sqrt(float(projected_gravity[0]) ** 2 + float(projected_gravity[1]) ** 2)
        action_l2 = math.sqrt(sum(float(value) * float(value) for value in action_row) / max(1, len(action_row)))
        action_jerk = 0.0
        if previous_actions is not None:
            previous = previous_actions[0].detach().float().cpu().tolist()
            action_jerk = math.sqrt(
                sum((float(now) - float(prev)) ** 2 for now, prev in zip(action_row, previous)) / max(1, len(action_row))
            )

        self.rewards.append(reward)
        self.command_errors.append(command_error)
        self.torso_tilt_terms.append(torso_tilt)
        self.action_l2_terms.append(action_l2)
        self.action_jerk_terms.append(action_jerk)
        if done and self.done_step is None:
            self.done_step = step

        self.frames.append(
            {
                "step": step,
                "reward": reward,
                "done": done,
                "base_lin_vel": base_lin_vel,
                "base_ang_vel": base_ang_vel,
                "projected_gravity": projected_gravity,
                "velocity_command": velocity_command,
                "command_error_xy": command_error,
                "torso_tilt_xy": torso_tilt,
                "action_l2": action_l2,
                "action_jerk": action_jerk,
                "joint_pos": obs_row[12:31],
                "joint_vel": obs_row[31:50],
                "actions": action_row,
            }
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "view": self.view_name,
            "scenario_id": self.scenario_id or None,
            "policy_id": self.policy_id,
            "seed": self.seed,
            "frame_count": len(self.frames),
            "done_step": self.done_step,
            "mean_reward": _mean(self.rewards),
            "mean_command_error_xy": _mean(self.command_errors),
            "max_torso_tilt_xy": max(self.torso_tilt_terms) if self.torso_tilt_terms else 0.0,
            "mean_torso_tilt_xy": _mean(self.torso_tilt_terms),
            "mean_action_l2": _mean(self.action_l2_terms),
            "mean_action_jerk": _mean(self.action_jerk_terms),
            "frames": self.frames,
        }


def diagnose_views(view_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not view_reports:
        return {
            "primary_failure": "stuck_or_no_progress",
            "secondary_failures": [],
            "evidence": {},
            "suggested_research_directions": ["rerun multiview render; no views were produced"],
        }

    min_frames = min(int(view.get("frame_count", 0)) for view in view_reports)
    mean_command_error = _mean([float(view.get("mean_command_error_xy", 0.0)) for view in view_reports])
    max_tilt = max(float(view.get("max_torso_tilt_xy", 0.0)) for view in view_reports)
    mean_jerk = _mean([float(view.get("mean_action_jerk", 0.0)) for view in view_reports])
    done_steps = [view.get("done_step") for view in view_reports if view.get("done_step") is not None]

    secondary: list[str] = []
    directions: list[str] = []
    if done_steps:
        primary = "fall_forward" if mean_command_error < 0.8 else "command_tracking_failure"
        directions.extend(["increase termination avoidance pressure", "increase torso upright/stability reward"])
    elif max_tilt > 0.45:
        primary = "torso_pitch_instability"
        directions.extend(["increase torso upright reward", "add angular velocity penalty"])
    elif mean_command_error > 0.65:
        primary = "command_tracking_failure"
        directions.extend(["increase command tracking reward", "extend flat velocity curriculum"])
    elif mean_jerk > 0.35:
        primary = "oscillatory_actions"
        directions.extend(["increase action rate penalty", "reduce action scale if actuator motion is too abrupt"])
    else:
        primary = "no_major_failure_detected"
        directions.extend(["generate harder scenarios", "evaluate on pushes, slopes, and rough terrain"])

    if max_tilt > 0.35 and primary != "torso_pitch_instability":
        secondary.append("torso_pitch_instability")
    if mean_command_error > 0.45 and primary != "command_tracking_failure":
        secondary.append("command_tracking_failure")
    if mean_jerk > 0.25 and primary != "oscillatory_actions":
        secondary.append("oscillatory_actions")

    return {
        "primary_failure": primary,
        "secondary_failures": secondary,
        "evidence": {
            "min_frame_count": min_frames,
            "done_steps": done_steps,
            "mean_command_error_xy": mean_command_error,
            "max_torso_tilt_xy": max_tilt,
            "mean_action_jerk": mean_jerk,
        },
        "suggested_research_directions": directions,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
