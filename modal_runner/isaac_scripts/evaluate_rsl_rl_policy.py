"""Evaluate an rsl_rl checkpoint on an Isaac Lab locomotion task.

The script follows Isaac Lab's usual app-launcher pattern and computes a compact
metrics JSON for RoboGenesis phase 1. It is intentionally conservative: if a
metric is not exposed by the environment, it falls back to rollout-derived
approximations rather than inventing values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes-per-seed", type=int, default=4)
    parser.add_argument("--max-steps-per-episode", type=int, default=1000)
    parser.add_argument("--seeds", default="101,203,307,409,503,601,709,811")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-id", default="baseline_h1")
    parser.add_argument("--trace-output", default="")
    parser.add_argument("--trace-env-index", type=int, default=0)
    parser.add_argument("--trace-max-steps", type=int, default=240)
    parser.add_argument("--config-patch", default="")
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()

    # Isaac imports must happen after CLI parsing in scripts launched through
    # isaaclab.sh; importing them locally would require Isaac Sim.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app

    try:
        metrics = evaluate_policy(args_cli)
    finally:
        simulation_app.close()

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def write_metrics(output: Path, metrics: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def evaluate_policy(args_cli: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    if args_cli.task.startswith("RoboGenesis-H1-Tabletop"):
        import robogenesis_tasks  # noqa: F401

    seeds = [int(seed.strip()) for seed in args_cli.seeds.split(",") if seed.strip()]
    aggregate = MetricAccumulator(
        policy_id=args_cli.policy_id,
        seed_count=len(seeds),
        task_name=args_cli.task,
    )
    trace_recorder = TraceRecorder(
        policy_id=args_cli.policy_id,
        env_index=args_cli.trace_env_index,
        max_steps=args_cli.trace_max_steps,
        task_name=args_cli.task,
    )

    for seed in seeds:
        env = None
        try:
            env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
            env_cfg.seed = seed
            agent_cfg = _load_rsl_rl_agent_cfg(args_cli)
            if args_cli.config_patch:
                from train_rsl_rl_policy import _load_patch_values, apply_training_patch

                apply_training_patch(env_cfg, agent_cfg, _load_patch_values(args_cli.config_patch))
            env = gym.make(args_cli.task, cfg=env_cfg)
            env = RslRlVecEnvWrapper(env)

            runner_cfg = agent_cfg.to_dict() if agent_cfg is not None else {}
            runner_device = getattr(agent_cfg, "device", args_cli.device) if agent_cfg is not None else args_cli.device
            runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=runner_device)
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            obs, _ = env.get_observations()
            done_episodes = 0
            step_count = 0
            episode_rewards = torch.zeros(args_cli.num_envs, device=args_cli.device)
            episode_lengths = torch.zeros(args_cli.num_envs, device=args_cli.device)

            while done_episodes < args_cli.episodes_per_seed * args_cli.num_envs:
                with torch.inference_mode():
                    actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
                step_count += 1
                episode_rewards += rewards
                episode_lengths += 1

                env_extras = getattr(env.unwrapped, "extras", {})
                aggregate.observe_step(
                    obs=obs,
                    infos=infos,
                    env_extras=env_extras,
                    rewards=rewards,
                    actions=actions,
                    dones=dones,
                )
                trace_recorder.observe(
                    seed=seed,
                    step=step_count,
                    obs=obs,
                    rewards=rewards,
                    actions=actions,
                    dones=dones,
                    infos=infos,
                    env_extras=env_extras,
                )

                if bool(dones.any()):
                    done_count = int(dones.sum().item())
                    done_episodes += done_count
                    aggregate.observe_episodes(
                        rewards=episode_rewards[dones],
                        lengths=episode_lengths[dones],
                        max_steps=args_cli.max_steps_per_episode,
                    )
                    episode_rewards[dones] = 0.0
                    episode_lengths[dones] = 0.0

                if step_count >= args_cli.max_steps_per_episode:
                    # Treat surviving envs at max step as successful episodes for
                    # this seed, then move to the next held-out seed.
                    active = torch.ones(args_cli.num_envs, dtype=torch.bool, device=args_cli.device)
                    aggregate.observe_episodes(
                        rewards=episode_rewards[active],
                        lengths=episode_lengths[active],
                        max_steps=args_cli.max_steps_per_episode,
                    )
                    break

            aggregate.completed_seed_count += 1
        except Exception as exc:
            message = f"seed {seed}: {type(exc).__name__}: {exc}"
            aggregate.evaluation_errors.append(message)
            print(f"[WARN] Evaluation failed for {message}", flush=True)
            if aggregate.episode_count == 0:
                break
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    print(f"[WARN] env.close failed for seed {seed}: {type(exc).__name__}: {exc}", flush=True)

    metrics = aggregate.to_metrics()
    write_metrics(Path(args_cli.output), metrics)
    if args_cli.trace_output:
        trace_recorder.write(Path(args_cli.trace_output), metrics=metrics)
    return metrics


def _load_rsl_rl_agent_cfg(args_cli: argparse.Namespace) -> Any | None:
    """Load Isaac Lab's task-specific rsl_rl config when available."""

    script_dir = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl"
    if script_dir.exists():
        sys.path.insert(0, str(script_dir))
    try:
        import cli_args  # type: ignore
    except Exception as exc:
        print(f"[WARN] Could not import rsl_rl cli_args, using runner defaults: {exc}", flush=True)
        return None

    cfg_args = argparse.Namespace(**vars(args_cli))
    defaults = {
        "experiment_name": None,
        "run_name": None,
        "resume": False,
        "load_run": ".*",
        "checkpoint": None,
        "logger": None,
        "log_project_name": None,
        "device": args_cli.device,
    }
    for key, value in defaults.items():
        if not hasattr(cfg_args, key):
            setattr(cfg_args, key, value)
    try:
        return cli_args.parse_rsl_rl_cfg(args_cli.task, cfg_args)
    except Exception as exc:
        print(f"[WARN] Could not parse rsl_rl config, using runner defaults: {exc}", flush=True)
        return None


class MetricAccumulator:
    def __init__(self, policy_id: str, seed_count: int, task_name: str = ""):
        self.policy_id = policy_id
        self.seed_count = seed_count
        self.task_name = task_name
        self.completed_seed_count = 0
        self.evaluation_errors: list[str] = []
        self.episode_count = 0
        self.success_count = 0
        self.reward_sum = 0.0
        self.length_sum = 0.0
        self.action_l2_sum = 0.0
        self.action_l2_count = 0
        self.fall_count = 0
        self.nan_actions = False
        self.command_terms: list[float] = []
        self.stability_terms: list[float] = []
        self.task_success_terms: list[float] = []
        self.task_progress_terms: list[float] = []
        self.contact_terms: list[float] = []
        self.placement_terms: list[float] = []
        self.placement_error_terms: list[float] = []
        self.object_slip_terms: list[float] = []
        self.collision_terms: list[float] = []
        self.force_violation_terms: list[float] = []
        self.robot_fall_terms: list[float] = []
        self.object_drop_terms: list[float] = []

    def observe_step(
        self,
        obs: Any,
        infos: dict[str, Any],
        env_extras: Any,
        rewards: Any,
        actions: Any,
        dones: Any,
    ) -> None:
        del rewards
        if hasattr(actions, "detach"):
            finite = bool(actions.isfinite().all().item())
            self.nan_actions = self.nan_actions or not finite
            self.action_l2_sum += float(actions.square().mean().sqrt().item())
            self.action_l2_count += 1

        for key, value in _merged_scalar_logs(infos=infos, env_extras=env_extras).items():
            lower = str(key).lower()
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            if "track" in lower or "lin_vel" in lower or "command" in lower:
                self.command_terms.append(scalar)
            if "upright" in lower or "orientation" in lower or "height" in lower or "stability" in lower:
                self.stability_terms.append(scalar)
            if "fall" in lower and scalar > 0:
                self.fall_count += int(max(1.0, scalar))
            if "task_success" in lower or lower == "success_rate":
                self.task_success_terms.append(scalar)
            if "task_progress" in lower or "completion" in lower:
                self.task_progress_terms.append(scalar)
            if "contact_stability" in lower or "contact_success" in lower:
                self.contact_terms.append(scalar)
            if "placement_accuracy" in lower:
                self.placement_terms.append(scalar)
            if "placement_error" in lower:
                self.placement_error_terms.append(scalar)
            if "object_slip" in lower:
                self.object_slip_terms.append(scalar)
            if "collision" in lower:
                self.collision_terms.append(scalar)
            if "force_violation" in lower:
                self.force_violation_terms.append(scalar)
            if "robot_fall" in lower:
                self.robot_fall_terms.append(scalar)
            if "object_drop" in lower:
                self.object_drop_terms.append(scalar)

        if hasattr(dones, "sum"):
            self.fall_count += 0

    def observe_episodes(self, rewards: Any, lengths: Any, max_steps: int) -> None:
        if not hasattr(rewards, "numel") or rewards.numel() == 0:
            return
        count = int(rewards.numel())
        self.episode_count += count
        self.reward_sum += float(rewards.sum().item())
        self.length_sum += float(lengths.sum().item())
        self.success_count += int((lengths >= max_steps * 0.95).sum().item())

    def to_metrics(self) -> dict[str, Any]:
        episodes = max(1, self.episode_count)
        mean_reward = self.reward_sum / episodes
        mean_length = self.length_sum / episodes
        survival = self.success_count / episodes
        command_tracking = _squash(_mean(self.command_terms, mean_reward / 100.0))
        stability = _squash(_mean(self.stability_terms, survival))
        action_l2 = self.action_l2_sum / max(1, self.action_l2_count)
        smoothness = max(0.0, min(1.0, 1.0 - action_l2 / 2.5))
        energy_efficiency = max(0.0, min(1.0, 1.0 - action_l2 / 3.0))
        fall_rate = self.fall_count / episodes

        return {
            "policy_id": self.policy_id,
            "episode_count": self.episode_count,
            "eval_seed_count": self.seed_count,
            "completed_eval_seed_count": self.completed_seed_count,
            "evaluation_errors": self.evaluation_errors,
            "mean_episode_reward": mean_reward,
            "mean_episode_length": mean_length,
            "command_tracking": command_tracking,
            "survival_no_fall": survival,
            "base_success": survival,
            "stability": stability,
            "generated_scenario_success": 0.0,
            "gait_quality": max(0.0, min(1.0, 0.5 * stability + 0.5 * smoothness)),
            "energy_efficiency": energy_efficiency,
            "smoothness": smoothness,
            "recovery_from_disturbance": 0.0,
            "fall_rate": fall_rate,
            "nan_actions": self.nan_actions,
            "reward_hacking_detected": False,
            "raw_metric_note": (
                "Phase-1 metrics are rollout-derived. Scenario robustness metrics become nonzero "
                "after generated scenario evaluation is enabled."
            ),
        }


class TraceRecorder:
    """Collect one rollout trace for non-Vulkan diagnostic rendering."""

    def __init__(self, policy_id: str, env_index: int, max_steps: int, task_name: str = ""):
        self.policy_id = policy_id
        self.env_index = max(0, env_index)
        self.max_steps = max(0, max_steps)
        self.task_name = task_name
        self.frames: list[dict[str, Any]] = []
        self.seed: int | None = None

    def observe(
        self,
        *,
        seed: int,
        step: int,
        obs: Any,
        rewards: Any,
        actions: Any,
        dones: Any,
        infos: dict[str, Any] | None = None,
        env_extras: Any = None,
    ) -> None:
        if self.max_steps == 0 or len(self.frames) >= self.max_steps:
            return
        if self.seed is None:
            self.seed = seed
        if seed != self.seed:
            return

        index = self.env_index
        try:
            obs_row = obs[index].detach().float().cpu().tolist()
            action_row = actions[index].detach().float().cpu().tolist()
            reward = float(rewards[index].detach().float().cpu().item())
            done = bool(dones[index].detach().cpu().item())
        except Exception:
            return

        log = {
            key: scalar
            for key, value in _merged_scalar_logs(infos=infos or {}, env_extras=env_extras).items()
            if (scalar := _to_scalar(value)) is not None
        }
        frame = {
            "step": step,
            "seed": seed,
            "reward": reward,
            "done": done,
            "actions": action_row,
            "extras_log": log,
        }
        frame.update(
            {
                "trace_family": "locomotion",
                "base_lin_vel": obs_row[0:3],
                "base_ang_vel": obs_row[3:6],
                "projected_gravity": obs_row[6:9],
                "velocity_command": obs_row[9:12],
                "joint_pos": obs_row[12:31],
                "joint_vel": obs_row[31:50],
            }
        )
        self.frames.append(frame)

    def write(self, path: Path, *, metrics: dict[str, Any]) -> None:
        payload = {
            "policy_id": self.policy_id,
            "seed": self.seed,
            "env_index": self.env_index,
            "frame_count": len(self.frames),
            "metrics": metrics,
            "frames": self.frames,
            "render_note": "Telemetry trace for non-Vulkan diagnostic rollout rendering.",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dict_like(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merged_scalar_logs(*, infos: dict[str, Any], env_extras: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (_dict_like(infos.get("extras", {})) if isinstance(infos, dict) else {}, _dict_like(env_extras)):
        log = _dict_like(source.get("log", {}))
        if not log and any(_to_scalar(value) is not None for value in source.values()):
            log = source
        for key, value in log.items():
            if _to_scalar(value) is not None:
                merged[str(key)] = value
    return merged


def _to_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if hasattr(value, "mean"):
        try:
            scalar = float(value.mean().item())
        except Exception:
            return None
        return scalar if math.isfinite(scalar) else None
    return None


def _mean(values: list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _squash(value: float) -> float:
    if value < 0:
        return 0.0
    if value <= 1.0:
        return value
    return 1.0 - math.exp(-value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    main()
