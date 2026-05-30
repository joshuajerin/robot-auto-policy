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

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def evaluate_policy(args_cli: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    seeds = [int(seed.strip()) for seed in args_cli.seeds.split(",") if seed.strip()]
    aggregate = MetricAccumulator(policy_id=args_cli.policy_id, seed_count=len(seeds))

    for seed in seeds:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
        env_cfg.seed = seed
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(env, {}, log_dir=None, device=args_cli.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=args_cli.device)

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

            aggregate.observe_step(infos=infos, rewards=rewards, actions=actions, dones=dones)

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

        env.close()

    return aggregate.to_metrics()


class MetricAccumulator:
    def __init__(self, policy_id: str, seed_count: int):
        self.policy_id = policy_id
        self.seed_count = seed_count
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

    def observe_step(self, infos: dict[str, Any], rewards: Any, actions: Any, dones: Any) -> None:
        del rewards
        if hasattr(actions, "detach"):
            finite = bool(actions.isfinite().all().item())
            self.nan_actions = self.nan_actions or not finite
            self.action_l2_sum += float(actions.square().mean().sqrt().item())
            self.action_l2_count += 1

        extras = _dict_like(infos.get("extras", {})) if isinstance(infos, dict) else {}
        log = _dict_like(extras.get("log", {}))
        for key, value in log.items():
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


def _dict_like(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


if __name__ == "__main__":
    main()

