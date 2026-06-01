"""Train an rsl_rl Isaac Lab policy with a bounded RoboGenesis patch file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "running_patch_type_safe_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--run-name", default="robogenesis_h1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config-patch", default="")
    parser.add_argument("--patch-report", default="")
    parser.add_argument("--train-report", default="")
    parser.add_argument("--resume-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app

    try:
        report = train_policy(args_cli)
    finally:
        simulation_app.close()

    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)


def train_policy(args_cli: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    env = None
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    agent_cfg = _load_rsl_rl_agent_cfg(args_cli)
    patch_values = _load_patch_values(args_cli.config_patch)
    patch_report = apply_training_patch(env_cfg, agent_cfg, patch_values)
    patch_report["script_version"] = SCRIPT_VERSION
    if args_cli.patch_report:
        _write_json(Path(args_cli.patch_report), patch_report)
    print(f"[RoboGenesis train] script_version={SCRIPT_VERSION}", flush=True)

    log_dir = _build_log_dir(agent_cfg, args_cli)
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)
        runner_cfg = agent_cfg.to_dict() if agent_cfg is not None else {}
        runner_device = getattr(agent_cfg, "device", args_cli.device) if agent_cfg is not None else args_cli.device
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device=runner_device)
        if args_cli.resume_checkpoint:
            runner.load(args_cli.resume_checkpoint)
            if hasattr(runner, "log_dir"):
                runner.log_dir = str(log_dir)
        start_iteration = int(getattr(runner, "current_learning_iteration", 0))
        requested_iterations = max(1, int(args_cli.max_iterations))
        target_iteration = start_iteration + requested_iterations
        print(
            f"[RoboGenesis train] start_iteration={start_iteration} requested_iterations={requested_iterations} target_iteration={target_iteration}",
            flush=True,
        )
        runner.learn(num_learning_iterations=requested_iterations, init_at_random_ep_len=True)
        print("[RoboGenesis train] runner.learn returned", flush=True)
        final_iteration = int(getattr(runner, "current_learning_iteration", target_iteration))
        final_checkpoint = log_dir / f"model_{max(0, final_iteration)}.pt"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            runner.save(str(final_checkpoint))
        except Exception as exc:
            patch_report["warnings"].append(f"final checkpoint save fallback failed: {type(exc).__name__}: {exc}")
        if not final_checkpoint.exists() and args_cli.resume_checkpoint:
            try:
                import shutil

                log_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args_cli.resume_checkpoint, final_checkpoint)
                patch_report["warnings"].append("copied resume checkpoint because runner.save did not create a file")
            except Exception as exc:
                patch_report["warnings"].append(f"resume checkpoint copy fallback failed: {type(exc).__name__}: {exc}")
        report = {
            "task": args_cli.task,
            "run_name": args_cli.run_name,
            "log_dir": str(log_dir),
            "resume_checkpoint": args_cli.resume_checkpoint or None,
            "start_iteration": start_iteration,
            "target_iteration": target_iteration,
            "final_iteration": final_iteration,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_exists": final_checkpoint.exists(),
            "patch_report": patch_report,
        }
        if args_cli.patch_report:
            _write_json(Path(args_cli.patch_report), patch_report)
        if args_cli.train_report:
            _write_json(Path(args_cli.train_report), report)
        return report
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"[WARN] env.close failed after training: {type(exc).__name__}: {exc}", flush=True)


def apply_training_patch(env_cfg: Any, agent_cfg: Any, patch: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"applied": [], "unsupported": [], "warnings": []}
    if not patch:
        return report

    for key, value in sorted(patch.items()):
        try:
            if key.startswith("reward_weights."):
                _apply_reward_weight(env_cfg, key.removeprefix("reward_weights."), value, report)
            elif key.startswith("actuators."):
                _apply_actuator_patch(env_cfg, key.removeprefix("actuators."), value, report)
            elif key.startswith("domain_randomization."):
                _apply_domain_randomization(env_cfg, key.removeprefix("domain_randomization."), value, report)
            elif key.startswith("terrain."):
                _apply_terrain_patch(env_cfg, key.removeprefix("terrain."), value, report)
            elif key.startswith("commands."):
                _apply_command_patch(env_cfg, key.removeprefix("commands."), value, report)
            elif key.startswith("ppo."):
                _apply_ppo_patch(agent_cfg, key.removeprefix("ppo."), value, report)
            elif key.startswith("curriculum."):
                _apply_curriculum_patch(env_cfg, key.removeprefix("curriculum."), value, report)
            else:
                report["unsupported"].append({"key": key, "reason": "unsupported root"})
        except Exception as exc:
            report["warnings"].append(f"{key}: {type(exc).__name__}: {exc}")
    return report


def _apply_reward_weight(env_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        direct_aliases = {
            "torso_upright": "balance_weight",
            "stability": "balance_weight",
            "energy_penalty": "action_l2_penalty",
            "smoothness": "action_rate_penalty",
        }
        target = direct_aliases.get(name, name)
        if hasattr(env_cfg, target):
            old = getattr(env_cfg, target)
            new = _signed_weight(old, value)
            setattr(env_cfg, target, new)
            report["applied"].append({"key": f"reward_weights.{name}", "target": target, "old": old, "new": new})
            return
        report["unsupported"].append({"key": f"reward_weights.{name}", "reason": "env_cfg has no rewards manager"})
        return
    aliases = {
        "command_tracking": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "track_lin_vel_xy_yaw_frame_exp"],
        "survival": ["is_alive", "alive"],
        "stability": ["flat_orientation_l2", "base_height_l2", "body_height_l2"],
        "torso_upright": ["flat_orientation_l2", "base_height_l2"],
        "foot_clearance": ["feet_air_time", "feet_clearance", "feet_height"],
        "foot_slip_penalty": ["feet_slide", "feet_slip"],
        "energy_penalty": ["dof_torques_l2", "joint_torques_l2", "torques"],
        "smoothness": ["action_rate_l2", "action_rate", "joint_acc_l2"],
        "recovery": ["undesired_contacts", "termination_penalty"],
        "gait_symmetry": ["feet_air_time", "feet_contact"],
    }
    targets = aliases.get(name, [name])
    applied = False
    for target in targets:
        term = getattr(rewards, target, None)
        if term is None or not hasattr(term, "weight"):
            continue
        old = getattr(term, "weight")
        new = _signed_weight(old, value)
        setattr(term, "weight", new)
        report["applied"].append({"key": f"reward_weights.{name}", "target": f"rewards.{target}.weight", "old": old, "new": new})
        applied = True
    if not applied:
        report["unsupported"].append({"key": f"reward_weights.{name}", "reason": f"no matching reward terms among {targets}"})


def _apply_actuator_patch(env_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    robot_cfg = _robot_cfg(env_cfg)
    if robot_cfg is None:
        report["unsupported"].append({"key": f"actuators.{name}", "reason": "env_cfg.scene.robot not found"})
        return
    if name in {"stiffness_scale", "damping_scale", "torque_limit_scale"}:
        fields = {
            "stiffness_scale": ["stiffness"],
            "damping_scale": ["damping"],
            "torque_limit_scale": ["effort_limit", "effort_limit_sim"],
        }[name]
        count = _scale_named_fields(robot_cfg, fields, float(value), report, f"actuators.{name}")
        if count == 0:
            report["unsupported"].append({"key": f"actuators.{name}", "reason": f"no fields found: {fields}"})
        return
    if name.endswith("_action_scale"):
        group = name.removesuffix("_action_scale")
        action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        if action_cfg is None or not hasattr(action_cfg, "scale"):
            report["unsupported"].append({"key": f"actuators.{name}", "reason": "joint_pos action scale not found"})
            return
        old = getattr(action_cfg, "scale")
        new = _set_group_scale(old, group, float(value))
        if new is old:
            report["unsupported"].append({"key": f"actuators.{name}", "reason": "group-specific scale requires dict action scale"})
            return
        setattr(action_cfg, "scale", new)
        report["applied"].append({"key": f"actuators.{name}", "target": "actions.joint_pos.scale", "old": old, "new": new})
        return
    report["unsupported"].append({"key": f"actuators.{name}", "reason": "unsupported actuator parameter"})


def _apply_domain_randomization(env_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    if name == "friction_range":
        values = _range_tuple(value)
        count = _set_named_fields(
            env_cfg,
            ["static_friction_range", "dynamic_friction_range", "friction_range"],
            values,
            report,
            "domain_randomization.friction_range",
        )
        if count == 0:
            report["unsupported"].append({"key": "domain_randomization.friction_range", "reason": "no friction fields found"})
        return
    if name == "push_force_range_n":
        values = _range_tuple(value)
        velocity_range = {axis: (-max(abs(values[0]), abs(values[1])) / 100.0, max(abs(values[0]), abs(values[1])) / 100.0) for axis in ("x", "y")}
        count = _set_push_velocity_ranges(env_cfg, velocity_range, report)
        if count == 0:
            report["unsupported"].append({"key": "domain_randomization.push_force_range_n", "reason": "no push velocity_range field found"})
        return
    if name in {"motor_strength_scale", "action_delay_steps", "payload_mass_kg", "push_impulse_probability"}:
        report["unsupported"].append({"key": f"domain_randomization.{name}", "reason": "requires custom Isaac event term injection"})
        return
    report["unsupported"].append({"key": f"domain_randomization.{name}", "reason": "unsupported domain randomization parameter"})


def _apply_terrain_patch(env_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is None:
        report["unsupported"].append({"key": f"terrain.{name}", "reason": "scene terrain config not found"})
        return
    if name == "height_noise_m":
        count = _set_terrain_height_noise(terrain_cfg, value, report)
        if count == 0:
            report["unsupported"].append({"key": "terrain.height_noise_m", "reason": "no terrain noise fields found"})
        return
    if name == "slope_range_deg":
        count = _set_named_fields(terrain_cfg, ["slope_range"], _range_tuple(value), report, "terrain.slope_range_deg")
        if count == 0:
            report["unsupported"].append({"key": "terrain.slope_range_deg", "reason": "no terrain slope fields found"})
        return
    if name in {"type", "rough_heightfield_enabled", "stairs_enabled", "stepping_stones_enabled"}:
        report["applied"].append({"key": f"terrain.{name}", "target": "task_selection", "old": None, "new": value, "note": "task id selects the terrain family before env_cfg creation"})
        return
    report["unsupported"].append({"key": f"terrain.{name}", "reason": "unsupported terrain parameter"})


def _apply_curriculum_patch(env_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is None:
        report["unsupported"].append({"key": f"curriculum.{name}", "reason": "env_cfg has no curriculum manager"})
        return
    if name.startswith("roughness") or name.startswith("slope") or name.startswith("push"):
        report["applied"].append({"key": f"curriculum.{name}", "target": "curriculum_directive", "old": None, "new": value, "note": "recorded for scenario scheduler; no direct manager term matched safely"})
        return
    report["unsupported"].append({"key": f"curriculum.{name}", "reason": "unsupported curriculum parameter"})


def _apply_command_patch(env_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    commands = getattr(env_cfg, "commands", None)
    base_velocity = getattr(commands, "base_velocity", None)
    ranges = getattr(base_velocity, "ranges", None)
    if ranges is None:
        report["unsupported"].append({"key": f"commands.{name}", "reason": "base velocity command ranges not found"})
        return
    field_map = {
        "linear_velocity_x": "lin_vel_x",
        "linear_velocity_y": "lin_vel_y",
        "yaw_velocity": "ang_vel_z",
    }
    target = field_map.get(name)
    if target is None or not hasattr(ranges, target):
        report["unsupported"].append({"key": f"commands.{name}", "reason": "unsupported command range"})
        return
    old = getattr(ranges, target)
    new = _range_tuple(value)
    setattr(ranges, target, new)
    report["applied"].append(
        {"key": f"commands.{name}", "target": f"commands.base_velocity.ranges.{target}", "old": old, "new": new}
    )


def _apply_ppo_patch(agent_cfg: Any, name: str, value: Any, report: dict[str, Any]) -> None:
    if agent_cfg is None:
        report["unsupported"].append({"key": f"ppo.{name}", "reason": "agent cfg unavailable"})
        return
    if hasattr(agent_cfg, name):
        old = getattr(agent_cfg, name)
        setattr(agent_cfg, name, value)
        report["applied"].append({"key": f"ppo.{name}", "target": f"agent_cfg.{name}", "old": old, "new": value})
        return
    report["unsupported"].append({"key": f"ppo.{name}", "reason": "handled by Modal command or unsupported by agent cfg"})


def _load_patch_values(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {}
    payload = json.loads(Path(path_value).read_text())
    if isinstance(payload, dict) and isinstance(payload.get("patch"), dict):
        return dict(payload["patch"])
    return dict(payload) if isinstance(payload, dict) else {}


def _load_rsl_rl_agent_cfg(args_cli: argparse.Namespace) -> Any | None:
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
        "run_name": args_cli.run_name,
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


def _build_log_dir(agent_cfg: Any, args_cli: argparse.Namespace) -> Path:
    experiment_name = getattr(agent_cfg, "experiment_name", None) or _task_log_name(args_cli.task)
    stamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    return Path.cwd() / "logs" / "rsl_rl" / str(experiment_name) / f"{stamp}_{args_cli.run_name}"


def _task_log_name(task: str) -> str:
    name = task.replace("Isaac-Velocity-", "").replace("-v0", "").lower()
    return name.replace("-", "_")


def _robot_cfg(env_cfg: Any) -> Any | None:
    return getattr(getattr(env_cfg, "scene", None), "robot", None)


def _signed_weight(old: Any, value: Any) -> float:
    new = abs(float(value))
    try:
        old_float = float(old)
    except (TypeError, ValueError):
        old_float = new
    return -new if old_float < 0 else new


def _scale_named_fields(root: Any, names: list[str], scale: float, report: dict[str, Any], key: str) -> int:
    count = 0
    for owner, field, old in _iter_named_fields(root, names):
        new = _scale_value(old, scale)
        if new is None:
            continue
        _set_field(owner, field, new)
        report["applied"].append({"key": key, "target": field, "old": old, "new": new})
        count += 1
    return count


def _set_named_fields(root: Any, names: list[str], value: Any, report: dict[str, Any], key: str) -> int:
    count = 0
    for owner, field, old in _iter_named_fields(root, names):
        _set_field(owner, field, value)
        report["applied"].append({"key": key, "target": field, "old": old, "new": value})
        count += 1
    return count


def _set_terrain_height_noise(terrain_cfg: Any, value: Any, report: dict[str, Any]) -> int:
    """Apply terrain roughness without changing scalar terrain fields to tuples."""

    noise = abs(float(value))
    count = 0
    for owner, field, old in _iter_named_fields(terrain_cfg, ["noise_range", "height_range", "vertical_scale"]):
        if field in {"noise_range", "height_range"}:
            if not isinstance(old, (list, tuple)) or len(old) < 2:
                continue
            new: Any = (-noise, noise)
        elif field == "vertical_scale":
            if not isinstance(old, (int, float)):
                continue
            new = noise
        else:
            continue
        _set_field(owner, field, new)
        report["applied"].append({"key": "terrain.height_noise_m", "target": field, "old": old, "new": new})
        count += 1
    return count


def _set_push_velocity_ranges(env_cfg: Any, velocity_range: dict[str, tuple[float, float]], report: dict[str, Any]) -> int:
    """Apply push/reset velocity only to dict-shaped event velocity ranges."""

    events = getattr(env_cfg, "events", None)
    if events is None:
        return 0
    count = 0
    for owner, field, old in _iter_named_fields(events, ["velocity_range"], max_depth=6):
        if not isinstance(old, dict):
            continue
        if "x" not in old and "y" not in old:
            continue
        new = dict(old)
        for axis, bounds in velocity_range.items():
            if axis in new:
                new[axis] = bounds
        _set_field(owner, field, new)
        report["applied"].append({"key": "domain_randomization.push_force_range_n", "target": field, "old": old, "new": new})
        count += 1
    return count


def _iter_named_fields(root: Any, names: list[str], *, max_depth: int = 5) -> list[tuple[Any, str, Any]]:
    matches: list[tuple[Any, str, Any]] = []
    seen: set[int] = set()

    def visit(obj: Any, depth: int) -> None:
        if depth > max_depth or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, dict):
            for key, child in list(obj.items()):
                if str(key) in names:
                    matches.append((obj, str(key), child))
                visit(child, depth + 1)
            return
        attrs = getattr(obj, "__dict__", {})
        if not isinstance(attrs, dict):
            return
        for key, child in list(attrs.items()):
            if key.startswith("_"):
                continue
            if key in names:
                matches.append((obj, key, child))
            visit(child, depth + 1)

    visit(root, 0)
    return matches


def _set_field(owner: Any, field: str, value: Any) -> None:
    if isinstance(owner, dict):
        owner[field] = value
    else:
        setattr(owner, field, value)


def _scale_value(value: Any, scale: float) -> Any | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) * scale
    if isinstance(value, dict):
        scaled: dict[Any, Any] = {}
        for key, child in value.items():
            scaled_child = _scale_value(child, scale)
            scaled[key] = scaled_child if scaled_child is not None else child
        return scaled
    return None


def _set_group_scale(old: Any, group: str, value: float) -> Any:
    if not isinstance(old, dict):
        return old
    group_tokens = {
        "hip": ["hip"],
        "knee": ["knee"],
        "ankle": ["ankle"],
        "arm": ["shoulder", "elbow"],
        "torso": ["torso"],
    }.get(group, [group])
    new = dict(old)
    changed = False
    for key in list(new):
        lower = str(key).lower()
        if any(token in lower for token in group_tokens):
            new[key] = value
            changed = True
    return new if changed else old


def _range_tuple(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    number = float(value)
    return (-abs(number), abs(number))


def _terrain_noise_value(value: Any) -> Any:
    number = abs(float(value))
    return (-number, number)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
