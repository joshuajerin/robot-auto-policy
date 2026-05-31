"""Render scenario-diverse Isaac Lab videos on a Lambda GPU host.

The tool is intentionally local-orchestration only: Modal trains and stores the
checkpoint, this script copies that checkpoint to a known Lambda host with
working Vulkan, runs the Isaac-side renderer on generated scenarios, and pulls
videos plus diagnostics back into local artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMBDA_HOST = "ubuntu@163.192.35.230"
DEFAULT_SSH_KEY = Path.home() / "niki"
ISAAC_SCRIPT = REPO_ROOT / "modal_runner" / "isaac_scripts" / "render_multiview_rsl_rl_policy.py"
TRAIN_PATCH_SCRIPT = REPO_ROOT / "modal_runner" / "isaac_scripts" / "train_rsl_rl_policy.py"
ROBOGENESIS_TASKS_DIR = REPO_ROOT / "modal_runner" / "isaac_scripts" / "robogenesis_tasks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--volume", default="robogenesis-runs")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--runner", default="rsl_rl")
    parser.add_argument("--experiment-name", default="", help="RSL-RL experiment directory; inferred from checkpoint path by default.")
    parser.add_argument("--lambda-host", default=DEFAULT_LAMBDA_HOST)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--local-output-root", type=Path, default=REPO_ROOT / "artifacts" / "multiview_lambda")
    parser.add_argument("--remote-root", default="/home/ubuntu/robogenesis-runs/multiview")
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--video-length", type=int, default=240)
    parser.add_argument("--views", default="side,diagonal")
    parser.add_argument("--videos-per-scenario", type=int, default=2)
    parser.add_argument("--scenario-file", default="")
    parser.add_argument("--scenario-limit", type=int, default=8)
    parser.add_argument("--config-patch", default="")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    args.ssh_key = args.ssh_key.expanduser()
    return args


def main() -> None:
    args = parse_args()
    local_experiment_dir = args.local_output_root / args.experiment_id / "modal_artifacts"
    local_render_dir = args.local_output_root / args.experiment_id / "lambda_render"
    local_render_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if local_experiment_dir.exists():
            shutil.rmtree(local_experiment_dir)
        local_experiment_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "modal",
                "volume",
                "get",
                "--force",
                args.volume,
                f"/experiments/{args.experiment_id}",
                str(local_experiment_dir),
            ]
        )

    checkpoint_path = select_checkpoint(local_experiment_dir, args.checkpoint)
    config_patch_path = select_config_patch(local_experiment_dir, args.config_patch)
    selected_views = bounded_views(args.views, args.videos_per_scenario)
    base_patch = load_patch_dict(config_patch_path)
    scenarios = load_scenarios(args.scenario_file, local_experiment_dir, limit=args.scenario_limit)
    scenario_jobs = build_scenario_jobs(scenarios, base_patch, selected_views, args)
    experiment_name = args.experiment_name or infer_rsl_experiment_name(checkpoint_path)
    run_dir = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name

    remote_base = f"{args.remote_root}/{args.experiment_id}"
    remote_logs = f"{remote_base}/logs"
    remote_run_dir = f"{remote_logs}/rsl_rl/{experiment_name}/{run_dir}"
    remote_scripts = f"{remote_base}/scripts"
    remote_artifacts = f"{remote_base}/artifacts"

    ssh(
        args,
        (
            f"sudo rm -rf {quote(remote_base)} && "
            f"mkdir -p {quote(remote_run_dir)} {quote(remote_scripts)} {quote(remote_artifacts)} && "
            f"sudo chown -R ubuntu:ubuntu {quote(remote_base)}"
        ),
    )
    scp(args, checkpoint_path, f"{args.lambda_host}:{remote_run_dir}/{checkpoint_name}")
    scp(args, ISAAC_SCRIPT, f"{args.lambda_host}:{remote_scripts}/{ISAAC_SCRIPT.name}")
    scp(args, TRAIN_PATCH_SCRIPT, f"{args.lambda_host}:{remote_scripts}/{TRAIN_PATCH_SCRIPT.name}")
    if ROBOGENESIS_TASKS_DIR.exists():
        scp(args, ROBOGENESIS_TASKS_DIR, f"{args.lambda_host}:{remote_scripts}/robogenesis_tasks", recursive=True)
    config_patch_args: list[str] = []
    if config_patch_path:
        scp(args, config_patch_path, f"{args.lambda_host}:{remote_artifacts}/config_patch.json")
        config_patch_args = ["--config-patch", "/artifacts/config_patch.json"]

    render_commands = build_render_commands(
        scenario_jobs=scenario_jobs,
        selected_views=selected_views,
        args=args,
        experiment_name=experiment_name,
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        config_patch_args=config_patch_args,
    )
    docker_script = f"""
set -euo pipefail
sudo docker run --rm -i --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/dri \\
  -v {quote(remote_logs)}:/workspace/isaaclab/logs \\
  -v {quote(remote_scripts)}:/robogenesis/isaac_scripts \\
  -v {quote(remote_artifacts)}:/artifacts \\
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \\
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \\
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
  --entrypoint /bin/bash nvcr.io/nvidia/isaac-lab:2.0.2 -s <<'DOCKER'
    set -euo pipefail
    cd /workspace/isaaclab
    : > /artifacts/multiview_render.log
{render_commands}
    find /artifacts -type f | sort
DOCKER
"""
    ssh(args, "bash -s", input_text=docker_script)

    if local_render_dir.exists():
        shutil.rmtree(local_render_dir)
    local_render_dir.mkdir(parents=True, exist_ok=True)
    scp(args, f"{args.lambda_host}:{remote_artifacts}/", str(local_render_dir), recursive=True)

    videos = sorted(local_render_dir.rglob("*.mp4"))
    diagnostics_files = [
        path for path in sorted(local_render_dir.rglob("*_diagnostics.json")) if path.name != "multiview_diagnostics.json"
    ]
    combined_diagnostics = combine_diagnostics(diagnostics_files)
    combined_path = local_render_dir / ("scenario_autoresearch_input.json" if scenario_jobs else "multiview_autoresearch_input.json")
    combined_path.write_text(json.dumps(combined_diagnostics, indent=2, sort_keys=True) + "\n")
    legacy_combined_path = local_render_dir / "multiview_autoresearch_input.json"
    if legacy_combined_path != combined_path:
        legacy_combined_path.write_text(json.dumps(combined_diagnostics, indent=2, sort_keys=True) + "\n")
    summary = {
        "experiment_id": args.experiment_id,
        "checkpoint": str(checkpoint_path),
        "rsl_rl_experiment_name": experiment_name,
        "run_dir": run_dir,
        "local_render_dir": str(local_render_dir),
        "diagnostics": str(combined_path),
        "scenario_count": len(scenario_jobs),
        "videos_per_scenario": len(selected_views),
        "per_view_diagnostics": [str(path) for path in diagnostics_files],
        "videos": [str(path) for path in videos],
        "video_probe": [probe_video(path) for path in videos],
    }
    summary_path = local_render_dir / "multiview_handoff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_render_commands(
    *,
    scenario_jobs: list[dict[str, Any]],
    selected_views: list[str],
    args: argparse.Namespace,
    experiment_name: str,
    run_dir: str,
    checkpoint_name: str,
    config_patch_args: list[str],
) -> str:
    lines: list[str] = []
    checkpoint = f"logs/rsl_rl/{experiment_name}/{run_dir}/{checkpoint_name}"

    if scenario_jobs:
        lines.extend(
            [
                "mkdir -p /artifacts/scenario_patches",
                "cat > /artifacts/scenario_jobs.json <<'JSON'",
                json.dumps(scenario_jobs, indent=2, sort_keys=True),
                "JSON",
            ]
        )
        for job in scenario_jobs:
            scenario_id = str(job["scenario_id"])
            patch_path = str(job["patch_path"])
            lines.extend(
                [
                    f"mkdir -p {quote(Path(patch_path).parent)}",
                    f"cat > {quote(patch_path)} <<'JSON'",
                    json.dumps(job["patch"], indent=2, sort_keys=True),
                    "JSON",
                ]
            )
            for view in job["views"]:
                output_dir = f"/artifacts/scenarios/{scenario_id}"
                command = [
                    "./isaaclab.sh",
                    "-p",
                    f"/robogenesis/isaac_scripts/{ISAAC_SCRIPT.name}",
                    "--task",
                    str(job["task"]),
                    "--checkpoint",
                    checkpoint,
                    "--output-dir",
                    output_dir,
                    "--policy-id",
                    args.experiment_id,
                    "--scenario-id",
                    scenario_id,
                    "--runner",
                    args.runner,
                    "--seed",
                    str(job["seed"]),
                    "--video-length",
                    str(int(args.video_length)),
                    "--views",
                    str(view),
                    "--config-patch",
                    patch_path,
                ]
                lines.append(f"echo {quote(f'=== rendering {scenario_id} {view} ===')} | tee -a /artifacts/multiview_render.log")
                lines.append(" ".join(shell_arg(part) for part in command) + " 2>&1 | tee -a /artifacts/multiview_render.log")
                lines.append(f"cp {quote(output_dir + '/multiview_diagnostics.json')} {quote(output_dir + '/' + str(view) + '_diagnostics.json')}")
        return "\n".join(lines)

    lines.append("mkdir -p /artifacts/multiview")
    for view in selected_views:
        command = [
            "./isaaclab.sh",
            "-p",
            f"/robogenesis/isaac_scripts/{ISAAC_SCRIPT.name}",
            "--task",
            args.task,
            "--checkpoint",
            checkpoint,
            "--output-dir",
            "/artifacts/multiview",
            "--policy-id",
            args.experiment_id,
            "--runner",
            args.runner,
            "--seed",
            str(int(args.seed)),
            "--video-length",
            str(int(args.video_length)),
            "--views",
            view,
            *config_patch_args,
        ]
        lines.append(f"echo {quote(f'=== rendering {view} ===')} | tee -a /artifacts/multiview_render.log")
        lines.append(" ".join(shell_arg(part) for part in command) + " 2>&1 | tee -a /artifacts/multiview_render.log")
        lines.append(f"cp /artifacts/multiview/multiview_diagnostics.json /artifacts/multiview/{view}_diagnostics.json")
    return "\n".join(lines)


def bounded_views(views_arg: str, max_videos: int) -> list[str]:
    available = {"front", "side", "diagonal"}
    requested = [view.strip() for view in views_arg.split(",") if view.strip()]
    views = [view for view in requested if view in available]
    if not views:
        views = ["side", "diagonal"]
    limit = max(1, int(max_videos or 1))
    return views[:limit]


def load_patch_dict(config_patch_path: Path | None) -> dict[str, Any]:
    if not config_patch_path or not config_patch_path.exists():
        return {}
    payload = json.loads(config_patch_path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("patch"), dict):
        return dict(payload["patch"])
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def load_scenarios(scenario_file: str, local_experiment_dir: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    candidates: list[Path] = []
    if scenario_file:
        path = Path(scenario_file)
        candidates.append(path if path.is_absolute() else REPO_ROOT / path)
    else:
        candidates.extend(sorted(local_experiment_dir.rglob("generated_scenarios.json")))
        candidates.extend(sorted(local_experiment_dir.rglob("scenario_bank.json")))
        candidates.extend(sorted(local_experiment_dir.rglob("experiment_spec.json")))
        candidates.extend(sorted(local_experiment_dir.rglob("*context.json")))

    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        for scenario in extract_scenarios(json.loads(path.read_text())):
            scenario_id = str(scenario.get("scenario_id") or "")
            if not scenario_id or scenario_id in seen:
                continue
            seen.add(scenario_id)
            scenarios.append(scenario)
            if len(scenarios) >= max(0, limit):
                return scenarios
    return scenarios


def extract_scenarios(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict) and item.get("scenario_id")]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("generated_scenarios"), list):
        return extract_scenarios(payload["generated_scenarios"])
    autoresearch = payload.get("autoresearch")
    if isinstance(autoresearch, dict) and isinstance(autoresearch.get("generated_scenarios"), list):
        return extract_scenarios(autoresearch["generated_scenarios"])
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        return extract_scenarios(scenarios)
    return []


def build_scenario_jobs(
    scenarios: list[dict[str, Any]],
    base_patch: dict[str, Any],
    selected_views: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        scenario_id = safe_name(str(scenario.get("scenario_id") or f"scenario_{index:03d}"))
        patch = {**strip_environment_patch(base_patch), **scenario_to_patch(scenario)}
        jobs.append(
            {
                "scenario_id": scenario_id,
                "difficulty": scenario.get("difficulty"),
                "task": scenario_to_task(args.task, scenario),
                "seed": int(args.seed) + index,
                "views": selected_views,
                "patch_path": f"/artifacts/scenario_patches/{scenario_id}.json",
                "patch": patch,
                "scenario": scenario,
            }
        )
    return jobs


def strip_environment_patch(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in patch.items()
        if not (key.startswith("terrain.") or key.startswith("domain_randomization."))
    }


def scenario_to_patch(scenario: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    terrain = scenario.get("terrain") if isinstance(scenario.get("terrain"), dict) else {}
    disturbances = scenario.get("disturbances") if isinstance(scenario.get("disturbances"), dict) else {}
    robot_variation = scenario.get("robot_variation") if isinstance(scenario.get("robot_variation"), dict) else {}

    if terrain.get("type"):
        terrain_type = "rough" if terrain["type"] == "slope" else terrain["type"]
        patch["terrain.type"] = terrain_type
        if terrain_type == "rough":
            patch["terrain.rough_heightfield_enabled"] = True
        elif terrain_type == "flat":
            patch["terrain.rough_heightfield_enabled"] = False
            patch["terrain.height_noise_m"] = 0.0
            patch["terrain.slope_range_deg"] = [0.0, 0.0]
    if "height_noise_m" in terrain:
        patch["terrain.height_noise_m"] = terrain["height_noise_m"]
    if "slope_range_deg" in terrain:
        patch["terrain.slope_range_deg"] = terrain["slope_range_deg"]
    if "friction_range" in terrain:
        patch["domain_randomization.friction_range"] = terrain["friction_range"]

    for key in ("push_force_range_n", "push_impulse_probability"):
        if key in disturbances:
            patch[f"domain_randomization.{key}"] = disturbances[key]
    for key in ("motor_strength_scale", "action_delay_steps", "payload_mass_kg"):
        if key in robot_variation:
            patch[f"domain_randomization.{key}"] = robot_variation[key]
    return patch


def scenario_to_task(base_task: str, scenario: dict[str, Any]) -> str:
    terrain = scenario.get("terrain") if isinstance(scenario.get("terrain"), dict) else {}
    rough_requested = (
        terrain.get("type") in {"rough", "slope"}
        or float(terrain.get("height_noise_m") or 0.0) > 0.0
        or "slope_range_deg" in terrain
    )
    if rough_requested and base_task == "Isaac-Velocity-Flat-H1-v0":
        return "Isaac-Velocity-Rough-H1-v0"
    return base_task


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")
    return safe or "scenario"


def combine_diagnostics(paths: list[Path]) -> dict[str, object]:
    reports = []
    for path in paths:
        try:
            reports.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    views = []
    diagnoses = []
    for report in reports:
        views.extend(report.get("views", []))
        if report.get("diagnosis"):
            diagnoses.append(report["diagnosis"])
    primary_failures = [str(item.get("primary_failure", "")) for item in diagnoses if item.get("primary_failure")]
    scenario_ids = sorted({str(view.get("scenario_id")) for view in views if view.get("scenario_id")})
    return {
        "view_count": len(views),
        "scenario_count": len(scenario_ids),
        "scenario_ids": scenario_ids,
        "views": views,
        "diagnoses": diagnoses,
        "primary_failures": primary_failures,
        "aggregate": {
            "any_done": any(view.get("done_step") is not None for view in views),
            "max_torso_tilt_xy": max((float(view.get("max_torso_tilt_xy", 0.0)) for view in views), default=0.0),
            "mean_command_error_xy": (
                sum(float(view.get("mean_command_error_xy", 0.0)) for view in views) / len(views) if views else 0.0
            ),
            "mean_action_jerk": (
                sum(float(view.get("mean_action_jerk", 0.0)) for view in views) / len(views) if views else 0.0
            ),
        },
    }


def select_checkpoint(local_experiment_dir: Path, checkpoint: str) -> Path:
    checkpoints = sorted(local_experiment_dir.rglob("model_*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise SystemExit(f"No model_*.pt checkpoints found under {local_experiment_dir}")
    if checkpoint and checkpoint != "latest":
        exact = [path for path in checkpoints if path.name == checkpoint]
        if not exact:
            raise SystemExit(f"Checkpoint {checkpoint} not found under {local_experiment_dir}")
        return exact[-1]
    return checkpoints[-1]


def select_config_patch(local_experiment_dir: Path, config_patch: str) -> Path | None:
    if config_patch:
        path = Path(config_patch)
        return path if path.is_absolute() else REPO_ROOT / path
    candidates = sorted(local_experiment_dir.rglob("isaac_training_patch.json"))
    return candidates[-1] if candidates else None


def infer_rsl_experiment_name(checkpoint_path: Path) -> str:
    parts = checkpoint_path.parts
    for index, part in enumerate(parts):
        if part == "rsl_rl" and index + 1 < len(parts):
            return parts[index + 1]
    return "h1_flat"


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1]), str(path)
    except Exception:
        return -1, str(path)


def probe_video(path: Path) -> dict[str, str]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"path": str(path), "error": proc.stderr.strip()}
    payload = json.loads(proc.stdout)
    stream = payload.get("streams", [{}])[0]
    return {"path": str(path), **{key: str(value) for key, value in stream.items()}}


def ssh(args: argparse.Namespace, command: str, *, input_text: str | None = None) -> None:
    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            str(args.ssh_key),
            args.lambda_host,
            command,
        ],
        input_text=input_text,
    )


def scp(args: argparse.Namespace, source: Path | str, destination: str, *, recursive: bool = False) -> None:
    command = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        str(args.ssh_key),
    ]
    if recursive:
        command.append("-r")
    command.extend([str(source), destination])
    run(command)


def run(command: list[str], *, input_text: str | None = None) -> None:
    print("$ " + " ".join(shell_arg(part) for part in command), flush=True)
    subprocess.run(command, input=input_text, text=True, check=True)


def shell_arg(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def quote(value: object) -> str:
    return shell_arg(value)


if __name__ == "__main__":
    main()
