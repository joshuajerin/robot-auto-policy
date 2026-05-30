"""Render front/side/diagonal Isaac Lab videos on a Lambda GPU host.

The tool is intentionally local-orchestration only: Modal trains and stores the
checkpoint, this script copies that checkpoint to a known Lambda host with
working Vulkan, runs the Isaac-side multiview renderer, and pulls videos plus
diagnostics back into local artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMBDA_HOST = "ubuntu@163.192.35.230"
DEFAULT_SSH_KEY = Path.home() / "niki"
ISAAC_SCRIPT = REPO_ROOT / "modal_runner" / "isaac_scripts" / "render_multiview_rsl_rl_policy.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--volume", default="robogenesis-runs")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--runner", default="rsl_rl")
    parser.add_argument("--lambda-host", default=DEFAULT_LAMBDA_HOST)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--local-output-root", type=Path, default=REPO_ROOT / "artifacts" / "multiview_lambda")
    parser.add_argument("--remote-root", default="/home/ubuntu/robogenesis-runs/multiview")
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--video-length", type=int, default=240)
    parser.add_argument("--views", default="front,side,diagonal")
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
    run_dir = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name

    remote_base = f"{args.remote_root}/{args.experiment_id}"
    remote_logs = f"{remote_base}/logs"
    remote_run_dir = f"{remote_logs}/rsl_rl/h1_flat/{run_dir}"
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

    view_args = " ".join(args.views.split(","))
    docker_script = f"""
set -euo pipefail
sudo docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/dri \\
  -v {quote(remote_logs)}:/workspace/isaaclab/logs \\
  -v {quote(remote_scripts)}:/robogenesis/isaac_scripts \\
  -v {quote(remote_artifacts)}:/artifacts \\
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \\
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \\
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
  --entrypoint /bin/bash nvcr.io/nvidia/isaac-lab:2.0.2 -lc '
    set -euo pipefail
    cd /workspace/isaaclab
    : > /artifacts/multiview_render.log
    for VIEW in {view_args}; do
      echo "=== rendering $VIEW ===" | tee -a /artifacts/multiview_render.log
      ./isaaclab.sh -p /robogenesis/isaac_scripts/{ISAAC_SCRIPT.name} \\
        --task {args.task} \\
        --checkpoint logs/rsl_rl/h1_flat/{run_dir}/{checkpoint_name} \\
        --output-dir /artifacts/multiview \\
        --policy-id {args.experiment_id} \\
        --runner {args.runner} \\
        --seed {int(args.seed)} \\
        --video-length {int(args.video_length)} \\
        --views $VIEW \\
        2>&1 | tee -a /artifacts/multiview_render.log
      cp /artifacts/multiview/multiview_diagnostics.json /artifacts/multiview/${{VIEW}}_diagnostics.json
    done
    find /artifacts -type f | sort
  '
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
    combined_path = local_render_dir / "multiview_autoresearch_input.json"
    combined_path.write_text(json.dumps(combined_diagnostics, indent=2, sort_keys=True) + "\n")
    summary = {
        "experiment_id": args.experiment_id,
        "checkpoint": str(checkpoint_path),
        "run_dir": run_dir,
        "local_render_dir": str(local_render_dir),
        "diagnostics": str(combined_path),
        "per_view_diagnostics": [str(path) for path in diagnostics_files],
        "videos": [str(path) for path in videos],
        "video_probe": [probe_video(path) for path in videos],
    }
    summary_path = local_render_dir / "multiview_handoff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


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
    return {
        "view_count": len(views),
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
