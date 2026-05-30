"""Modal app for Isaac Lab training and evaluation."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import modal


APP_NAME = "robogenesis-isaac-autoresearch"
ISAAC_LAB_IMAGE = "nvcr.io/nvidia/isaac-lab:2.0.2"
DEFAULT_GPU = "A10G"
CPU_COUNT = 16.0
MEMORY_MB = 131_072
TIMEOUT_SECONDS = 6 * 60 * 60
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SCRIPT_ROOT = Path("/robogenesis/isaac_scripts")
ISAAC_LAB_ROOT_CANDIDATES = (
    Path("/workspace/IsaacLab"),
    Path("/workspace/isaaclab"),
    Path("/isaac-lab"),
    Path("/root/IsaacLab"),
)


app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(ISAAC_LAB_IMAGE, add_python="3.10")
    .entrypoint([])
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "PYTHONUNBUFFERED": "1"})
)
try:
    image = image.add_local_dir(REPO_ROOT / "modal_runner" / "isaac_scripts", remote_path=str(REMOTE_SCRIPT_ROOT))
except AttributeError:
    # Older Modal clients may not expose add_local_dir at import time. The
    # scripts can still be baked into a custom image or copied manually.
    pass

runs_volume = modal.Volume.from_name("robogenesis-runs", create_if_missing=True)
volumes = {"/runs": runs_volume}


def _isaac_lab_root() -> Path:
    for root in ISAAC_LAB_ROOT_CANDIDATES:
        if (root / "isaaclab.sh").exists():
            return root
    raise FileNotFoundError("Could not locate isaaclab.sh in the Isaac Lab image")


def _run(cmd: list[str], cwd: Path | None = None, ok_codes: tuple[int, ...] = (0,)) -> int:
    cwd = cwd or _isaac_lab_root()
    print(f"$ {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc not in ok_codes:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return safe.strip(".-_") or "experiment"


def _sync_artifacts(experiment_id: str) -> str:
    root = _isaac_lab_root()
    output_root = Path("/runs") / "experiments" / _safe_name(experiment_id)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "data_storage"):
        src = root / name
        if src.exists():
            shutil.copytree(src, output_root / name, dirs_exist_ok=True)
    runs_volume.commit()
    return str(output_root)


def _artifact_root(experiment_id: str) -> Path:
    root = Path("/runs") / "experiments" / _safe_name(experiment_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _with_timeout(cmd: list[str], seconds: int) -> list[str]:
    return ["timeout", str(seconds), *cmd] if seconds > 0 else cmd


def _find_checkpoint(experiment_id: str, checkpoint: str = "latest") -> Path:
    logs_root = _isaac_lab_root() / "logs" / "rsl_rl"
    candidates = sorted(logs_root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {logs_root}")

    scoped = [path for path in candidates if _safe_name(experiment_id) in str(path)]
    if scoped:
        candidates = scoped

    if checkpoint and checkpoint != "latest":
        exact = [path for path in candidates if path.name == checkpoint]
        if exact:
            return sorted(exact)[-1]
        raise FileNotFoundError(f"Checkpoint {checkpoint} not found under {logs_root}")

    return sorted(candidates, key=_checkpoint_sort_key)[-1]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, str(path))


def _find_video(artifact_root: Path) -> str | None:
    videos = sorted(artifact_root.rglob("*.mp4"))
    return str(videos[-1]) if videos else None


def _score_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    safety_reasons: list[str] = []
    if float(raw_metrics.get("fall_rate", 0.0)) > 0.35:
        safety_reasons.append("fall rate too high")
    if bool(raw_metrics.get("nan_actions", False)):
        safety_reasons.append("policy produced NaN actions")
    if float(raw_metrics.get("joint_limit_violation_rate", 0.0)) > 0.02:
        safety_reasons.append("joint limit violation rate too high")

    raw_metrics = dict(raw_metrics)
    raw_metrics["safety_passed"] = not safety_reasons
    if safety_reasons:
        raw_metrics["safety_penalty"] = max(float(raw_metrics.get("safety_penalty", 0.0)), 0.2)

    command_tracking = _clamp01(raw_metrics.get("command_tracking"))
    survival_no_fall = _clamp01(raw_metrics.get("survival_no_fall"))
    stability = _clamp01(raw_metrics.get("stability"))
    generated_scenario_success = _clamp01(raw_metrics.get("generated_scenario_success"))
    gait_quality = _clamp01(raw_metrics.get("gait_quality"))
    energy_efficiency = _clamp01(raw_metrics.get("energy_efficiency"))
    smoothness = _clamp01(raw_metrics.get("smoothness"))
    recovery_from_disturbance = _clamp01(raw_metrics.get("recovery_from_disturbance"))
    safety_penalty = _clamp01(raw_metrics.get("safety_penalty"))
    regression_penalty = _clamp01(raw_metrics.get("regression_penalty"))
    total_score = (
        0.20 * command_tracking
        + 0.20 * survival_no_fall
        + 0.15 * stability
        + 0.15 * generated_scenario_success
        + 0.10 * gait_quality
        + 0.10 * energy_efficiency
        + 0.05 * smoothness
        + 0.05 * recovery_from_disturbance
        - safety_penalty
        - regression_penalty
    )
    return {
        **raw_metrics,
        "total_score": max(0.0, min(1.0, total_score)),
        "safety_reasons": safety_reasons,
    }


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


@app.function(image=image, gpu=DEFAULT_GPU, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=30 * 60, volumes=volumes)
def smoke_test() -> str:
    _run(["nvidia-smi"], cwd=Path("/"))
    root = _isaac_lab_root()
    _run(["bash", "-lc", "./isaaclab.sh --help | head -80"], cwd=root)
    return f"Isaac Lab smoke test passed at {root}"


@app.function(image=image, gpu=DEFAULT_GPU, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes=volumes)
def train_and_eval_job(experiment_spec_json: str) -> dict[str, Any]:
    spec = json.loads(experiment_spec_json)
    experiment_id = _safe_name(str(spec.get("experiment_id", "experiment")))
    task = str(spec.get("task", "Isaac-Velocity-Flat-H1-v0"))
    runner = str(spec.get("runner", "rsl_rl"))
    num_envs = int(spec.get("num_envs", 4096))
    max_iterations = int(spec.get("max_iterations", 1000))
    seed = spec.get("seed")

    train_script = f"scripts/reinforcement_learning/{runner}/train.py"
    cmd = [
        "./isaaclab.sh",
        "-p",
        train_script,
        "--task",
        task,
        "--headless",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--run_name",
        experiment_id,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    _run(cmd)
    artifact_root = _sync_artifacts(experiment_id)
    return {
        "experiment_id": experiment_id,
        "task": task,
        "runner": runner,
        "artifact_root": artifact_root,
        "metrics": _load_metrics_if_present(Path(artifact_root), experiment_id),
    }


@app.function(image=image, gpu=DEFAULT_GPU, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=24 * 60 * 60, volumes=volumes)
def phase1_baseline_job(experiment_spec_json: str) -> dict[str, Any]:
    """Run the full phase-1 H1 baseline: train, evaluate, render, manifest."""

    spec = json.loads(experiment_spec_json)
    experiment_id = _safe_name(str(spec.get("experiment_id", "baseline_h1_001")))
    task = str(spec.get("task", "Isaac-Velocity-Flat-H1-v0"))
    runner = str(spec.get("runner", "rsl_rl"))
    device = str(spec.get("device", "cuda:0"))
    train_spec = dict(spec.get("train", {}))
    eval_spec = dict(spec.get("eval", {}))
    render_spec = dict(spec.get("render", {}))
    artifact_root = _artifact_root(experiment_id)

    (artifact_root / "experiment_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    if spec.get("style_context"):
        (artifact_root / "style_context.json").write_text(
            json.dumps(spec["style_context"], indent=2, sort_keys=True) + "\n"
        )

    h1_report_path = artifact_root / "h1_asset_report.json"
    _run(
        [
            "python",
            str(REMOTE_SCRIPT_ROOT / "inspect_h1_asset.py"),
            "--task",
            task,
            "--output",
            str(h1_report_path),
        ],
        cwd=Path("/"),
    )

    train_script = f"scripts/reinforcement_learning/{runner}/train.py"
    train_cmd = [
        "./isaaclab.sh",
        "-p",
        train_script,
        "--task",
        task,
        "--headless",
        "--num_envs",
        str(int(train_spec.get("num_envs", spec.get("num_envs", 4096)))),
        "--max_iterations",
        str(int(train_spec.get("max_iterations", spec.get("max_iterations", 1000)))),
        "--run_name",
        experiment_id,
    ]
    seed = train_spec.get("seed", spec.get("seed"))
    if seed is not None:
        train_cmd.extend(["--seed", str(seed)])
    if bool(train_spec.get("video", False)):
        train_cmd.extend(["--video", "--video_length", "120", "--video_interval", "1", "--enable_cameras"])

    train_timeout = int(train_spec.get("command_timeout_seconds", 0))
    _run(_with_timeout(train_cmd, train_timeout), ok_codes=(0, 124) if train_timeout > 0 else (0,))

    checkpoint_path = _find_checkpoint(experiment_id, str(render_spec.get("checkpoint", "latest")))
    print(f"Selected checkpoint: {checkpoint_path}", flush=True)

    raw_metrics_path = artifact_root / "raw_eval_metrics.json"
    eval_cmd = [
        "./isaaclab.sh",
        "-p",
        str(REMOTE_SCRIPT_ROOT / "evaluate_rsl_rl_policy.py"),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(raw_metrics_path),
        "--num-envs",
        str(int(eval_spec.get("num_envs", 32))),
        "--episodes-per-seed",
        str(int(eval_spec.get("episodes_per_seed", 4))),
        "--max-steps-per-episode",
        str(int(eval_spec.get("max_steps_per_episode", 1000))),
        "--seeds",
        ",".join(str(seed) for seed in eval_spec.get("seeds", [101, 203, 307, 409, 503, 601, 709, 811])),
        "--device",
        device,
        "--policy-id",
        experiment_id,
    ]
    _run(eval_cmd)

    raw_metrics = json.loads(raw_metrics_path.read_text())
    score = _score_metrics(raw_metrics)
    score_path = artifact_root / "eval_metrics.json"
    score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")

    render_cmd = [
        "./isaaclab.sh",
        "-p",
        f"scripts/reinforcement_learning/{runner}/play.py",
        "--task",
        task,
        "--headless",
        "--num_envs",
        str(int(render_spec.get("num_envs", 1))),
        "--load_run",
        checkpoint_path.parent.name,
        "--checkpoint",
        checkpoint_path.name,
        "--video",
        "--video_length",
        str(int(render_spec.get("video_length", 240))),
        "--enable_cameras",
    ]
    render_seed = render_spec.get("seed")
    if render_seed is not None:
        render_cmd.extend(["--seed", str(render_seed)])
    _run(_with_timeout(render_cmd, int(render_spec.get("command_timeout_seconds", 900))), ok_codes=(0, 124))

    synced_root = Path(_sync_artifacts(experiment_id))
    rollout_video_path = _find_video(synced_root)
    manifest_path = synced_root / "artifact_manifest.json"
    _run(
        [
            "python",
            str(REMOTE_SCRIPT_ROOT / "write_artifact_manifest.py"),
            "--experiment-id",
            experiment_id,
            "--artifact-root",
            str(synced_root),
            "--task",
            task,
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(raw_metrics_path),
            "--score",
            str(score_path),
            "--video",
            rollout_video_path or "",
            "--h1-asset-report",
            str(h1_report_path),
            "--output",
            str(manifest_path),
        ],
        cwd=Path("/"),
    )
    runs_volume.commit()
    return json.loads(manifest_path.read_text())


def _load_metrics_if_present(artifact_root: Path, experiment_id: str) -> dict[str, Any]:
    metrics_path = artifact_root / "eval_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {
        "policy_id": experiment_id,
        "eval_seed_count": 0,
        "note": "Training artifacts synced; run evaluate.py or attach Isaac evaluator metrics.",
    }


@app.local_entrypoint()
def main(action: str = "smoke", experiment_spec_json: str = "{}") -> None:
    if action == "smoke":
        print(smoke_test.remote())
        return
    if action == "train-and-eval":
        print(json.dumps(train_and_eval_job.remote(experiment_spec_json), indent=2, sort_keys=True))
        return
    if action == "phase1":
        print(json.dumps(phase1_baseline_job.remote(experiment_spec_json), indent=2, sort_keys=True))
        return
    raise ValueError("action must be 'smoke', 'train-and-eval', or 'phase1'")
