"""Modal app for Isaac Lab training and evaluation."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import modal


APP_NAME = "robogenesis-isaac-autoresearch"
ISAAC_LAB_IMAGE = "nvcr.io/nvidia/isaac-lab:2.0.2"
DEFAULT_GPU = "H100"
RENDER_GPU = "RTX-PRO-6000"
CPU_COUNT = 32.0
MEMORY_MB = 262_144
RENDER_CPU_COUNT = 16.0
RENDER_MEMORY_MB = 131_072
MAX_PHASE1_CONTAINERS = 4
MAX_RENDER_CONTAINERS = 2
SCALEDOWN_WINDOW_SECONDS = 300
TIMEOUT_SECONDS = 6 * 60 * 60
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SCRIPT_ROOT = Path("/robogenesis/isaac_scripts")
REMOTE_H1_ASSET_ROOT = Path("/robogenesis/assets/unitree_h1")
H1_TABLETOP_TASK_PREFIX = "RoboGenesis-H1-Tabletop"
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
    .pip_install("matplotlib>=3.8", "imageio>=2.33", "imageio-ffmpeg>=0.4", "numpy>=1.24")
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "PYTHONUNBUFFERED": "1"})
)
try:
    image = image.add_local_dir(REPO_ROOT / "modal_runner" / "isaac_scripts", remote_path=str(REMOTE_SCRIPT_ROOT))
except AttributeError:
    # Older Modal clients may not expose add_local_dir at import time. The
    # scripts can still be baked into a custom image or copied manually.
    pass
if (REPO_ROOT / "assets" / "unitree_h1").exists():
    try:
        image = image.add_local_dir(REPO_ROOT / "assets" / "unitree_h1", remote_path=str(REMOTE_H1_ASSET_ROOT))
    except AttributeError:
        pass

runs_volume = modal.Volume.from_name("robogenesis-runs", create_if_missing=True)
volumes = {"/runs": runs_volume}


def _isaac_lab_root() -> Path:
    for root in ISAAC_LAB_ROOT_CANDIDATES:
        if (root / "isaaclab.sh").exists():
            return root
    raise FileNotFoundError("Could not locate isaaclab.sh in the Isaac Lab image")


def _is_h1_tabletop_task(task: str) -> bool:
    return task.startswith(H1_TABLETOP_TASK_PREFIX)


def _rsl_rl_train_script(task: str, runner: str) -> str:
    if runner == "rsl_rl" and _is_h1_tabletop_task(task):
        return str(REMOTE_SCRIPT_ROOT / "train_h1_tabletop_transfer.py")
    return f"scripts/reinforcement_learning/{runner}/train.py"


def _rsl_rl_play_script(task: str, runner: str) -> str:
    if runner == "rsl_rl" and _is_h1_tabletop_task(task):
        return str(REMOTE_SCRIPT_ROOT / "play_h1_tabletop_transfer.py")
    return f"scripts/reinforcement_learning/{runner}/play.py"


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


def _reload_runs_volume() -> None:
    try:
        runs_volume.reload()
    except AttributeError:
        pass


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


def _checkpoint_from_train_report(artifact_root: Path) -> Path | None:
    report = _load_json_if_present(artifact_root / "isaac_train_report.json")
    if not report:
        return None
    checkpoint = report.get("final_checkpoint")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    return path if path.exists() else None


def _restore_logs_from_artifact(experiment_id: str) -> None:
    artifact_logs = _artifact_root(experiment_id) / "logs"
    if not artifact_logs.exists():
        raise FileNotFoundError(f"No synced logs found for {experiment_id} under {artifact_logs}")
    isaac_logs = _isaac_lab_root() / "logs"
    shutil.copytree(artifact_logs, isaac_logs, dirs_exist_ok=True)


def _find_isaac_camera_videos(artifact_root: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(artifact_root.rglob("*.mp4"))
        if path.name != "rollout_telemetry.mp4"
    ]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, str(path))


def _find_videos(artifact_root: Path) -> list[str]:
    return [str(path) for path in sorted(artifact_root.rglob("*.mp4"))]


def _find_video(artifact_root: Path) -> str | None:
    videos = _find_videos(artifact_root)
    return videos[-1] if videos else None


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _now_ms() -> int:
    return int(time.time() * 1000)


class _RaindropTaskManifest:
    """Persist Modal-side stage timing so local artifact sync can publish it."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        experiment_id: str,
        event_name: str,
        source: str,
        metadata: dict[str, Any],
    ):
        self.path = artifact_root / "raindrop_trace.json"
        self.experiment_id = experiment_id
        self.event_name = event_name
        self.source = source
        self.metadata = metadata
        self.started_ms = _now_ms()
        self.tasks: list[dict[str, Any]] = []
        existing = _load_json_if_present(self.path)
        if existing:
            self.started_ms = int(existing.get("started_ms") or self.started_ms)
            self.tasks = [task for task in existing.get("tasks", []) if isinstance(task, dict)]

    @contextmanager
    def step(self, name: str, input_payload: dict[str, Any] | None = None):
        start_ms = _now_ms()
        output: dict[str, Any] = {}
        try:
            yield output
        except BaseException as exc:
            self.record(
                name,
                input_payload=input_payload,
                output_payload=output,
                start_ms=start_ms,
                end_ms=_now_ms(),
                status="error",
                error=str(exc),
            )
            self.write(status="error")
            raise
        status = str(output.get("status", "ok"))
        error = output.get("error") or output.get("render_error")
        self.record(
            name,
            input_payload=input_payload,
            output_payload=output,
            start_ms=start_ms,
            end_ms=_now_ms(),
            status=status,
            error=str(error) if status == "error" and error is not None else None,
        )
        self.write(status="running")

    def record(
        self,
        name: str,
        *,
        input_payload: Any | None,
        output_payload: Any | None,
        start_ms: int,
        end_ms: int,
        status: str,
        error: str | None = None,
    ) -> None:
        task = {
            "name": name,
            "input": input_payload,
            "output": output_payload,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": max(0, end_ms - start_ms),
            "status": status,
        }
        if error:
            task["error"] = error
        self.tasks.append(task)

    def write(
        self,
        *,
        status: str,
        video_paths: list[str] | None = None,
        primary_video_path: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "version": 1,
            "event_name": self.event_name,
            "event_id": self.experiment_id,
            "experiment_id": self.experiment_id,
            "source": self.source,
            "metadata": self.metadata,
            "status": status,
            "started_ms": self.started_ms,
            "ended_ms": _now_ms(),
            "tasks": self.tasks,
            "video_paths": video_paths or [],
            "primary_video_path": primary_video_path,
            "summary": summary or {},
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _score_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    if raw_metrics.get("metric_family") == "manipulation":
        collision_rate = _clamp01(raw_metrics.get("collision_rate", raw_metrics.get("fall_rate", 0.0)))
        force_violation_rate = _clamp01(raw_metrics.get("force_violation_rate", 0.0))
        safety_reasons: list[str] = []
        if collision_rate > 0.35:
            safety_reasons.append("collision or fall rate too high")
        if force_violation_rate > 0.1:
            safety_reasons.append("force violation rate too high")
        if raw_metrics.get("evaluation_errors"):
            safety_reasons.append("evaluation emitted errors")
        if bool(raw_metrics.get("nan_actions", False)):
            safety_reasons.append("policy produced NaN actions")

        task_success = _clamp01(raw_metrics.get("task_success_rate", raw_metrics.get("success_rate", 0.0)))
        task_progress = _clamp01(raw_metrics.get("task_progress", task_success))
        contact_stability = _clamp01(raw_metrics.get("contact_stability", raw_metrics.get("contact_success_rate", 0.0)))
        placement_accuracy = _clamp01(raw_metrics.get("placement_accuracy", 0.0))
        generated_scenario_success = _clamp01(raw_metrics.get("generated_scenario_success", 0.0))
        energy_efficiency = _clamp01(raw_metrics.get("energy_efficiency", 1.0))
        smoothness = _clamp01(raw_metrics.get("smoothness", 1.0))
        recovery_from_disturbance = _clamp01(raw_metrics.get("recovery_from_disturbance", 0.0))
        safety_penalty = _clamp01(raw_metrics.get("safety_penalty", max(collision_rate, force_violation_rate) * 0.25))
        regression_penalty = _clamp01(raw_metrics.get("regression_penalty", 0.0))
        total_score = (
            0.25 * task_success
            + 0.15 * task_progress
            + 0.15 * contact_stability
            + 0.15 * placement_accuracy
            + 0.10 * generated_scenario_success
            + 0.08 * energy_efficiency
            + 0.07 * smoothness
            + 0.05 * recovery_from_disturbance
            - safety_penalty
            - regression_penalty
        )
        return {
            **raw_metrics,
            "command_tracking": task_progress,
            "survival_no_fall": task_success,
            "base_success": _clamp01(raw_metrics.get("base_success", task_success)),
            "stability": contact_stability,
            "gait_quality": placement_accuracy,
            "generated_scenario_success": generated_scenario_success,
            "energy_efficiency": energy_efficiency,
            "smoothness": smoothness,
            "recovery_from_disturbance": recovery_from_disturbance,
            "safety_passed": not safety_reasons,
            "safety_reasons": safety_reasons,
            "safety_penalty": safety_penalty,
            "regression_penalty": regression_penalty,
            "total_score": max(0.0, min(1.0, total_score)),
        }

    safety_reasons: list[str] = []
    if float(raw_metrics.get("fall_rate", 0.0)) > 0.35:
        safety_reasons.append("fall rate too high")
    if raw_metrics.get("evaluation_errors"):
        safety_reasons.append("evaluation emitted errors")
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


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=30 * 60,
    volumes=volumes,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
def smoke_test() -> str:
    _run(["nvidia-smi"], cwd=Path("/"))
    root = _isaac_lab_root()
    _run(["bash", "-lc", "./isaaclab.sh --help | head -80"], cwd=root)
    return f"Isaac Lab smoke test passed at {root}"


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes=volumes,
    min_containers=0,
    max_containers=MAX_PHASE1_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
def train_and_eval_job(experiment_spec_json: str) -> dict[str, Any]:
    spec = json.loads(experiment_spec_json)
    experiment_id = _safe_name(str(spec.get("experiment_id", "experiment")))
    task = str(spec.get("task", "Isaac-Velocity-Flat-H1-v0"))
    runner = str(spec.get("runner", "rsl_rl"))
    num_envs = int(spec.get("num_envs", 4096))
    max_iterations = int(spec.get("max_iterations", 1000))
    seed = spec.get("seed")

    train_script = _rsl_rl_train_script(task, runner)
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


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=24 * 60 * 60,
    volumes=volumes,
    min_containers=0,
    max_containers=MAX_PHASE1_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
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
    task_manifest = _RaindropTaskManifest(
        artifact_root,
        experiment_id=experiment_id,
        event_name="robogenesis-sim-run",
        source="modal.phase1_baseline_job",
        metadata={"task": task, "runner": runner, "device": device},
    )
    task_manifest.write(status="running")

    (artifact_root / "experiment_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    training_patch_path: Path | None = None
    if spec.get("style_context"):
        (artifact_root / "style_context.json").write_text(
            json.dumps(spec["style_context"], indent=2, sort_keys=True) + "\n"
        )
    if spec.get("motion_context"):
        (artifact_root / "motion_context.json").write_text(
            json.dumps(spec["motion_context"], indent=2, sort_keys=True) + "\n"
        )
    if spec.get("autoresearch"):
        autoresearch_payload = spec["autoresearch"]
        (artifact_root / "autoresearch_context.json").write_text(
            json.dumps(autoresearch_payload, indent=2, sort_keys=True) + "\n"
        )
        (artifact_root / "generated_scenarios.json").write_text(
            json.dumps(autoresearch_payload.get("generated_scenarios", []), indent=2, sort_keys=True) + "\n"
        )
        (artifact_root / "training_surface.json").write_text(
            json.dumps(autoresearch_payload.get("training_surface", {}), indent=2, sort_keys=True) + "\n"
        )
        if isinstance(autoresearch_payload.get("patch"), dict):
            training_patch_path = artifact_root / "isaac_training_patch.json"
            training_patch_path.write_text(
                json.dumps(autoresearch_payload["patch"], indent=2, sort_keys=True) + "\n"
            )

    h1_report_path = artifact_root / "h1_asset_report.json"
    h1_export_dir = artifact_root / "h1_asset_export"
    inspect_cmd = [
        "./isaaclab.sh",
        "-p",
        str(REMOTE_SCRIPT_ROOT / "inspect_h1_asset.py"),
        "--task",
        task,
        "--output",
        str(h1_report_path),
        "--bundled-asset-dir",
        str(REMOTE_H1_ASSET_ROOT),
        "--export-dir",
        str(h1_export_dir),
    ]
    with task_manifest.step("inspect_h1_asset", {"command": inspect_cmd, "task": task}) as step:
        _run(inspect_cmd)
        step["h1_report_path"] = str(h1_report_path)
        step["h1_export_dir"] = str(h1_export_dir)

    resume_from_raw = str(train_spec.get("resume_from_experiment_id", "")).strip()
    resume_from_experiment_id = _safe_name(resume_from_raw) if resume_from_raw else ""
    resume_checkpoint_name = str(train_spec.get("resume_checkpoint", "latest"))
    resume_checkpoint_path: Path | None = None
    if resume_from_experiment_id:
        with task_manifest.step(
            "restore_resume_checkpoint",
            {"resume_from_experiment_id": resume_from_experiment_id, "resume_checkpoint": resume_checkpoint_name},
        ) as step:
            _reload_runs_volume()
            _restore_logs_from_artifact(resume_from_experiment_id)
            resume_checkpoint_path = _find_checkpoint(resume_from_experiment_id, resume_checkpoint_name)
            step["resume_checkpoint_path"] = str(resume_checkpoint_path)

    use_patched_runner = bool(train_spec.get("use_patched_runner", False) and training_patch_path)
    train_num_envs = int(train_spec.get("num_envs", spec.get("num_envs", 4096)))
    train_max_iterations = int(train_spec.get("max_iterations", spec.get("max_iterations", 1000)))
    if use_patched_runner:
        train_script = str(REMOTE_SCRIPT_ROOT / "train_rsl_rl_policy.py")
        train_cmd = [
            "./isaaclab.sh",
            "-p",
            train_script,
            "--task",
            task,
            "--num-envs",
            str(train_num_envs),
            "--max-iterations",
            str(train_max_iterations),
            "--run-name",
            experiment_id,
            "--device",
            device,
            "--config-patch",
            str(training_patch_path),
            "--patch-report",
            str(artifact_root / "isaac_patch_report.json"),
            "--train-report",
            str(artifact_root / "isaac_train_report.json"),
        ]
    else:
        train_script = _rsl_rl_train_script(task, runner)
        train_cmd = [
            "./isaaclab.sh",
            "-p",
            train_script,
            "--task",
            task,
            "--headless",
            "--num_envs",
            str(train_num_envs),
            "--max_iterations",
            str(train_max_iterations),
            "--run_name",
            experiment_id,
        ]
    seed = train_spec.get("seed", spec.get("seed"))
    if seed is not None:
        train_cmd.extend(["--seed", str(seed)])
    if resume_checkpoint_path is not None:
        if use_patched_runner:
            train_cmd.extend(["--resume-checkpoint", str(resume_checkpoint_path)])
        else:
            train_cmd.extend(
                [
                    "--resume",
                    "True",
                    "--load_run",
                    resume_checkpoint_path.parent.name,
                    "--checkpoint",
                    resume_checkpoint_path.name,
                ]
            )
    if bool(train_spec.get("video", False)):
        train_cmd.extend(["--video", "--video_length", "120", "--video_interval", "1", "--enable_cameras"])

    train_timeout = int(train_spec.get("command_timeout_seconds", 0))
    with task_manifest.step(
        "train_policy",
        {
            "command": train_cmd,
            "timeout_seconds": train_timeout,
            "num_envs": train_num_envs,
            "max_iterations": train_max_iterations,
            "seed": seed,
            "use_patched_runner": use_patched_runner,
            "training_patch_path": str(training_patch_path) if training_patch_path else None,
            "resume_from_experiment_id": resume_from_experiment_id or None,
            "resume_checkpoint": resume_checkpoint_name if resume_from_experiment_id else None,
            "resume_checkpoint_path": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        },
    ) as step:
        rc = _run(_with_timeout(train_cmd, train_timeout), ok_codes=(0, 124) if train_timeout > 0 else (0,))
        step["return_code"] = rc

    with task_manifest.step("select_checkpoint", {"checkpoint": str(render_spec.get("checkpoint", "latest"))}) as step:
        checkpoint_path = _checkpoint_from_train_report(artifact_root) if use_patched_runner else None
        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint(experiment_id, str(render_spec.get("checkpoint", "latest")))
        print(f"Selected checkpoint: {checkpoint_path}", flush=True)
        step["checkpoint_path"] = str(checkpoint_path)
    with task_manifest.step("sync_training_artifacts", {"experiment_id": experiment_id}) as step:
        step["artifact_root"] = _sync_artifacts(experiment_id)

    raw_metrics_path = artifact_root / "raw_eval_metrics.json"
    rollout_trace_path = artifact_root / "rollout_trace.json"
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
        "--trace-output",
        str(rollout_trace_path),
        "--trace-max-steps",
        str(int(render_spec.get("trace_max_steps", render_spec.get("video_length", 240)))),
    ]
    if training_patch_path is not None:
        eval_cmd.extend(["--config-patch", str(training_patch_path)])
    eval_timeout = int(eval_spec.get("command_timeout_seconds", 300))
    with task_manifest.step(
        "evaluate_policy",
        {
            "command": eval_cmd,
            "timeout_seconds": eval_timeout,
            "raw_metrics_path": str(raw_metrics_path),
            "rollout_trace_path": str(rollout_trace_path),
        },
    ) as step:
        try:
            rc = _run(_with_timeout(eval_cmd, eval_timeout), ok_codes=(0, 124) if eval_timeout > 0 else (0,))
            step["return_code"] = rc
        except subprocess.CalledProcessError as exc:
            fallback = _fallback_eval_metrics(experiment_id, f"eval command failed with exit code {exc.returncode}")
            raw_metrics_path.write_text(json.dumps(fallback, indent=2, sort_keys=True) + "\n")
            step["status"] = "fallback"
            step["return_code"] = exc.returncode
            step["fallback_metrics"] = fallback

    with task_manifest.step("score_eval_metrics", {"raw_metrics_path": str(raw_metrics_path)}) as step:
        if raw_metrics_path.exists():
            raw_metrics = json.loads(raw_metrics_path.read_text())
        else:
            raw_metrics = _fallback_eval_metrics(experiment_id, "eval command completed without writing raw metrics")
            raw_metrics_path.write_text(json.dumps(raw_metrics, indent=2, sort_keys=True) + "\n")
        score = _score_metrics(raw_metrics)
        score_path = artifact_root / "eval_metrics.json"
        score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
        step["score_path"] = str(score_path)
        step["total_score"] = score.get("total_score")
        step["safety_passed"] = score.get("safety_passed")

    render_cmd = _build_telemetry_render_cmd(
        trace_path=rollout_trace_path,
        score_path=score_path,
        output_path=artifact_root / "rollout_telemetry.mp4",
        render_spec=render_spec,
    )
    render_error_path = artifact_root / "render_error.json"
    render_timeout = int(render_spec.get("command_timeout_seconds", 900))
    with task_manifest.step(
        "render_telemetry_video",
        {"command": render_cmd, "timeout_seconds": render_timeout, "output_path": str(artifact_root / "rollout_telemetry.mp4")},
    ) as step:
        try:
            rc = _run(_with_timeout(render_cmd, render_timeout), ok_codes=(0, 124))
            step["return_code"] = rc
        except subprocess.CalledProcessError as exc:
            render_error = {
                "command": exc.cmd,
                "returncode": exc.returncode,
                "note": "Telemetry render failed; preserving train/eval artifacts and manifest without rollout video.",
            }
            render_error_path.write_text(json.dumps(render_error, indent=2, sort_keys=True) + "\n")
            step["status"] = "error"
            step["render_error_path"] = str(render_error_path)
            step["render_error"] = render_error

    with task_manifest.step("sync_final_artifacts", {"experiment_id": experiment_id}) as step:
        synced_root = Path(_sync_artifacts(experiment_id))
        rollout_video_paths = _find_videos(synced_root)
        rollout_video_path = rollout_video_paths[-1] if rollout_video_paths else None
        score["rollout_video_path"] = rollout_video_path
        score["rollout_video_paths"] = rollout_video_paths
        score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
        (synced_root / "rollout_videos.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "primary_video_path": rollout_video_path,
                    "video_count": len(rollout_video_paths),
                    "video_paths": rollout_video_paths,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        step["synced_root"] = str(synced_root)
        step["video_count"] = len(rollout_video_paths)
        step["primary_video_path"] = rollout_video_path
    manifest_path = synced_root / "artifact_manifest.json"
    manifest_cmd = [
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
        "--rollout-trace",
        str(rollout_trace_path if rollout_trace_path.exists() else ""),
        "--output",
        str(manifest_path),
    ]
    with task_manifest.step("write_artifact_manifest", {"command": manifest_cmd, "manifest_path": str(manifest_path)}) as step:
        _run(
            manifest_cmd,
            cwd=Path("/"),
        )
        step["manifest_path"] = str(manifest_path)
    task_manifest.write(
        status="done",
        video_paths=rollout_video_paths,
        primary_video_path=rollout_video_path,
        summary={"score_path": str(score_path), "total_score": score.get("total_score"), "manifest_path": str(manifest_path)},
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


def _build_telemetry_render_cmd(
    *,
    trace_path: Path,
    score_path: Path,
    output_path: Path,
    render_spec: dict[str, Any],
) -> list[str]:
    return [
        "python",
        str(REMOTE_SCRIPT_ROOT / "render_telemetry_video.py"),
        "--trace",
        str(trace_path),
        "--metrics",
        str(score_path),
        "--output",
        str(output_path),
        "--title",
        str(render_spec.get("title", "RoboGenesis H1 telemetry rollout")),
        "--fps",
        str(int(render_spec.get("fps", 20))),
    ]


def _build_isaac_camera_render_cmd(
    *,
    runner: str,
    task: str,
    checkpoint_path: Path,
    render_spec: dict[str, Any],
) -> list[str]:
    # Isaac Lab 2.0.x rsl_rl/play.py does not accept a seed argument.
    return [
        "./isaaclab.sh",
        "-p",
        _rsl_rl_play_script(task, runner),
        "--task",
        task,
        "--headless",
        "--num_envs",
        str(int(render_spec.get("num_envs", 1))),
        "--load_run",
        checkpoint_path.parent.name,
        "--checkpoint",
        str(checkpoint_path),
        "--video",
        "--video_length",
        str(int(render_spec.get("video_length", 240))),
        "--enable_cameras",
    ]


@app.function(
    image=image,
    gpu=RENDER_GPU,
    cpu=RENDER_CPU_COUNT,
    memory=RENDER_MEMORY_MB,
    timeout=2 * 60 * 60,
    volumes=volumes,
    min_containers=0,
    max_containers=MAX_RENDER_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
def render_isaac_h1_video_job(render_spec_json: str) -> dict[str, Any]:
    """Render an actual Isaac camera video for an existing H1 experiment."""

    _reload_runs_volume()
    spec = json.loads(render_spec_json)
    experiment_id = _safe_name(str(spec["experiment_id"]))
    task = str(spec.get("task", "Isaac-Velocity-Flat-H1-v0"))
    runner = str(spec.get("runner", "rsl_rl"))
    checkpoint = str(spec.get("checkpoint", "latest"))
    artifact_root = _artifact_root(experiment_id)
    task_manifest = _RaindropTaskManifest(
        artifact_root,
        experiment_id=experiment_id,
        event_name="robogenesis-sim-run",
        source="modal.render_isaac_h1_video_job",
        metadata={"task": task, "runner": runner, "gpu": RENDER_GPU},
    )

    with task_manifest.step("render_worker_smoke", {"gpu": RENDER_GPU}) as step:
        _run(["nvidia-smi"], cwd=Path("/"))
        step["gpu"] = RENDER_GPU
    with task_manifest.step("restore_training_logs", {"experiment_id": experiment_id}) as step:
        _restore_logs_from_artifact(experiment_id)
        step["logs_restored"] = True
    with task_manifest.step("select_render_checkpoint", {"checkpoint": checkpoint}) as step:
        checkpoint_path = _find_checkpoint(experiment_id, checkpoint)
        step["checkpoint_path"] = str(checkpoint_path)
    render_cmd = _build_isaac_camera_render_cmd(
        runner=runner,
        task=task,
        checkpoint_path=checkpoint_path,
        render_spec=spec,
    )
    timeout_seconds = int(spec.get("command_timeout_seconds", 1800))
    render_error_path = artifact_root / "isaac_camera_render_error.json"
    with task_manifest.step(
        "render_isaac_camera_video",
        {"command": render_cmd, "timeout_seconds": timeout_seconds, "gpu": RENDER_GPU},
    ) as step:
        try:
            rc = _run(_with_timeout(render_cmd, timeout_seconds), ok_codes=(0, 124) if timeout_seconds > 0 else (0,))
            step["return_code"] = rc
        except subprocess.CalledProcessError as exc:
            render_error = {
                "command": exc.cmd,
                "returncode": exc.returncode,
                "gpu": RENDER_GPU,
                "note": "Actual Isaac camera render failed on RTX render worker.",
            }
            render_error_path.write_text(json.dumps(render_error, indent=2, sort_keys=True) + "\n")
            step["status"] = "error"
            step["render_error_path"] = str(render_error_path)
            step["render_error"] = render_error
            task_manifest.write(status="error")
            runs_volume.commit()
            raise

    with task_manifest.step("sync_isaac_camera_video", {"experiment_id": experiment_id}) as step:
        synced_root = Path(_sync_artifacts(experiment_id))
        actual_videos = _find_isaac_camera_videos(synced_root)
        all_videos = _find_videos(synced_root)
        step["synced_root"] = str(synced_root)
        step["actual_video_count"] = len(actual_videos)
        step["primary_actual_video_path"] = actual_videos[-1] if actual_videos else None
    report = {
        "experiment_id": experiment_id,
        "task": task,
        "runner": runner,
        "gpu": RENDER_GPU,
        "checkpoint_path": str(checkpoint_path),
        "actual_video_count": len(actual_videos),
        "actual_video_paths": actual_videos,
        "primary_actual_video_path": actual_videos[-1] if actual_videos else None,
        "all_video_paths": all_videos,
        "render_command": render_cmd,
    }
    (synced_root / "isaac_camera_videos.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    task_manifest.write(
        status="done",
        video_paths=all_videos,
        primary_video_path=actual_videos[-1] if actual_videos else None,
        summary=report,
    )
    runs_volume.commit()
    return report


def _fallback_eval_metrics(experiment_id: str, reason: str) -> dict[str, Any]:
    return {
        "policy_id": experiment_id,
        "episode_count": 0,
        "eval_seed_count": 0,
        "completed_eval_seed_count": 0,
        "evaluation_errors": [reason],
        "mean_episode_reward": 0.0,
        "mean_episode_length": 0.0,
        "command_tracking": 0.0,
        "survival_no_fall": 0.0,
        "base_success": 0.0,
        "stability": 0.0,
        "generated_scenario_success": 0.0,
        "gait_quality": 0.0,
        "energy_efficiency": 0.0,
        "smoothness": 0.0,
        "recovery_from_disturbance": 0.0,
        "fall_rate": 1.0,
        "nan_actions": False,
        "reward_hacking_detected": False,
        "raw_metric_note": "Fallback metrics written because phase-1 evaluation did not produce a raw metrics file.",
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
    if action == "phase1-detach":
        call = phase1_baseline_job.spawn(experiment_spec_json)
        print(json.dumps({"function_call_id": call.object_id, "status": "spawned"}, indent=2, sort_keys=True))
        return
    if action == "phase1-batch-detach":
        specs = json.loads(experiment_spec_json)
        if not isinstance(specs, list):
            raise ValueError("phase1-batch-detach expects experiment_spec_json to be a JSON list")
        calls = []
        for spec in specs:
            calls.append(phase1_baseline_job.spawn(json.dumps(spec, sort_keys=True)))
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "max_containers": MAX_PHASE1_CONTAINERS,
                    "gpu": DEFAULT_GPU,
                    "function_call_ids": [call.object_id for call in calls],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "render-isaac":
        print(json.dumps(render_isaac_h1_video_job.remote(experiment_spec_json), indent=2, sort_keys=True))
        return
    if action == "render-isaac-detach":
        call = render_isaac_h1_video_job.spawn(experiment_spec_json)
        print(
            json.dumps(
                {
                    "function_call_id": call.object_id,
                    "gpu": RENDER_GPU,
                    "status": "spawned",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise ValueError(
        "action must be 'smoke', 'train-and-eval', 'phase1', 'phase1-detach', "
        "'phase1-batch-detach', 'render-isaac', or 'render-isaac-detach'"
    )
