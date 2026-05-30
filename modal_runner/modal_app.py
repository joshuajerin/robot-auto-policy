"""Modal app for Isaac Lab training and evaluation."""

from __future__ import annotations

import json
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
    raise ValueError("action must be 'smoke' or 'train-and-eval'")

