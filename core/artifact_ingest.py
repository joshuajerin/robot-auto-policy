"""Ingest Modal/Isaac experiment artifacts into RoboGenesis research memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adapters.locomotion import LocomotionAdapter
from core.experiment_db import ExperimentDB
from core.schemas import PatchSpec, ScoreBreakdown
from core.scoring import score_from_metrics


def ingest_artifact_dir(
    artifact_dir: str | Path,
    *,
    db_path: str | Path = "artifacts/research.db",
    parent_policy_id: str | None = None,
    accepted: bool = True,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    if not artifact_root.exists():
        raise FileNotFoundError(artifact_root)

    manifest = _load_json_if_exists(artifact_root / "artifact_manifest.json") or {}
    experiment_id = str(manifest.get("experiment_id") or artifact_root.name)
    raw_metrics = _load_metrics(artifact_root, manifest)
    score = _score(raw_metrics, experiment_id)
    checkpoint_path = _find_checkpoint_path(artifact_root, manifest)
    rollout_video_path = _find_rollout_video(artifact_root, manifest)

    db = ExperimentDB(db_path)
    patch = _baseline_patch(experiment_id, artifact_root)
    db.insert_experiment(
        experiment_id=experiment_id,
        parent_policy_id=parent_policy_id,
        patch=patch,
        status="completed",
        score_before=None,
        score_after=score.total_score,
        accepted=accepted,
        modal_job_id=str(manifest.get("modal_job_id") or ""),
    )
    db.insert_policy(
        policy_id=score.policy_id,
        parent_policy_id=parent_policy_id,
        checkpoint_path=checkpoint_path,
        metrics=score,
        accepted=accepted,
    )
    failure_report = LocomotionAdapter().diagnose([], raw_metrics)
    db.insert_failure_report(experiment_id, score.policy_id, failure_report)
    db.close()

    summary = {
        "experiment_id": experiment_id,
        "policy_id": score.policy_id,
        "score": score.total_score,
        "checkpoint_path": checkpoint_path,
        "rollout_video_path": rollout_video_path,
        "primary_failure": failure_report.primary_failure,
        "db_path": str(db_path),
    }
    return summary


def _load_metrics(artifact_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        _localize_path(artifact_root, manifest.get("score_path")),
        artifact_root / "eval_metrics.json",
        _localize_path(artifact_root, manifest.get("raw_metrics_path")),
        artifact_root / "raw_eval_metrics.json",
    ]
    for path in candidates:
        if path and path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"No eval metrics found under {artifact_root}")


def _score(raw_metrics: dict[str, Any], experiment_id: str) -> ScoreBreakdown:
    policy_id = str(raw_metrics.get("policy_id") or experiment_id)
    return score_from_metrics(policy_id, raw_metrics)


def _find_checkpoint_path(artifact_root: Path, manifest: dict[str, Any]) -> str | None:
    manifest_checkpoint = _localize_path(artifact_root, manifest.get("checkpoint_path"))
    if manifest_checkpoint and manifest_checkpoint.exists():
        return str(manifest_checkpoint)
    checkpoints = sorted(artifact_root.rglob("*.pt"))
    return str(checkpoints[-1]) if checkpoints else str(manifest.get("checkpoint_path") or "")


def _find_rollout_video(artifact_root: Path, manifest: dict[str, Any]) -> str | None:
    manifest_video = _localize_path(artifact_root, manifest.get("rollout_video_path"))
    if manifest_video and manifest_video.exists():
        return str(manifest_video)
    videos = sorted(artifact_root.rglob("*.mp4"))
    return str(videos[-1]) if videos else None


def _localize_path(artifact_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    if path.is_absolute():
        candidate = artifact_root / path.name
        return candidate
    return artifact_root / path


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _baseline_patch(experiment_id: str, artifact_root: Path) -> PatchSpec:
    spec = _load_json_if_exists(artifact_root / "experiment_spec.json") or {}
    train = dict(spec.get("train", {}))
    return PatchSpec(
        experiment_name=f"{experiment_id}_baseline_ingest",
        hypothesis="Ingested completed H1 baseline run from Modal artifacts.",
        allowed_files=["configs/locomotion/ppo.yaml"],
        patch={"ppo.max_iterations": int(train.get("max_iterations", 1000))},
        expected_effect="Establish a measured baseline policy for subsequent AutoResearch iterations.",
        risk="Baseline may be undertrained or overfit to the base flat task.",
        rollback="Keep prior accepted policy if this baseline is not promoted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--db", default="artifacts/research.db")
    parser.add_argument("--parent-policy-id", default="")
    parser.add_argument("--rejected", action="store_true")
    args = parser.parse_args()

    summary = ingest_artifact_dir(
        args.artifact_dir,
        db_path=args.db,
        parent_policy_id=args.parent_policy_id or None,
        accepted=not args.rejected,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
