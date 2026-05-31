"""Modal artifact download and research-memory ingestion helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.artifact_ingest import ingest_artifact_dir
from core.experiment_db import ExperimentDB
from core.raindrop_trace import publish_artifact_run
from core.scoring import should_accept, score_from_metrics
from core.schemas import ScoreBreakdown


@dataclass(frozen=True)
class ArtifactSyncResult:
    experiment_id: str
    artifact_dir: str
    downloaded: bool
    ingested: bool
    accepted: bool
    review_reasons: list[str]
    ingest_summary: dict[str, Any] | None
    raindrop_summary: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def modal_experiment_remote_path(experiment_id: str) -> str:
    return f"/experiments/{_safe_experiment_id(experiment_id)}"


def modal_experiment_local_path(destination_root: str | Path, experiment_id: str) -> Path:
    return Path(destination_root) / _safe_experiment_id(experiment_id)


def build_modal_get_command(
    volume: str,
    experiment_id: str,
    destination_root: str | Path,
    *,
    force: bool = True,
    environment_name: str | None = None,
) -> list[str]:
    command = ["modal", "volume", "get"]
    if force:
        command.append("--force")
    if environment_name:
        command.extend(["--env", environment_name])
    command.extend(
        [
            volume,
            modal_experiment_remote_path(experiment_id),
            str(destination_root),
        ]
    )
    return command


def download_modal_experiment(
    experiment_id: str,
    *,
    volume: str = "robogenesis-runs",
    destination_root: str | Path = "artifacts/modal_downloads",
    environment_name: str | None = None,
) -> Path:
    destination = modal_experiment_local_path(destination_root, experiment_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    subprocess.run(
        build_modal_get_command(
            volume,
            experiment_id,
            destination_root,
            environment_name=environment_name,
        ),
        check=True,
    )
    return destination


def sync_and_ingest_modal_experiment(
    experiment_id: str,
    *,
    db_path: str | Path = "artifacts/research.db",
    volume: str = "robogenesis-runs",
    destination_root: str | Path = "artifacts/modal_downloads",
    parent_policy_id: str | None = None,
    accepted: bool | None = None,
    environment_name: str | None = None,
) -> ArtifactSyncResult:
    artifact_dir = download_modal_experiment(
        experiment_id,
        volume=volume,
        destination_root=destination_root,
        environment_name=environment_name,
    )
    existing = _existing_experiment_fields(db_path, experiment_id)
    parent_score = _load_parent_score(db_path, parent_policy_id)
    accepted, review_reasons = _review_artifact_acceptance(
        artifact_dir,
        db_path=db_path,
        parent_policy_id=parent_policy_id,
        parent_score=parent_score,
        accepted_override=accepted,
    )
    summary = ingest_artifact_dir(
        artifact_dir,
        db_path=db_path,
        parent_policy_id=parent_policy_id,
        accepted=accepted,
    )
    score_before = existing.get("score_before")
    if score_before is None and parent_score is not None:
        score_before = parent_score.total_score
    _mark_review_status(
        db_path,
        experiment_id,
        accepted=accepted,
        modal_job_id=existing.get("modal_job_id"),
        score_before=score_before,
    )
    raindrop_summary = publish_artifact_run(
        artifact_dir,
        ingest_summary=summary,
        accepted=accepted,
        review_reasons=review_reasons,
        db_path=db_path,
    )
    return ArtifactSyncResult(
        experiment_id=experiment_id,
        artifact_dir=str(artifact_dir),
        downloaded=True,
        ingested=True,
        accepted=accepted,
        review_reasons=review_reasons,
        ingest_summary=summary,
        raindrop_summary=raindrop_summary,
    )


def _safe_experiment_id(experiment_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in experiment_id)
    return safe.strip(".-_") or "experiment"


def _review_artifact_acceptance(
    artifact_dir: Path,
    *,
    db_path: str | Path,
    parent_policy_id: str | None,
    parent_score: ScoreBreakdown | None = None,
    accepted_override: bool | None,
) -> tuple[bool, list[str]]:
    if accepted_override is not None:
        return accepted_override, ["manual_accept_override" if accepted_override else "manual_reject_override"]

    parent = parent_score or _load_parent_score(db_path, parent_policy_id)
    if parent is None:
        return False, ["missing_parent_policy_for_locked_review"]

    raw_metrics = _load_metrics_for_review(artifact_dir)
    policy_id = str(raw_metrics.get("policy_id") or artifact_dir.name)
    candidate = score_from_metrics(policy_id, raw_metrics)
    accepted = should_accept(parent, candidate)
    if accepted:
        return True, ["passes_locked_acceptance_rule"]

    reasons: list[str] = []
    if candidate.total_score < parent.total_score + 0.03:
        reasons.append(
            f"score_delta_too_small:{candidate.total_score:.4f}<{parent.total_score + 0.03:.4f}"
        )
    if not candidate.safety_passed:
        reasons.append("safety_failed")
    if candidate.base_success < parent.base_success - 0.05:
        reasons.append(f"base_success_regressed:{candidate.base_success:.4f}<{parent.base_success - 0.05:.4f}")
    if candidate.generated_scenario_success < parent.generated_scenario_success:
        reasons.append(
            "generated_scenario_success_regressed:"
            f"{candidate.generated_scenario_success:.4f}<{parent.generated_scenario_success:.4f}"
        )
    if candidate.eval_seed_count < 8:
        reasons.append(f"insufficient_eval_seeds:{candidate.eval_seed_count}<8")
    if candidate.reward_hacking_detected:
        reasons.append("reward_hacking_detected")
    return False, reasons or ["locked_acceptance_rule_failed"]


def _load_parent_score(db_path: str | Path, parent_policy_id: str | None) -> ScoreBreakdown | None:
    db = ExperimentDB(db_path)
    try:
        if parent_policy_id:
            row = db.conn.execute(
                "SELECT metrics_json FROM policies WHERE policy_id = ?",
                (parent_policy_id,),
            ).fetchone()
        else:
            row = db.conn.execute(
                """
                SELECT metrics_json
                FROM policies
                WHERE accepted = 1
                ORDER BY score DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return ScoreBreakdown(**json.loads(row["metrics_json"]))
    finally:
        db.close()


def _load_metrics_for_review(artifact_dir: Path) -> dict[str, Any]:
    for path in (artifact_dir / "raw_eval_metrics.json", artifact_dir / "eval_metrics.json"):
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"No metrics found for locked review under {artifact_dir}")


def _existing_experiment_fields(db_path: str | Path, experiment_id: str) -> dict[str, Any]:
    db = ExperimentDB(db_path)
    try:
        row = db.conn.execute(
            "SELECT modal_job_id, score_before FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return {"modal_job_id": None, "score_before": None}
        return {
            "modal_job_id": str(row["modal_job_id"]) if row["modal_job_id"] else None,
            "score_before": row["score_before"],
        }
    finally:
        db.close()


def _mark_review_status(
    db_path: str | Path,
    experiment_id: str,
    *,
    accepted: bool,
    modal_job_id: str | None,
    score_before: float | None,
) -> None:
    db = ExperimentDB(db_path)
    try:
        db.update_experiment_status(
            experiment_id,
            status="accepted" if accepted else "rejected",
            score_before=score_before,
            accepted=accepted,
            modal_job_id=modal_job_id,
        )
    finally:
        db.close()
