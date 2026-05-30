"""Summarize RoboGenesis experiment artifacts stored in a Modal Volume."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class ExperimentArtifactStatus:
    experiment_id: str
    files: list[str]

    @property
    def has_raw_metrics(self) -> bool:
        return "raw_eval_metrics.json" in self.files

    @property
    def has_eval_metrics(self) -> bool:
        return "eval_metrics.json" in self.files

    @property
    def has_manifest(self) -> bool:
        return "artifact_manifest.json" in self.files

    @property
    def has_render_error(self) -> bool:
        return "render_error.json" in self.files

    @property
    def has_rollout_trace(self) -> bool:
        return "rollout_trace.json" in self.files

    @property
    def video_count(self) -> int:
        return sum(1 for path in self.files if path.endswith(".mp4"))

    @property
    def video_paths(self) -> list[str]:
        return [path for path in self.files if path.endswith(".mp4")]

    @property
    def primary_video_path(self) -> str | None:
        return self.video_paths[-1] if self.video_paths else None

    @property
    def checkpoint_count(self) -> int:
        return sum(1 for path in self.files if path.endswith(".pt"))

    @property
    def ready_for_review(self) -> bool:
        return (
            self.has_raw_metrics
            and self.has_eval_metrics
            and self.has_manifest
            and self.has_rollout_trace
            and not self.has_render_error
            and self.checkpoint_count > 0
            and self.video_count > 0
        )

    @property
    def review_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.has_raw_metrics:
            blockers.append("missing raw_eval_metrics.json")
        if not self.has_eval_metrics:
            blockers.append("missing eval_metrics.json")
        if not self.has_manifest:
            blockers.append("missing artifact_manifest.json")
        if not self.has_rollout_trace:
            blockers.append("missing rollout_trace.json")
        if self.has_render_error:
            blockers.append("render_error.json present")
        if self.checkpoint_count == 0:
            blockers.append("missing policy checkpoint")
        if self.video_count == 0:
            blockers.append("missing rollout video")
        return blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "has_raw_metrics": self.has_raw_metrics,
            "has_eval_metrics": self.has_eval_metrics,
            "has_manifest": self.has_manifest,
            "has_render_error": self.has_render_error,
            "has_rollout_trace": self.has_rollout_trace,
            "video_count": self.video_count,
            "video_paths": self.video_paths,
            "primary_video_path": self.primary_video_path,
            "checkpoint_count": self.checkpoint_count,
            "ready_for_review": self.ready_for_review,
            "review_blockers": self.review_blockers,
            "file_count": len(self.files),
        }


def list_experiment_files(volume: str, experiment_id: str) -> list[str]:
    root = f"/experiments/{experiment_id}"
    return sorted(_list_files_recursive(volume, root, f"experiments/{experiment_id}/"))


def _list_files_recursive(volume: str, path: str, prefix: str) -> list[str]:
    proc = subprocess.run(
        ["modal", "volume", "ls", "--json", volume, path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    files: list[str] = []
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse modal volume ls JSON for {path}: {exc}") from exc
    for entry in entries:
        value = str(entry.get("Filename", "")).strip()
        entry_type = str(entry.get("Type", "")).strip()
        if not value:
            continue
        if entry_type == "dir":
            files.extend(_list_files_recursive(volume, "/" + value, prefix))
            continue
        if entry_type == "file" and value.startswith(prefix):
            files.append(str(PurePosixPath(value[len(prefix) :])))
    return files


def summarize_experiments(volume: str, experiment_ids: list[str]) -> list[ExperimentArtifactStatus]:
    return [
        ExperimentArtifactStatus(experiment_id=experiment_id, files=list_experiment_files(volume, experiment_id))
        for experiment_id in experiment_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", default="robogenesis-runs")
    parser.add_argument("--experiment", action="append", required=True)
    args = parser.parse_args()

    try:
        statuses = summarize_experiments(args.volume, args.experiment)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps([status.to_dict() for status in statuses], indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
