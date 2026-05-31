"""Publish local render artifact folders into Raindrop Workshop."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.raindrop_trace import publish_artifact_run


MARKER_FILES = (
    "raindrop_trace.json",
    "artifact_manifest.json",
    "isaac_camera_videos.json",
    "rollout_videos.json",
)


@dataclass(frozen=True)
class RenderRun:
    experiment_id: str
    artifact_dir: Path
    video_root: Path
    video_paths: tuple[Path, ...]


def discover_render_runs(artifact_root: str | Path = "artifacts") -> list[RenderRun]:
    root = Path(artifact_root)
    if not root.exists():
        return []

    runs: dict[tuple[str, str], RenderRun] = {}
    for marker_dir in _marker_dirs(root):
        experiment_id = _experiment_id(marker_dir)
        video_root = _outermost_named_dir(root, marker_dir, experiment_id) or marker_dir
        videos = _mp4s(video_root)
        if not videos:
            videos = _mp4s(marker_dir)
            video_root = marker_dir
        if videos:
            _put_run(runs, RenderRun(experiment_id, marker_dir, video_root, tuple(videos)))

    covered_roots = [run.video_root.resolve() for run in runs.values()]
    for video in _mp4s(root):
        if _is_covered(video, covered_roots):
            continue
        video_root = _loose_video_root(root, video)
        videos = _mp4s(video_root)
        if videos:
            _put_run(runs, RenderRun(_experiment_id(video_root), video_root, video_root, tuple(videos)))

    return sorted(runs.values(), key=lambda run: (run.experiment_id, str(run.video_root)))


def publish_render_runs(
    *,
    artifact_root: str | Path = "artifacts",
    db_path: str | Path = "artifacts/research.db",
    repo_root: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    runs = discover_render_runs(artifact_root)
    if limit is not None:
        runs = runs[:limit]

    summaries: list[dict[str, Any]] = []
    run_counts: dict[str, int] = {}
    for run in runs:
        run_counts[run.experiment_id] = run_counts.get(run.experiment_id, 0) + 1
    for run in runs:
        ingest_summary = _ingest_summary(run)
        event_id = _event_id(run, artifact_root=Path(artifact_root), duplicate=run_counts[run.experiment_id] > 1)
        summary: dict[str, Any] = {
            "experiment_id": run.experiment_id,
            "event_id": event_id,
            "artifact_dir": str(run.artifact_dir),
            "video_root": str(run.video_root),
            "video_count": len(run.video_paths),
            "published": False,
        }
        if not dry_run:
            summary["raindrop"] = publish_artifact_run(
                run.artifact_dir,
                ingest_summary=ingest_summary,
                accepted=None,
                review_reasons=[],
                db_path=db_path,
                event_id=event_id,
                repo_root=repo,
            )
            summary["published"] = True
        summaries.append(summary)
    return summaries


def _marker_dirs(root: Path) -> list[Path]:
    dirs: dict[Path, int] = {}
    for priority, marker in enumerate(MARKER_FILES):
        for path in root.rglob(marker):
            existing = dirs.get(path.parent)
            if existing is None or priority < existing:
                dirs[path.parent] = priority
    return [path for path, _ in sorted(dirs.items(), key=lambda item: (item[1], str(item[0])))]


def _put_run(runs: dict[tuple[str, str], RenderRun], run: RenderRun) -> None:
    key = (run.experiment_id, str(run.video_root.resolve()))
    existing = runs.get(key)
    if existing is None or _marker_priority(run.artifact_dir) < _marker_priority(existing.artifact_dir):
        runs[key] = run


def _marker_priority(path: Path) -> int:
    for index, marker in enumerate(MARKER_FILES):
        if (path / marker).exists():
            return index
    return len(MARKER_FILES)


def _experiment_id(directory: Path) -> str:
    for marker in MARKER_FILES:
        path = directory / marker
        data = _read_json(path)
        if data:
            value = data.get("experiment_id") or data.get("event_id")
            if value:
                return str(value)
    return directory.name


def _ingest_summary(run: RenderRun) -> dict[str, Any]:
    metrics = _find_metrics(run.artifact_dir) or _find_metrics(run.video_root) or {}
    score = metrics.get("total_score")
    if score is None:
        score = metrics.get("score")
    return {
        "experiment_id": run.experiment_id,
        "score": score if score is not None else "unknown",
        "primary_failure": metrics.get("primary_failure") or "unknown",
        "rollout_video_path": str(run.video_paths[0]) if run.video_paths else None,
        "rollout_video_paths": [str(path) for path in run.video_paths],
    }


def _event_id(run: RenderRun, *, artifact_root: Path, duplicate: bool) -> str:
    if not duplicate:
        return run.experiment_id
    try:
        rel = run.video_root.resolve().relative_to(artifact_root.resolve())
        source = str(rel)
    except (OSError, ValueError):
        source = str(run.video_root)
    suffix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{run.experiment_id}--{suffix}"


def _find_metrics(directory: Path) -> dict[str, Any] | None:
    candidates = [
        directory / "eval_metrics.json",
        directory / "raw_eval_metrics.json",
        *directory.rglob("eval_metrics.json"),
        *directory.rglob("raw_eval_metrics.json"),
    ]
    for path in candidates:
        data = _read_json(path)
        if data:
            return data
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _mp4s(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path.resolve() for path in root.rglob("*.mp4") if path.is_file())


def _outermost_named_dir(root: Path, directory: Path, experiment_id: str) -> Path | None:
    try:
        directory.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    candidates = [directory, *directory.parents]
    named = [candidate for candidate in candidates if candidate.name == experiment_id and _is_under(candidate, root)]
    return min(named, key=lambda candidate: len(candidate.relative_to(root).parts), default=None)


def _loose_video_root(root: Path, video: Path) -> Path:
    try:
        rel = video.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return video.parent
    if len(rel.parts) >= 3:
        return root / rel.parts[0] / rel.parts[1]
    return video.parent


def _is_covered(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--db", default="artifacts/research.db")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summaries = publish_render_runs(
        artifact_root=args.artifact_root,
        db_path=args.db,
        repo_root=args.repo_root,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps({"published_count": sum(1 for item in summaries if item["published"]), "runs": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
