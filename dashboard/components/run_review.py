"""Run-review data loader for the Streamlit dashboard."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


SECTION_LABELS = {
    "actuators": "Actuators",
    "curriculum": "Curriculum",
    "domain_randomization": "Domain Randomization",
    "ppo": "PPO",
    "render": "Render",
    "reward_weights": "Rewards",
    "terrain": "Terrain",
    "train": "Training",
}

VIDEO_MANIFEST_NAMES = (
    "rollout_videos.json",
    "isaac_camera_videos.json",
    "raindrop_trace.json",
)
VIDEO_LIST_PATH_KEYS = {
    "actual_video_paths",
    "all_video_paths",
    "rollout_video_files",
    "rollout_video_paths",
    "video_paths",
}
VIDEO_SCALAR_PATH_KEYS = {
    "actual_video_path",
    "primary_actual_video_path",
    "primary_video_path",
    "rollout_video_path",
    "video_path",
}


def load_run_reviews(
    conn: sqlite3.Connection,
    *,
    repo_root: str | Path,
    artifact_root: str | Path = "artifacts",
) -> list[dict[str, Any]]:
    """Load per-experiment review records from SQLite and local artifacts."""

    repo = Path(repo_root)
    artifacts = Path(artifact_root)
    if not artifacts.is_absolute():
        artifacts = repo / artifacts

    failures_by_experiment = _load_failures(conn)
    scenario_evals = _load_scenario_evals(conn)
    rows = conn.execute(
        """
        SELECT id, parent_policy_id, patch_json, hypothesis, status, created_at,
               modal_job_id, score_before, score_after, accepted
        FROM experiments
        ORDER BY rowid DESC
        """
    ).fetchall()

    reviews: list[dict[str, Any]] = []
    for row in rows:
        experiment = _experiment_row(row)
        experiment_id = str(experiment["experiment_id"])
        artifact_bundle = find_run_artifacts(repo, artifacts, experiment_id)
        patch = _json_object(experiment.get("patch_json"))
        spec = artifact_bundle.get("experiment_spec") or {}
        run_meta = _run_meta(spec, artifact_bundle.get("context") or {})
        failure_reports = failures_by_experiment.get(experiment_id, [])
        primary_failure = failure_reports[0]["report"] if failure_reports else {}
        config_changes = run_meta.get("config_changes") or {}
        change_rows = build_change_rows(config_changes, patch)
        videos = _collect_videos(
            repo=repo,
            artifact_dirs=artifact_bundle.get("artifact_dirs", []),
            manifest=artifact_bundle.get("manifest") or {},
            video_manifests=artifact_bundle.get("video_manifests") or [],
            spec=spec,
            run_meta=run_meta,
            failure_reports=failure_reports,
            scenario_evals=scenario_evals,
            experiment_id=experiment_id,
        )
        score_before = _float_or_none(experiment.get("score_before"))
        score_after = _float_or_none(experiment.get("score_after"))

        reviews.append(
            {
                **experiment,
                "patch": patch,
                "rationale": _rationale(patch, primary_failure),
                "failure_report": primary_failure,
                "failure_reports": failure_reports,
                "config_changes": config_changes,
                "change_rows": change_rows,
                "changed_sections": sorted({row["section"] for row in change_rows}),
                "generated_scenarios": run_meta.get("generated_scenarios") or [],
                "quick_iteration": run_meta.get("quick_iteration") or {},
                "training_surface": run_meta.get("training_surface") or {},
                "task": spec.get("task") or run_meta.get("task"),
                "train": spec.get("train") or {},
                "render": spec.get("render") or {},
                "videos": videos,
                "source_videos": [video for video in videos if video["kind"] == "source"],
                "run_videos": [video for video in videos if video["kind"] == "run"],
                "artifact_paths": artifact_bundle.get("paths", {}),
                "score_delta": score_after - score_before if score_before is not None and score_after is not None else None,
            }
        )
    return reviews


def find_run_artifacts(repo_root: Path, artifact_root: Path, experiment_id: str) -> dict[str, Any]:
    """Resolve local files that belong to a run without requiring DB schema changes."""

    artifact_dirs = _find_artifact_dirs(artifact_root, experiment_id)
    spec_paths = _dedupe_paths(
        [
            *(artifact_root.rglob(f"{experiment_id}.json") if artifact_root.exists() else []),
            *(directory / "experiment_spec.json" for directory in artifact_dirs),
        ]
    )
    base_id = _base_experiment_id(experiment_id)
    context_paths = _dedupe_paths(artifact_root.rglob(f"{base_id}_context.json") if artifact_root.exists() else [])
    patch_paths = _dedupe_paths(artifact_root.rglob(f"{base_id}_patch.json") if artifact_root.exists() else [])

    manifests = _dedupe_paths(directory / "artifact_manifest.json" for directory in artifact_dirs)
    manifest_path = next((path for path in manifests if path.exists()), None)
    video_manifests = _load_video_manifests(artifact_dirs)

    spec_path, spec = _select_experiment_spec(spec_paths, experiment_id)
    context_path = next((path for path in context_paths if path.exists()), None)
    patch_path = next((path for path in patch_paths if path.exists()), None)

    paths = {
        "experiment_spec": str(spec_path) if spec_path else None,
        "context": str(context_path) if context_path else None,
        "patch": str(patch_path) if patch_path else None,
        "manifest": str(manifest_path) if manifest_path else None,
        "video_manifests": {
            name: [str(entry["path"]) for entry in video_manifests if entry["name"] == name]
            for name in VIDEO_MANIFEST_NAMES
        },
        "artifact_dirs": [str(path) for path in artifact_dirs],
    }

    return {
        "artifact_dirs": artifact_dirs,
        "experiment_spec": spec,
        "context": _read_json(context_path) if context_path else {},
        "patch_file": _read_json(patch_path) if patch_path else {},
        "manifest": _read_json(manifest_path) if manifest_path else {},
        "video_manifests": video_manifests,
        "paths": paths,
    }


def build_change_rows(config_changes: dict[str, Any], patch: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten config changes into sectioned rows suitable for UI tables."""

    rows: list[dict[str, Any]] = []
    seen_parameters: set[str] = set()
    for file_path, changes in sorted((config_changes or {}).items()):
        if not isinstance(changes, dict):
            continue
        for parameter, value in sorted(changes.items()):
            if isinstance(value, dict) and ("old" in value or "new" in value):
                old_value = value.get("old")
                new_value = value.get("new")
            else:
                old_value = None
                new_value = value
            rows.append(_change_row(str(file_path), str(parameter), old_value, new_value))
            seen_parameters.add(str(parameter))

    patch_values = patch.get("patch") if isinstance(patch.get("patch"), dict) else {}
    allowed_files = patch.get("allowed_files") if isinstance(patch.get("allowed_files"), list) else []
    for parameter, new_value in sorted(patch_values.items()):
        if parameter in seen_parameters:
            continue
        rows.append(_change_row(_infer_file(str(parameter), allowed_files), str(parameter), None, new_value))
    return rows


def format_value(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def format_score(value: Any) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number:.3f}"


def format_score_delta(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.3f}"


def _load_failures(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT experiment_id, policy_id, failure_report_json, created_at
        FROM failure_reports
        ORDER BY rowid DESC
        """
    ).fetchall()
    failures: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        experiment_id, policy_id, report_json, created_at = row
        failures.setdefault(str(experiment_id), []).append(
            {
                "policy_id": policy_id,
                "created_at": created_at,
                "report": _json_object(report_json),
            }
        )
    return failures


def _load_scenario_evals(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT scenario_id, policy_id, rollout_video_path
            FROM scenario_evals
            WHERE rollout_video_path IS NOT NULL
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "scenario_id": row[0],
            "policy_id": row[1],
            "rollout_video_path": row[2],
        }
        for row in rows
    ]


def _experiment_row(row: Any) -> dict[str, Any]:
    return {
        "experiment_id": row[0],
        "parent_policy_id": row[1],
        "patch_json": row[2],
        "hypothesis": row[3],
        "status": row[4],
        "created_at": row[5],
        "modal_job_id": row[6],
        "score_before": row[7],
        "score_after": row[8],
        "accepted": bool(row[9]),
    }


def _find_artifact_dirs(artifact_root: Path, experiment_id: str) -> list[Path]:
    if not artifact_root.exists():
        return []
    candidates = [
        artifact_root / experiment_id,
        artifact_root / "modal_downloads" / experiment_id,
        artifact_root / "downloaded_modal" / experiment_id,
        artifact_root / "h100" / experiment_id,
    ]
    candidates.extend(path for path in artifact_root.rglob(experiment_id) if path.is_dir())
    return [path for path in _dedupe_paths(candidates) if path.exists() and path.is_dir()]


def _select_experiment_spec(paths: list[Path], experiment_id: str) -> tuple[Path | None, dict[str, Any]]:
    loaded: list[tuple[Path, dict[str, Any]]] = [(path, _read_json(path)) for path in paths if path.exists()]
    for path, data in loaded:
        if data.get("experiment_id") == experiment_id:
            return path, data
    for path, data in loaded:
        if data.get("autoresearch"):
            return path, data
    return loaded[0] if loaded else (None, {})


def _run_meta(spec: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    spec_meta = spec.get("autoresearch") if isinstance(spec.get("autoresearch"), dict) else {}
    merged: dict[str, Any] = {}
    for source in (context, spec_meta):
        for key, value in source.items():
            if value not in (None, {}, []):
                merged[key] = value
    if "task" not in merged and spec.get("task"):
        merged["task"] = spec["task"]
    return merged


def _collect_videos(
    *,
    repo: Path,
    artifact_dirs: list[Path],
    manifest: dict[str, Any],
    video_manifests: list[dict[str, Any]],
    spec: dict[str, Any],
    run_meta: dict[str, Any],
    failure_reports: list[dict[str, Any]],
    scenario_evals: list[dict[str, Any]],
    experiment_id: str,
) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    for path in _manifest_video_paths(repo, artifact_dirs, manifest, video_manifests, experiment_id):
        videos.append(_video_row("run", path, "Run render"))
    for directory in artifact_dirs:
        for path in sorted(directory.rglob("*.mp4")):
            videos.append(_video_row("run", path, path.name))

    for value in _source_video_values(spec, run_meta, failure_reports):
        path = _resolve_video_path(repo, artifact_dirs, value)
        if path:
            videos.append(_video_row("source", path, _source_label(value)))

    for eval_row in scenario_evals:
        path_value = str(eval_row.get("rollout_video_path") or "")
        policy_id = str(eval_row.get("policy_id") or "")
        if policy_id != experiment_id and experiment_id not in path_value:
            continue
        path = _resolve_video_path(repo, artifact_dirs, path_value, experiment_id=experiment_id)
        if path:
            videos.append(_video_row("run", path, str(eval_row.get("scenario_id") or path.name)))

    return sorted(_dedupe_videos(videos), key=_video_sort_key)


def _manifest_video_paths(
    repo: Path,
    artifact_dirs: list[Path],
    manifest: dict[str, Any],
    video_manifests: list[dict[str, Any]],
    experiment_id: str,
) -> list[Path]:
    values = _video_path_values(manifest)
    for entry in video_manifests:
        values.extend(_video_path_values(entry.get("data") or {}))
    paths: list[Path] = []
    for value in values:
        path = _resolve_video_path(repo, artifact_dirs, value, experiment_id=experiment_id)
        if path:
            paths.append(path)
    return paths


def _source_video_values(spec: dict[str, Any], run_meta: dict[str, Any], failure_reports: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for source in (run_meta.get("multiview_context"), run_meta.get("multiview"), spec.get("multiview")):
        if isinstance(source, dict):
            for view in source.get("views", []):
                values.append(view.get("local_video_path") or view.get("primary_video_path"))

    failure_candidates: list[dict[str, Any]] = []
    if isinstance(run_meta.get("failure_report"), dict):
        failure_candidates.append(run_meta["failure_report"])
    failure_candidates.extend(report.get("report") for report in failure_reports if isinstance(report.get("report"), dict))
    for report in failure_candidates:
        evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
        for video in evidence.get("video_inputs", []):
            if isinstance(video, dict):
                values.append(video.get("local_video_path") or video.get("primary_video_path"))
    return [value for value in values if value]


def _load_video_manifests(artifact_dirs: list[Path]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for directory in artifact_dirs:
        for name in VIDEO_MANIFEST_NAMES:
            path = directory / name
            data = _read_json(path)
            if data:
                manifests.append({"name": name, "path": path, "data": data})
    return manifests


def _video_path_values(value: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key in VIDEO_SCALAR_PATH_KEYS:
            candidate = value.get(key)
            if candidate:
                values.append(candidate)
        for key in VIDEO_LIST_PATH_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, list):
                values.extend(item for item in candidate if item)
            elif candidate:
                values.append(candidate)
        for key in ("output", "summary"):
            values.extend(_video_path_values(value.get(key)))
        tasks = value.get("tasks")
        if isinstance(tasks, list):
            values.extend(_video_path_values(tasks))
    elif isinstance(value, list):
        for item in value:
            values.extend(_video_path_values(item))
    return values


def _resolve_video_path(
    repo: Path,
    artifact_dirs: list[Path],
    value: Any,
    *,
    experiment_id: str | None = None,
) -> Path | None:
    if not value:
        return None
    raw_text = str(value)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_text):
        return None
    raw = Path(raw_text)
    candidates: list[Path] = []
    if raw.is_absolute():
        _append_path(candidates, raw)
    else:
        _append_path(candidates, repo / raw)
        _append_path(candidates, raw)
    suffixes = _experiment_path_suffixes(raw, experiment_id)
    for directory in artifact_dirs:
        if raw.is_absolute():
            for suffix in suffixes:
                _append_path(candidates, directory / suffix)
            _append_path(candidates, directory / raw.name)
        else:
            _append_path(candidates, directory / raw)
            for suffix in suffixes:
                _append_path(candidates, directory / suffix)
            _append_path(candidates, directory / raw.name)
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve() if candidates else None


def _experiment_path_suffixes(path: Path, experiment_id: str | None) -> list[Path]:
    suffixes: list[Path] = []
    parts = path.parts
    if experiment_id:
        for index, part in enumerate(parts):
            if part == experiment_id and index + 1 < len(parts):
                _append_path(suffixes, Path(*parts[index + 1 :]))
    for index, part in enumerate(parts):
        if part == "experiments" and index + 2 < len(parts):
            _append_path(suffixes, Path(*parts[index + 2 :]))
    return suffixes


def _append_path(paths: list[Path], path: Path) -> None:
    if str(path) not in {str(existing) for existing in paths}:
        paths.append(path)


def _video_row(kind: str, path: Path, label: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "label": label,
        "exists": path.exists(),
    }


def _dedupe_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for video in videos:
        key = f"{video['kind']}:{video['path']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(video)
    return deduped


def _video_sort_key(video: dict[str, Any]) -> tuple[int, str]:
    name = Path(str(video.get("path") or "")).name.lower()
    telemetry_rank = 1 if video.get("kind") == "run" and "telemetry" in name else 0
    return (telemetry_rank, str(video.get("label") or name))


def _source_label(value: Any) -> str:
    path = Path(str(value))
    parent = path.parent.name
    if parent in {"front", "side", "diagonal"}:
        return f"{parent.title()} source render"
    return "Source render"


def _rationale(patch: dict[str, Any], failure_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "hypothesis": patch.get("hypothesis"),
        "expected_effect": patch.get("expected_effect"),
        "risk": patch.get("risk"),
        "rollback": patch.get("rollback"),
        "primary_failure": failure_report.get("primary_failure"),
        "secondary_failures": failure_report.get("secondary_failures") or [],
        "likely_causes": failure_report.get("likely_causes") or [],
        "suggested_research_directions": failure_report.get("suggested_research_directions") or [],
    }


def _change_row(file_path: str, parameter: str, old_value: Any, new_value: Any) -> dict[str, Any]:
    return {
        "section": _section_label(parameter),
        "parameter": parameter,
        "old": format_value(old_value),
        "new": format_value(new_value),
        "old_value": old_value,
        "new_value": new_value,
        "file": file_path,
    }


def _section_label(parameter: str) -> str:
    prefix = parameter.split(".", 1)[0]
    return SECTION_LABELS.get(prefix, prefix.replace("_", " ").title())


def _infer_file(parameter: str, allowed_files: list[Any]) -> str:
    prefix = parameter.split(".", 1)[0]
    for file_path in allowed_files:
        text = str(file_path)
        if prefix in text:
            return text
    return "recorded patch"


def _base_experiment_id(experiment_id: str) -> str:
    value = re.sub(r"_iter-\d+_seed-\d+$", "", experiment_id)
    return re.sub(r"_seed-\d+$", "", value)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_paths(paths: Any) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        path = Path(path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
