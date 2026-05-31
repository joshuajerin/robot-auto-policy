"""Prune duplicate review-camera videos while keeping scenario diversity.

The AutoResearch loop should compare different generated environments, not
three near-identical views of one environment. This tool keeps a bounded number
of videos per rendered scenario directory and moves extras into a quarantine
folder so nothing is destroyed during fast iteration.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VIEW_NAMES = {"front", "side", "diagonal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/multiview_lambda"))
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--preferred-views", default="side,diagonal,front")
    parser.add_argument("--quarantine-name", default="pruned_duplicate_views")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    preferred = [view.strip() for view in args.preferred_views.split(",") if view.strip()]
    groups = group_review_videos(root, args.quarantine_name)
    actions = []
    for group, videos in sorted(groups.items()):
        keep, prune = split_keep_prune(videos, preferred, max(1, args.max_videos))
        for video in prune:
            actions.extend(prune_actions(root, group, video, args.quarantine_name))

    if args.apply:
        for source, destination in actions:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
        rewritten_jsons = rewrite_review_jsons(root, args.quarantine_name)
    else:
        rewritten_jsons = []

    summary = {
        "root": str(root),
        "applied": args.apply,
        "groups_scanned": len(groups),
        "files_to_move": len(actions),
        "rewritten_jsons": [str(path) for path in rewritten_jsons],
        "actions": [{"source": str(source), "destination": str(destination)} for source, destination in actions],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def group_review_videos(root: Path, quarantine_name: str = "pruned_duplicate_views") -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for video in root.rglob("*.mp4"):
        if quarantine_name in video.parts:
            continue
        if video.parent.name not in VIEW_NAMES:
            continue
        groups.setdefault(video.parent.parent, []).append(video)
    return groups


def split_keep_prune(videos: list[Path], preferred: list[str], max_videos: int) -> tuple[list[Path], list[Path]]:
    def sort_key(path: Path) -> tuple[int, str]:
        view = path.parent.name
        priority = preferred.index(view) if view in preferred else len(preferred)
        return priority, str(path)

    ordered = sorted(videos, key=sort_key)
    keep = ordered[:max_videos]
    prune = [video for video in ordered if video not in set(keep)]
    return keep, prune


def prune_actions(root: Path, group: Path, video: Path, quarantine_name: str) -> list[tuple[Path, Path]]:
    actions: list[tuple[Path, Path]] = []
    quarantine_root = root / quarantine_name
    view = video.parent.name
    for source in [video, group / f"{view}_diagnostics.json"]:
        if not source.exists():
            continue
        destination = quarantine_root / source.relative_to(root)
        actions.append((source, destination))
    return actions


def rewrite_review_jsons(root: Path, quarantine_name: str) -> list[Path]:
    rewritten: list[Path] = []
    for render_dir in sorted(root.rglob("lambda_render")):
        if quarantine_name in render_dir.parts:
            continue
        diagnostics_files = [
            path
            for path in sorted((render_dir / "artifacts").rglob("*_diagnostics.json"))
            if path.name != "multiview_diagnostics.json" and quarantine_name not in path.parts
        ]
        if not diagnostics_files:
            continue
        combined = combine_diagnostics(diagnostics_files)
        target = (
            render_dir / "scenario_autoresearch_input.json"
            if (render_dir / "artifacts" / "scenarios").exists()
            else render_dir / "multiview_autoresearch_input.json"
        )
        target.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
        rewritten.append(target)
        legacy = render_dir / "multiview_autoresearch_input.json"
        if target != legacy:
            legacy.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
            rewritten.append(legacy)
        summary_path = render_dir / "multiview_handoff_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["diagnostics"] = str(target)
            summary["per_view_diagnostics"] = [str(path) for path in diagnostics_files]
            summary["videos"] = [str(path) for path in sorted(render_dir.rglob("*.mp4")) if quarantine_name not in path.parts]
            summary["videos_per_scenario"] = max_videos_per_scenario(summary["videos"])
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            rewritten.append(summary_path)
    return rewritten


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
    scenario_ids = sorted({str(view.get("scenario_id")) for view in views if view.get("scenario_id")})
    primary_failures = [str(item.get("primary_failure", "")) for item in diagnoses if item.get("primary_failure")]
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


def max_videos_per_scenario(video_paths: list[str]) -> int:
    counts: dict[str, int] = {}
    for value in video_paths:
        path = Path(value)
        if path.parent.name not in VIEW_NAMES:
            continue
        scenario_dir = str(path.parent.parent)
        counts[scenario_dir] = counts.get(scenario_dir, 0) + 1
    return max(counts.values(), default=0)


if __name__ == "__main__":
    main()
