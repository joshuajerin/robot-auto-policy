"""Inspect and export the Unitree H1 asset inside an Isaac Lab container.

This script runs inside Modal. It reports the H1 asset that Isaac Lab resolves
for the task and, when a bundled public Unitree H1 USD exists, exports that
bundle into experiment artifacts for provenance and non-Vulkan visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


H1_PATTERNS = (
    "*H1*.usd",
    "*h1*.usd",
    "*Unitree*H1*.usd",
    "*unitree*h1*.usd",
)


def find_h1_usd(search_roots: list[Path]) -> list[str]:
    matches: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in H1_PATTERNS:
            for path in root.rglob(pattern):
                lower = str(path).lower()
                if "h1" in lower and path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                    matches.add(str(path))
    return sorted(matches)


def inspect_task_asset(task: str) -> dict[str, Any]:
    report: dict[str, Any] = {"resolved": False, "candidates": [], "error": None}
    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except Exception as exc:  # pragma: no cover - requires Isaac Lab
        report["error"] = f"import failed: {type(exc).__name__}: {exc}"
        return report

    try:
        env_cfg = parse_env_cfg(task, device="cpu", num_envs=1)
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
        spawn_cfg = getattr(robot_cfg, "spawn", None)
        candidates: list[dict[str, str | None]] = []
        for label, obj in (("robot.spawn", spawn_cfg), ("scene.robot", robot_cfg)):
            if obj is None:
                continue
            for attr in ("usd_path", "asset_path", "prim_path"):
                value = getattr(obj, attr, None)
                if isinstance(value, str) and value:
                    candidates.append(
                        {
                            "object": label,
                            "attribute": attr,
                            "value": value,
                            "resolved_path": resolve_asset_path(value),
                        }
                    )
        report["resolved"] = bool(candidates)
        report["candidates"] = candidates
    except Exception as exc:  # pragma: no cover - requires Isaac Lab
        report["error"] = f"parse failed: {type(exc).__name__}: {exc}"
    return report


def resolve_asset_path(value: str) -> str | None:
    expanded = os.path.expandvars(value)
    for token in re.findall(r"\{([^}]+)\}", expanded):
        replacement = os.environ.get(token)
        if replacement:
            expanded = expanded.replace("{" + token + "}", replacement)
    path = Path(expanded)
    return str(path) if path.exists() else None


def export_asset_bundle(candidates: list[str], export_dir: Path, bundled_asset_dir: Path | None) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "export_dir": str(export_dir),
        "exported": False,
        "source": None,
        "files": [],
        "error": None,
    }
    source_root: Path | None = None
    if bundled_asset_dir and (bundled_asset_dir / "usd" / "h1.usd").exists():
        source_root = bundled_asset_dir
    else:
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                source_root = path.parent.parent if path.parent.name == "configuration" else path.parent
                break
    if source_root is None:
        report["error"] = "no local H1 asset bundle found to export"
        return report

    target = export_dir / source_root.name
    try:
        if source_root.is_dir():
            shutil.copytree(source_root, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root, target)
        report["exported"] = True
        report["source"] = str(source_root)
        report["files"] = [str(path.relative_to(target)) for path in sorted(target.rglob("*")) if path.is_file()]
    except Exception as exc:  # pragma: no cover - environment-specific filesystem failure
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--bundled-asset-dir", "--official-asset-root", default="")
    args = parser.parse_args()

    bundled_asset_dir = Path(args.bundled_asset_dir) if args.bundled_asset_dir else None
    if bundled_asset_dir and bundled_asset_dir.exists():
        roots = [bundled_asset_dir]
    else:
        roots = [
            Path("/workspace"),
            Path("/root"),
            Path("/isaac-lab"),
            Path("/opt"),
            Path(os.environ.get("ISAACLAB_PATH", "/missing")),
        ]

    matches = find_h1_usd(roots)
    task_asset = inspect_task_asset(args.task)
    task_candidates = [
        str(item.get("resolved_path") or item.get("value"))
        for item in task_asset.get("candidates", [])
        if isinstance(item, dict) and (item.get("resolved_path") or item.get("value"))
    ]
    bundled_asset_path = bundled_asset_dir / "usd" / "h1.usd" if bundled_asset_dir else None
    export_report = None
    if args.export_dir:
        export_report = export_asset_bundle(task_candidates + matches, Path(args.export_dir), bundled_asset_dir)

    primary_bundled_asset = str(bundled_asset_path) if bundled_asset_path and bundled_asset_path.exists() else None
    report = {
        "robot_id": "unitree_h1",
        "task": args.task,
        "task_asset": task_asset,
        "usd_candidates": matches,
        "bundled_asset_path": primary_bundled_asset,
        "resolved_asset_path": task_candidates[0] if task_candidates else primary_bundled_asset or (matches[0] if matches else None),
        "export": export_report,
        "asset_resolution_note": (
            "Isaac Lab simulation uses the task-configured H1 articulation. "
            "The bundled public Unitree H1 USD is exported with artifacts when present."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
