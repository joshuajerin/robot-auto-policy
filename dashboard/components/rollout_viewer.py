"""Rollout artifact helpers."""

from __future__ import annotations

from pathlib import Path


def list_rollout_videos(artifact_root: str | Path) -> list[str]:
    root = Path(artifact_root)
    if not root.exists():
        return []
    return [str(path) for path in sorted(root.rglob("*.mp4"))]

