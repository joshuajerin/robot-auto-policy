"""Fetch the official Unitree H1 USD asset bundle used for visualization.

The upstream Unitree repository stores the USD files through Git LFS. A normal
git clone can fail when the repository's LFS bandwidth is exhausted, so this
script downloads the public media URLs directly and writes a manifest with
content hashes for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


SOURCE_REPO = "https://github.com/unitreerobotics/unitree_model"
SOURCE_COMMIT = "f865c72ffd0c50ad4db6a1be270eb34d22e6221e"
FILES = {
    "usd/h1.usd": "https://media.githubusercontent.com/media/unitreerobotics/unitree_model/main/H1/h1/usd/h1.usd",
    "usd/configuration/h1_base.usd": "https://media.githubusercontent.com/media/unitreerobotics/unitree_model/main/H1/h1/usd/configuration/h1_base.usd",
    "usd/configuration/h1_physics.usd": "https://media.githubusercontent.com/media/unitreerobotics/unitree_model/main/H1/h1/usd/configuration/h1_physics.usd",
    "usd/configuration/h1_sensor.usd": "https://media.githubusercontent.com/media/unitreerobotics/unitree_model/main/H1/h1/usd/configuration/h1_sensor.usd",
    "LICENSE": "https://raw.githubusercontent.com/unitreerobotics/unitree_model/main/LICENSE",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="assets/unitree_h1")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for relative_path, url in FILES.items():
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.refresh or not path.exists():
            _download(url, path)
        entries.append(
            {
                "path": str(path.relative_to(output_dir)),
                "url": url,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "robot_id": "unitree_h1",
        "source_repo": SOURCE_REPO,
        "source_commit_observed": SOURCE_COMMIT,
        "asset_root": str(output_dir),
        "primary_usd": "usd/h1.usd",
        "files": entries,
        "note": (
            "Fetched from the public Unitree H1 USD media URLs. Isaac Lab still "
            "uses its task-native H1 articulation for training; this bundle is "
            "mounted into Modal for explicit asset provenance and non-Vulkan "
            "visualization/export workflows."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _download(url: str, output: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        output.write_bytes(response.read())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
