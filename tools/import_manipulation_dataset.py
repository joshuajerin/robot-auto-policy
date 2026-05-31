"""Import manipulation object datasets into local artifacts.

The importer is intentionally separate from source-controlled primitive USD
assets. Large datasets such as YCB are downloaded into `artifacts/datasets/`
and summarized with a manifest that scenario builders can reference.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def load_dataset_config(config_path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"dataset config must be a mapping: {config_path}")
    return data


def plan_dataset_import(config: dict[str, Any], *, object_ids: list[str] | None = None) -> list[dict[str, Any]]:
    selected = set(object_ids or [])
    plan: list[dict[str, Any]] = []
    for obj in config.get("objects", []):
        object_id = str(obj.get("object_id", ""))
        if selected and object_id not in selected:
            continue
        plan.append(
            {
                "object_id": object_id,
                "category": obj.get("category", ""),
                "archive_url": obj.get("archive_url", ""),
                "archive_name": obj.get("expected_archive_name") or Path(str(obj.get("archive_url", ""))).name,
            }
        )
    return plan


def import_dataset(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    object_ids: list[str] | None = None,
    dry_run: bool = False,
    extract: bool = True,
    refresh: bool = False,
) -> dict[str, Any]:
    config = load_dataset_config(config_path)
    plan = plan_dataset_import(config, object_ids=object_ids)
    output = Path(output_dir)
    raw_dir = output / str(config.get("output_layout", {}).get("raw", "raw"))
    extracted_dir = output / str(config.get("output_layout", {}).get("extracted", "extracted"))
    manifest = {
        "dataset_id": config["dataset_id"],
        "dataset_name": config["dataset_name"],
        "source_homepage": config.get("source_homepage", ""),
        "license_note": config.get("license_note", ""),
        "dry_run": dry_run,
        "objects": [],
    }

    for item in plan:
        archive_path = raw_dir / item["archive_name"]
        extracted_path = extracted_dir / item["object_id"]
        record = {
            **item,
            "archive_path": str(archive_path),
            "extracted_path": str(extracted_path),
            "downloaded": False,
            "extracted": False,
        }
        if not dry_run:
            raw_dir.mkdir(parents=True, exist_ok=True)
            extracted_path.mkdir(parents=True, exist_ok=True)
            _download_if_needed(item["archive_url"], archive_path, refresh=refresh)
            record["downloaded"] = True
            if extract:
                _safe_extract_tgz(archive_path, extracted_path)
                record["extracted"] = True
        manifest["objects"].append(record)

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dataset_manifest.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _download_if_needed(url: str, destination: Path, *, refresh: bool) -> None:
    if destination.exists() and not refresh:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "RoboGenesis dataset importer"})
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())


def _safe_extract_tgz(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = output_dir / member.name
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"unsafe archive member path: {member.name}")
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/ycb_core_subset.yaml")
    parser.add_argument("--output-dir", default="artifacts/datasets/ycb_core_subset")
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    summary = import_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        object_ids=args.object_id or None,
        dry_run=args.dry_run,
        extract=not args.no_extract,
        refresh=args.refresh,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
