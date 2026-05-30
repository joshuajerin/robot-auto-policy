"""Prepare a V1 locomotion video prompt artifact.

The artifact is intentionally style/context only. It does not retarget human
motion to H1 joints; it records a walking style summary that can condition
planner prompts and experiment metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.video_context import summarize_locomotion_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--license", default="")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    if not args.url and not args.input:
        raise SystemExit("Provide --url or --input")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        suffix = Path(urllib.request.urlparse(args.url).path).suffix or ".mp4"
        video_path = output_dir / f"source_video{suffix}"
        request = urllib.request.Request(
            args.url,
            headers={"User-Agent": "RoboGenesis research prototype"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            video_path.write_bytes(response.read())
        source_url = args.url
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        video_path = output_dir / f"source_video{input_path.suffix or '.mp4'}"
        shutil.copy2(input_path, video_path)
        source_url = None

    style_context = summarize_locomotion_video(
        video_path,
        source_url=source_url,
        license_name=args.license or None,
        description=args.description or None,
    )
    context_path = output_dir / "style_context.json"
    context_path.write_text(json.dumps(style_context, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"video": str(video_path), "style_context": str(context_path)}, indent=2))


if __name__ == "__main__":
    main()
