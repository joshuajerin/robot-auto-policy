"""Submit RoboGenesis jobs to a deployed Modal app.

Ephemeral `modal run` apps are useful while developing, but the orchestration
loop should target a deployed app so submissions are stable and not tied to a
local entrypoint lifetime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import modal


DEFAULT_APP_NAME = "robogenesis-isaac-autoresearch"
PHASE1_FUNCTION = "phase1_baseline_job"
RENDER_FUNCTION = "render_isaac_h1_video_job"


def submit_phase1_specs_to_deployed(
    specs: list[dict[str, Any]],
    *,
    app_name: str = DEFAULT_APP_NAME,
    environment_name: str | None = None,
) -> list[str]:
    function = modal.Function.from_name(app_name, PHASE1_FUNCTION, environment_name=environment_name)
    call_ids: list[str] = []
    for spec in specs:
        call = function.spawn(json.dumps(spec, sort_keys=True))
        call_ids.append(call.object_id)
    return call_ids


def submit_render_specs_to_deployed(
    specs: list[dict[str, Any]],
    *,
    app_name: str = DEFAULT_APP_NAME,
    environment_name: str | None = None,
) -> list[str]:
    function = modal.Function.from_name(app_name, RENDER_FUNCTION, environment_name=environment_name)
    call_ids: list[str] = []
    for spec in specs:
        call = function.spawn(json.dumps(spec, sort_keys=True))
        call_ids.append(call.object_id)
    return call_ids


def load_specs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("spec file must contain a JSON object or list of JSON objects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--env", default="")
    parser.add_argument("--function", choices=("phase1", "render"), default="phase1")
    args = parser.parse_args()

    specs = load_specs(Path(args.spec_file))
    submit = submit_render_specs_to_deployed if args.function == "render" else submit_phase1_specs_to_deployed
    call_ids = submit(specs, app_name=args.app_name, environment_name=args.env or None)
    print(json.dumps({"app_name": args.app_name, "function_call_ids": call_ids}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
