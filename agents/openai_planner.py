"""OpenAI-backed PatchSpec planner with deterministic fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.openai_client import create_structured_json, has_openai_key
from agents.planner import propose_locomotion_patch
from core.schemas import PatchSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCHEMA = REPO_ROOT / "specs" / "patch_spec.schema.json"
PLANNER_PROMPT = REPO_ROOT / "agents" / "prompts" / "locomotion_planner.md"


def propose_patch_with_openai(context: dict[str, Any], use_fallback: bool = True) -> PatchSpec:
    if not has_openai_key():
        if use_fallback:
            return propose_locomotion_patch(context)
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = create_structured_json(
        schema_path=PATCH_SCHEMA,
        system_prompt=PLANNER_PROMPT.read_text(),
        user_payload=context,
    )
    return PatchSpec(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-json", required=True)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    context = json.loads(Path(args.context_json).read_text())
    patch = propose_patch_with_openai(context, use_fallback=not args.no_fallback)
    print(json.dumps(patch.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

