"""CLI wrapper for the locked locomotion scoring logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.scoring import score_from_metrics
from eval.safety_checks import run_locomotion_safety_checks


def score_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    safety = run_locomotion_safety_checks(raw_metrics)
    raw_metrics = dict(raw_metrics)
    raw_metrics["safety_passed"] = safety.passed
    if not safety.passed:
        raw_metrics["safety_penalty"] = max(float(raw_metrics.get("safety_penalty", 0.0)), 0.2)
    policy_id = str(raw_metrics.get("policy_id", "candidate_policy"))
    score = score_from_metrics(policy_id, raw_metrics).to_dict()
    score["safety_reasons"] = safety.reasons
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text())
    print(json.dumps(score_metrics(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

