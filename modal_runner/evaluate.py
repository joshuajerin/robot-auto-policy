"""Placeholder evaluator entrypoint for Isaac-produced metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.locomotion_score import score_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    scored = score_metrics(json.loads(Path(args.raw_metrics).read_text()))
    output = json.dumps(scored, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()

