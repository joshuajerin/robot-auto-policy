"""Prepare or submit quick AutoResearch iterations against the deployed Modal app."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.artifact_sync import sync_and_ingest_modal_experiment
from core.orchestration import OrchestrationConfig, reconcile_modal_experiments, run_orchestration
from modal_runner.deployed import DEFAULT_APP_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/research.db")
    parser.add_argument("--task-family", default="locomotion")
    parser.add_argument("--phase1-config", default="configs/locomotion/phase1_h1.yaml")
    parser.add_argument("--output-dir", default="artifacts/autoresearch_specs")
    parser.add_argument("--experiment-prefix", default="autoresearch_h1")
    parser.add_argument("--experiments", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--video-length", type=int, default=60)
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Spawn jobs on the deployed Modal app.")
    parser.add_argument("--reconcile", action="store_true", help="Poll Modal artifacts and update running DB rows.")
    parser.add_argument("--sync-artifacts", action="store_true", help="Download Modal artifacts and ingest them into SQLite.")
    parser.add_argument("--modal-volume", default="robogenesis-runs")
    parser.add_argument("--download-dir", default="artifacts/modal_downloads")
    parser.add_argument("--experiment-id", action="append", default=[])
    parser.add_argument("--parent-policy-id", default="")
    parser.add_argument("--accept", action="store_true", help="Force promotion during artifact ingest.")
    parser.add_argument("--rejected", action="store_true")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--env", default="")
    args = parser.parse_args()

    if args.sync_artifacts:
        if not args.experiment_id:
            raise SystemExit("--sync-artifacts requires at least one --experiment-id")
        if args.accept and args.rejected:
            raise SystemExit("--accept and --rejected are mutually exclusive")
        accepted_override = True if args.accept else False if args.rejected else None
        results = [
            sync_and_ingest_modal_experiment(
                experiment_id,
                db_path=args.db,
                volume=args.modal_volume,
                destination_root=args.download_dir,
                parent_policy_id=args.parent_policy_id or None,
                accepted=accepted_override,
                environment_name=args.env or None,
            ).to_dict()
            for experiment_id in args.experiment_id
        ]
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if args.reconcile:
        results = reconcile_modal_experiments(
            Path(args.db),
            experiment_ids=args.experiment_id or None,
            volume=args.modal_volume,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True))
        return

    config = OrchestrationConfig(
        repo_root=REPO_ROOT,
        db_path=Path(args.db),
        task_family=args.task_family,
        phase1_config=Path(args.phase1_config),
        output_dir=Path(args.output_dir),
        experiment_prefix=args.experiment_prefix,
        experiments=args.experiments,
        seed_start=args.seed_start,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        video_length=args.video_length,
        use_openai=args.use_openai,
        submit=args.submit,
        app_name=args.app_name,
        environment_name=args.env or None,
    )
    steps = run_orchestration(config)
    print(json.dumps([step.to_dict() for step in steps], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
