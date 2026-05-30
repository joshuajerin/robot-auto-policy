# Phase 1 Runbook: H1 Baseline

Phase 1 is the non-agent baseline:

1. train an H1 locomotion policy in Isaac Lab
2. save a checkpoint
3. evaluate fixed held-out seeds
4. render an actual Isaac H1 rollout video
5. write raw metrics, score JSON, and artifact manifest

## Prerequisites

- Modal CLI authenticated
- Isaac Lab container access
- enough Modal GPU quota for an A10G job
- optional: `OPENAI_API_KEY` for planner setup later

## Generate A Spec

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_001 \
  --config configs/locomotion/phase1_h1.yaml
```

For a short systems test:

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_smoke \
  --max-iterations 2 \
  --num-envs 64
```

## Launch On Modal

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_001 \
  --launch-modal
```

Equivalent explicit Modal command:

```bash
modal run modal_runner/modal_app.py \
  --action phase1 \
  --experiment-spec-json "$(python modal_runner/phase1.py --experiment baseline_h1_001 | python -c 'import sys,json; print(json.dumps(json.load(sys.stdin)))')"
```

## Artifacts

The phase-1 job writes to the `robogenesis-runs` Modal volume:

```text
/runs/experiments/<experiment_id>/
  experiment_spec.json
  h1_asset_report.json
  raw_eval_metrics.json
  eval_metrics.json
  artifact_manifest.json
  logs/
    rsl_rl/
      ...
        model_*.pt
        videos/
          ...
```

Download:

```bash
modal volume get robogenesis-runs /experiments ./modal_experiments
```

## Notes For Robot Researchers

- The first score is not the final robustness score. Generated scenario success
  remains zero until phase 5 enables push/slope/roughness evaluations.
- Fixed seed evaluation protects the baseline from changing while the agent
  later explores generated frontier scenarios.
- The H1 USD is not vendored. `h1_asset_report.json` records the runtime asset
  candidates found inside the Isaac Lab container, and the rollout video comes
  from the actual Isaac H1 task.
