# Phase 1 Runbook: H1 Baseline

Phase 1 is the non-agent baseline:

1. train an H1 locomotion policy in Isaac Lab
2. save a checkpoint
3. evaluate fixed held-out seeds
4. render a non-Vulkan telemetry rollout MP4
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

## Add A Research Motion Reference

Version 1 uses research motion capture as style context, not direct motion
retargeting. The default source is CMU subject 07 walking motion in ASF/AMC
format.

```bash
python tools/prepare_motion_reference.py \
  --motion-id 07_01 \
  --output-dir artifacts/motion_references/cmu_07_01

python modal_runner/phase1.py \
  --experiment baseline_h1_cmu_walk_001 \
  --motion-context artifacts/motion_references/cmu_07_01/motion_context.json \
  --launch-modal \
  --detach
```

The motion context is copied into the Modal artifact directory as
`motion_context.json`, and its nested style context is also copied as
`style_context.json` for planner compatibility.

`--detach` submits direct detached Modal function calls, so the training inputs
are not tied to a local log tail. The default phase-1 config runs only 10 PPO
iterations with one held-out eval seed so train/eval/render can be debugged
quickly before scaling.

Phase 1 does not call Isaac camera rendering. `evaluate_rsl_rl_policy.py`
records rollout telemetry, and `render_telemetry_video.py` converts that trace
to MP4 without Vulkan, RTX, `play.py --video`, or `--enable_cameras`.

## Autoscaled Batch

The Modal app is configured for up to four concurrent phase-1 containers using
H100 GPUs by default.

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_cmu_walk \
  --motion-context artifacts/motion_references/cmu_07_01/motion_context.json \
  --num-runs 4 \
  --seed-start 42 \
  --launch-modal \
  --detach
```

This launches:

```text
baseline_h1_cmu_walk-seed-42
baseline_h1_cmu_walk-seed-43
baseline_h1_cmu_walk-seed-44
baseline_h1_cmu_walk-seed-45
```

Each seed is submitted separately and can be scheduled on its own H100-backed
container.

## Deployed App Submission

For the orchestration loop, deploy the app once and submit future jobs by name:

```bash
modal deploy modal_runner/modal_app.py --name robogenesis-isaac-autoresearch

python modal_runner/phase1.py \
  --experiment baseline_h1_cmu_walk \
  --motion-context artifacts/motion_references/cmu_07_01/motion_context.json \
  --num-runs 4 \
  --seed-start 42 \
  --launch-modal \
  --use-deployed
```

This uses `modal.Function.from_name(...).spawn(...)` and avoids creating
one-off ephemeral apps for normal orchestration.

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
  rollout_trace.json
  rollout_telemetry.mp4
  rollout_videos.json
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

Check that each run has the policy checkpoint, eval metrics, manifest, and
non-Vulkan rollout video needed for review:

```bash
python tools/modal_artifact_status.py \
  --experiment baseline_h1_cmu_walk-seed-42 \
  --experiment baseline_h1_cmu_walk-seed-43
```

`ready_for_review` is true only when the run has metrics, a checkpoint, a
manifest, a rollout telemetry trace, and at least one `.mp4` rollout video. The
manifest records `rollout_trace_path`, `rollout_video_paths`, and relative
`rollout_video_files`, so downloaded artifacts can still find the videos
locally.

## Notes For Robot Researchers

- The first score is not the final robustness score. Generated scenario success
  remains zero until phase 5 enables push/slope/roughness evaluations.
- Fixed seed evaluation protects the baseline from changing while the agent
  later explores generated frontier scenarios.
- The H1 USD is not vendored. `h1_asset_report.json` records the runtime asset
  candidates found inside the Isaac Lab container, and the rollout video comes
  from the actual Isaac H1 task.
