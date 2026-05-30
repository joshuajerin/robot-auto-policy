# Modal Guardian

The Modal guardian is the phase-1 sidecar. It polls running Modal apps and log
tails while code changes are happening, writes JSONL status events, and surfaces
failure lines immediately.

It does not edit source code, cancel jobs, or mutate the evaluator. Relaunch is
explicit and only happens when a phase-1 spec path is supplied.

## Watch Active Phase-1 Jobs

```bash
python tools/modal_guardian.py \
  --app-id ap-i3aEd782OdKXBfpBI07b42 \
  --app-id ap-btS1Up9GcRt2k6cImod8M9 \
  --app-id ap-f3cCQsRLbkHASiwOD2aM41 \
  --app-id ap-vM7qfXpPLUXaQVwrdZxlXf \
  --iterations 12 \
  --interval-seconds 60 \
  --output artifacts/modal_guardian/events.jsonl
```

## Relaunch Explicitly On Error

Write a known-good phase-1 spec first:

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_cmu_walk_quick \
  --motion-context artifacts/motion_references/cmu_07_01/motion_context.json \
  --num-envs 256 \
  --max-iterations 20 \
  --video-length 120 \
  --write-spec artifacts/specs/baseline_h1_cmu_walk_quick.json
```

Then run the guardian with relaunch enabled:

```bash
python tools/modal_guardian.py \
  --app-id ap-current \
  --phase1-spec artifacts/specs/baseline_h1_cmu_walk_quick.json \
  --relaunch-on-error
```
