# Research Motion References

Phase 1 uses motion references as style and reward-target context. It does not
retarget human joints into H1 actions yet, and it never lets an LLM emit robot
torques.

## Default Source

The default reference is the CMU Graphics Lab Motion Capture Database subject
07 walking set:

- skeleton: `http://mocap.cs.cmu.edu/subjects/07/07.asf`
- walking motions: `07_01`, `07_02`, `07_03`
- brisk walking motion: `07_12`
- sample rate: 120 Hz

The generated `motion_context.json` is copied into Modal experiment artifacts.
The phase-1 spec also exposes its nested `style_context` to the planner path so
reward/curriculum proposals can bias toward normal mocap walking.

## Prepare A Reference

```bash
python tools/prepare_motion_reference.py \
  --motion-id 07_01 \
  --output-dir artifacts/motion_references/cmu_07_01
```

Launch training with that reference:

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_cmu_walk_001 \
  --motion-context artifacts/motion_references/cmu_07_01/motion_context.json \
  --launch-modal \
  --use-deployed
```

## Why This Replaced The Public Walking Clip

A casual video clip is weak supervision: unknown camera geometry, unknown frame
rate, inconsistent scale, and no skeleton. The CMU reference provides an actual
mocap skeleton and time-indexed joint channels. That is still not full imitation,
but it is a defensible research artifact for style-conditioned locomotion and a
clean stepping stone toward retargeting.
