# RoboGenesis / Real2Sim AutoResearch

RoboGenesis is a generalized autonomous training loop for robot policies. It starts with locomotion and turns the manual robotics outer loop into a constrained AutoResearch cycle:

```text
propose -> patch -> train -> evaluate -> diagnose -> generate scenarios -> keep/revert -> repeat
```

The first implementation target is Isaac Lab locomotion on Modal. The system keeps the loop disciplined:

- the research agent submits structured `PatchSpec` objects
- patches are limited to approved locomotion config files
- evaluation, safety checks, seeds, and scoring are locked
- policies are accepted only if fixed and generated evaluations improve without regressions
- every experiment writes artifacts and lineage into SQLite

## MVP

```text
Robot: H1 or G1 humanoid, with AnyMal/Go2 fallback
Task: Isaac-Velocity-Flat-H1-v0
Stretch task: Isaac-Velocity-Rough-H1-v0
Trainer: rsl_rl PPO
Execution: Modal GPU jobs
Editable surface: reward, curriculum, randomization, actuator, terrain, PPO YAML
Evaluator: fixed seeds plus generated locomotion scenarios
```

## Repository Map

```text
program.md                 Agent operating rules
core/                      AutoResearch controller, DB, scoring, validation
specs/                     JSON Schemas for robot/task/scenario/patch/eval specs
adapters/                  Task adapter interface and locomotion implementation
configs/locomotion/        Safe editable configs for the first task family
agents/                    Structured proposal helpers and prompt templates
modal_runner/              Modal/Isaac execution entrypoints
eval/                      Locked seeds and safety/evaluation logic
dashboard/                 Lightweight Streamlit dashboard skeleton
tests/                     Unit tests for local deterministic components
```

## Local Dry Run

The local loop does not require Isaac Lab. It validates patch safety, generates scenarios, computes deterministic toy metrics, applies the accept/reject rule, and records lineage.

```bash
python -m core.autoresearch_loop --dry-run --experiments 3
```

## Isaac Lab Modal Workflow

The Modal entrypoint is intentionally separated from the research controller so the evaluator and runner can be locked.

```bash
modal run modal_runner/modal_app.py::smoke

modal run modal_runner/modal_app.py::train_and_eval \
  --experiment-spec-json '{"experiment_id":"baseline_001","task":"Isaac-Velocity-Flat-H1-v0"}'
```

## Acceptance Rule

```python
new.total_score >= old.total_score + 0.03
new.safety_passed
new.base_success >= old.base_success - 0.05
new.generated_success >= old.generated_success
new.eval_seed_count >= MIN_EVAL_SEEDS
not new.reward_hacking_detected
```

The point is not to let an LLM control a robot. The LLM proposes bounded research changes; Isaac Lab, PPO, the evaluator, and the safety checks decide whether those changes survive.
