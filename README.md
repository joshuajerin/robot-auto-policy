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

The command writes SQLite research memory to `artifacts/research.db` by default.
Use a different path when testing:

```bash
python -m core.autoresearch_loop \
  --dry-run \
  --experiments 5 \
  --db artifacts/local_research.db
```

## Tests

```bash
pytest -q
```

The tests cover:

- locked patch validation
- YAML dry-run patch application
- accept/reject scoring
- scenario generation
- SQLite research memory
- the deterministic dry-run controller

## Phase 1 H1 Baseline

Generate the H1 baseline experiment spec:

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_001 \
  --config configs/locomotion/phase1_h1.yaml
```

Launch the full phase-1 job on Modal:

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_001 \
  --launch-modal
```

Launch with the normal-walking video style prompt:

```bash
python tools/prepare_video_prompt.py \
  --url https://commons.wikimedia.org/wiki/Special:Redirect/file/Big_City_Life.webm \
  --output-dir artifacts/video_prompts/normal_walk \
  --license "CC0 1.0" \
  --description "Public-domain video clip containing people walking normally in an urban setting."

python modal_runner/phase1.py \
  --experiment baseline_h1_normal_walk_001 \
  --style-context artifacts/video_prompts/normal_walk/style_context.json \
  --launch-modal \
  --detach
```

Autoscaled multi-seed launch:

```bash
python modal_runner/phase1.py \
  --experiment baseline_h1_normal_walk \
  --style-context artifacts/video_prompts/normal_walk/style_context.json \
  --num-runs 4 \
  --seed-start 42 \
  --launch-modal \
  --detach
```

Each seed is submitted as a direct detached Modal function call, allowing Modal
to schedule multiple H100-backed jobs concurrently.

That Modal job uses `Isaac-Velocity-Flat-H1-v0`, trains with `rsl_rl`, resolves
the H1 asset inside the Isaac Lab container, evaluates fixed held-out seeds,
renders an actual H1 rollout video, and writes:

```text
experiment_spec.json
h1_asset_report.json
raw_eval_metrics.json
eval_metrics.json
artifact_manifest.json
checkpoint and video artifacts under logs/
```

See `docs/phase1_runbook.md` for the full runbook.

## Isaac Lab Modal Workflow

The Modal entrypoint is intentionally separated from the research controller so the evaluator and runner can be locked.

```bash
modal run modal_runner/modal_app.py --action smoke

python modal_runner/train.py \
  --experiment baseline_001 \
  --task Isaac-Velocity-Flat-H1-v0 \
  --num-envs 4096 \
  --max-iterations 1000

modal run modal_runner/modal_app.py \
  --action train-and-eval \
  --experiment-spec-json '{"experiment_id":"baseline_001","task":"Isaac-Velocity-Flat-H1-v0","num_envs":4096,"max_iterations":1000}'
```

Score a raw metrics file with the locked evaluator:

```bash
python modal_runner/evaluate.py \
  --raw-metrics artifacts/baseline_001/raw_metrics.json \
  --output artifacts/baseline_001/eval_metrics.json
```

Run the dashboard after a dry run:

```bash
streamlit run dashboard/app.py -- --db artifacts/research.db
```

## OpenAI Setup

Copy `.env.example` to `.env` or export:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.4-mini"
```

The OpenAI-backed planner returns a schema-constrained `PatchSpec`; if no key is
set, the deterministic planner fallback is used. See `docs/openai_setup.md`.

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
