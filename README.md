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

## Project Layout

This repository is organized around a constrained robotics AutoResearch loop.
The first target is H1 humanoid locomotion in Isaac Lab, with Modal providing
GPU execution.

### Top-Level Files

- `README.md` - project overview, quickstart commands, and run instructions.
- `program.md` - operating rules for the autonomous research agent, including
  editable files, locked files, safety rules, and acceptance criteria.
- `pyproject.toml` - Python package metadata and dependencies.
- `.env.example` - local environment template for `OPENAI_API_KEY` and
  `OPENAI_MODEL`.

### Configs And Assets

- `configs/locomotion/phase1_h1.yaml` - phase-1 H1 baseline config: task,
  runner, train/eval/render settings, Modal volume metadata, and autoscale
  notes.
- `configs/locomotion/rewards.yaml` - editable locomotion reward weights.
- `configs/locomotion/curriculum.yaml` - editable curriculum knobs.
- `configs/locomotion/domain_randomization.yaml` - editable friction, push,
  payload, motor strength, and action delay settings.
- `configs/locomotion/actuators.yaml` - safe actuator scaling config.
- `configs/locomotion/ppo.yaml` - PPO settings.
- `configs/locomotion/terrain.yaml` - terrain config for generated challenges.
- `configs/locomotion/eval.yaml` - evaluation config.
- `assets/h1_robot_spec.json` - Unitree H1 robot spec used by the orchestration
  layer.
- `assets/h1_isaac_reference.usda` - lightweight reference stub; the actual H1
  USD is resolved inside the Isaac Lab container.

### Modal And Isaac Execution

- `modal_runner/modal_app.py` - Modal app with H1 train/evaluate/render
  functions, artifact syncing, scoring, and manifest creation.
- `modal_runner/phase1.py` - CLI for generating phase-1 specs and launching H1
  runs locally, detached, batched, or through the deployed app.
- `modal_runner/train.py` - helper for building training specs.
- `modal_runner/evaluate.py` - helper for scoring raw metrics files.
- `modal_runner/render.py` - helper for rollout render specs.
- `modal_runner/isaac_scripts/inspect_h1_asset.py` - runs inside Isaac Lab to
  report H1 asset resolution.
- `modal_runner/isaac_scripts/evaluate_rsl_rl_policy.py` - evaluates an rsl_rl
  checkpoint on fixed seeds.
- `modal_runner/isaac_scripts/write_artifact_manifest.py` - writes the final
  experiment artifact manifest.

### AutoResearch Core

- `core/autoresearch_loop.py` - local dry-run AutoResearch loop.
- `core/experiment_db.py` - SQLite research memory for experiments, policies,
  scenarios, scenario evals, and failure reports.
- `core/patch_validator.py` - deterministic `PatchSpec` validator and safe YAML
  patcher.
- `core/scoring.py` - locked locomotion score and accept/reject rule.
- `core/schemas.py` - shared dataclasses for robot/task/scenario/patch/score
  objects.

### Agents And Adapters

- `adapters/base.py` - generic `TaskAdapter` protocol.
- `adapters/locomotion/adapter.py` - locomotion task adapter.
- `adapters/locomotion/scenario_generator.py` - generated locomotion challenge
  specs.
- `adapters/locomotion/failure_diagnosis.py` - metric-based locomotion failure
  taxonomy.
- `adapters/locomotion/metrics.py` - locomotion score adapter.
- `adapters/manipulation/` - stub for future manipulation support.
- `agents/planner.py` - deterministic fallback research planner.
- `agents/openai_client.py` - OpenAI Responses API structured-output helper.
- `agents/openai_planner.py` - OpenAI-backed `PatchSpec` planner.
- `agents/reviewer.py` - candidate policy reviewer.
- `agents/scenario_agent.py` - scenario generation wrapper.
- `agents/video_context.py` - normal-walking video style context extraction.
- `agents/prompts/` - planner, reviewer, failure diagnosis, and scenario
  prompts.

### Evaluation, Dashboard, Tools, And Tests

- `eval/fixed_eval_seeds.json` - locked held-out evaluation seeds.
- `eval/locomotion_score.py` - CLI wrapper for locked scoring.
- `eval/safety_checks.py` - safety checks outside the editable agent surface.
- `dashboard/app.py` - Streamlit dashboard skeleton for research memory.
- `dashboard/components/` - dashboard loaders for experiments, metrics,
  scenarios, and rollout videos.
- `tools/prepare_video_prompt.py` - prepares a walking-video style context
  artifact.
- `specs/` - JSON Schemas for `RobotSpec`, `TaskSpec`, `ScenarioSpec`,
  `PatchSpec`, and `EvalSpec`.
- `docs/phase1_runbook.md` - full H1 phase-1 runbook.
- `docs/openai_setup.md` - OpenAI key/model setup.
- `docs/implementation_log.md` - implementation history.
- `tests/` - unit tests for patch safety, scoring, scenarios, DB, phase-1
  specs, video context, OpenAI fallback, and dry-run orchestration.

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

Persistent deployed-app orchestration:

```bash
modal deploy modal_runner/modal_app.py --name robogenesis-isaac-autoresearch

python modal_runner/phase1.py \
  --experiment baseline_h1_normal_walk \
  --style-context artifacts/video_prompts/normal_walk/style_context.json \
  --num-runs 4 \
  --seed-start 42 \
  --launch-modal \
  --use-deployed
```

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
