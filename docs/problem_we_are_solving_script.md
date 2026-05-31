# Problem We Are Solving Script

## 5-10 Second Version

Robot policy training is slow trial and error. RoboGenesis automates that loop: diagnose a run, patch safe training configs, retrain in Isaac Lab, and keep only policies that score better.

## Technical Demo Walkthrough

Use this while clicking through robot run videos:

"This is the H1 humanoid policy running in Isaac Lab. The motion is coming from an `rsl_rl` PPO checkpoint, not from the AI directly controlling the robot.

For each run, we collect rollout metrics like survival, command tracking, torso stability, foot slip, energy use, and safety failures.

AutoResearch reads those metrics and diagnoses the failure mode. For example, if the robot pitches forward or has unstable gait, it turns that into a structured failure report.

Then the planner proposes one bounded `PatchSpec`. It can only touch approved training configs like rewards, curriculum, terrain, randomization, actuators, or PPO settings. It cannot edit the evaluator, robot asset, safety checks, or direct robot actions.

That patch gets trained again on Modal/Isaac Lab, evaluated on fixed seeds and generated challenge scenarios, then scored against the parent policy.

If the new policy improves enough and passes safety, it becomes the next parent policy. If not, we reject it but keep the result as research evidence for the next iteration."

## Super Short Video Voiceover

"This video shows the robot policy rollout. Under the hood, RoboGenesis scores the run, diagnoses what failed, proposes a safe training-config patch, retrains in simulation, and only accepts the new policy if fixed tests and safety checks improve."

## 30-Second Version

Robots are hard to improve because every policy change usually needs a robotics expert to design an experiment, edit training settings, run simulations, inspect failures, and decide what to try next. That loop is slow, manual, and easy to make inconsistent.

RoboGenesis turns that process into a controlled research loop. It proposes one small policy-training change, runs it in simulation, evaluates it on locked tests, diagnoses failures, generates harder scenarios, and only keeps the change if it improves the robot without breaking safety or robustness.

The goal is not to let an AI directly control a robot. The goal is to automate the research workflow around robot policy training.

## 2-Minute Version

The problem we are solving is the bottleneck in robot learning.

Today, training a robot policy is not just "press train and get a better robot." A researcher has to choose reward weights, curriculum settings, randomization ranges, terrain settings, PPO parameters, and safety constraints. Then they run a simulation, watch what failed, guess why it failed, change the setup, and run the next experiment.

That is expensive and slow. It also does not scale well, because each new robot task needs many rounds of expert trial and error.

RoboGenesis is an autonomous outer loop for that process.

For the first version, we focus on Unitree H1 humanoid locomotion in Isaac Lab. The system starts with a baseline walking policy. Then it runs a loop:

```text
propose -> patch -> train -> evaluate -> diagnose -> generate scenarios -> keep or revert -> repeat
```

The important part is that the AI is constrained. It cannot change the evaluator. It cannot edit robot assets. It cannot directly send actions or torques to the robot. It can only propose structured changes to approved training config files, like rewards, curriculum, randomization, terrain, actuators, and PPO settings.

Every candidate policy is judged by fixed seeds, generated challenge scenarios, safety checks, and a locked scoring rule. If the policy improves enough and does not regress on robustness, we keep it. If it fails, we record the failure and use it as research evidence for the next experiment.

So the product is not a robot brain. It is a research engine for making robot policies better, faster, and more reproducibly.

## What To Say In A Demo

Start with the pain:

"Robot policy training is still mostly a manual research loop. You try a reward change, train for hours, evaluate it, watch the robot fall, diagnose why, and then decide what experiment to run next. That process is slow and hard to scale."

Explain the system:

"This project automates that outer loop. It proposes a small training change, applies it only to safe config surfaces, launches training and evaluation, scores the result, diagnoses failure modes, generates harder scenarios, and decides whether the new policy should replace the old one."

Clarify safety:

"The AI does not control the robot. It does not change the evaluator. It does not get to move the goalposts. It only proposes bounded research patches, and the simulator plus locked safety checks decide whether those patches survive."

Make the value concrete:

"That gives us a repeatable way to improve robot policies. Every experiment has a hypothesis, patch, expected metric improvement, risk, rollback plan, score, and lineage. Even failed runs become useful evidence."

Close:

"The first target is H1 humanoid locomotion, but the bigger idea is a general AutoResearch loop for robotics: let agents handle the repetitive experiment design and diagnosis while keeping evaluation, safety, and acceptance rules fixed."

## One-Sentence Explanation

RoboGenesis automates the slow human research loop around robot policy training while keeping the robot, evaluator, and safety rules locked down.

## Key Points To Repeat

- We are solving the robotics experimentation bottleneck.
- The system improves training policies, not direct robot control.
- The AI proposes bounded config patches.
- Isaac Lab, PPO, fixed seeds, generated scenarios, and safety checks judge the result.
- Accepted policies must improve without hiding regressions.
- Failed experiments are recorded instead of thrown away.
