# H1 Manipulation Adapter

Manipulation reuses the same `TaskAdapter` interface as locomotion:

- build task-specific scenarios
- score fixed and generated evaluations
- diagnose failures
- expose a constrained editable training surface

This adapter is Unitree H1-only. Isaac Lab 2.0.x does not ship a stock H1
manipulation task in the available container; its stock manipulation examples
are fixed-arm tasks such as Franka and UR10. RoboGenesis therefore targets a
custom environment:

```text
RoboGenesis-H1-Tabletop-Manipulation-v0
```

Do not substitute Franka/UR10 when the task family is `manipulation`; those are
useful only as external reference baselines, not RoboGenesis training targets.

The first implementation focuses on H1 tabletop object interaction with 3D
assets:

- target cube secure-and-move
- target pose placement
- cluttered target interaction
- occluding distractors
- low-friction and high-mass objects
- narrow-bin placement

The adapter can generate and validate AutoResearch experiments now. Actual H1
manipulation training requires the custom Isaac Lab environment to combine:

- H1 articulation and balance terms from the locomotion stack
- arm/end-effector contact actions
- tabletop object scene generation
- fixed evaluator scenarios and real Isaac camera rendering
