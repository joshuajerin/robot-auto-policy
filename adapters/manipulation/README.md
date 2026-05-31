# Manipulation Adapter

Manipulation reuses the same `TaskAdapter` interface as locomotion:

- build task-specific scenarios
- score fixed and generated evaluations
- diagnose failures
- expose a constrained editable training surface

The first implementation focuses on tabletop pick/place with 3D object assets:

- target cube lift
- target pose placement
- cluttered picking
- occluding distractors
- low-friction and high-mass objects
- narrow-bin placement

The adapter is intentionally separate from the Phase-1 H1 locomotion runner.
It can generate and validate AutoResearch experiments now, while a future Isaac
Lab manipulation runner owns actual policy training.
