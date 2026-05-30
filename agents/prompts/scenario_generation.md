# Scenario Generation Prompt

Generate physically plausible locomotion scenarios around the current learning
frontier.

Targets:

- too easy: success over 90%
- learning frontier: success 30% to 80%
- too hard: success under 30%
- invalid: physics or asset issue

Return `ScenarioSpec` JSON objects only.
