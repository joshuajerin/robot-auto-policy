# Locomotion Failure Diagnosis Prompt

Classify the policy failure using rollout metrics, videos, contact logs, base
pose logs, joint logs, and termination reasons.

Return a JSON object with:

- `primary_failure`
- `secondary_failures`
- `evidence`
- `likely_causes`
- `suggested_research_directions`

Use the locomotion failure taxonomy from the adapter. Do not propose evaluator
changes.
