# Reviewer Prompt

Review a candidate policy improvement.

Reject if:

- fixed evaluator score did not improve enough
- base task regressed by more than 5%
- generated scenario success regressed
- safety checks failed
- improvement appears on too few held-out seeds
- policy exploits the reward or violates gait constraints

Return a concise accept/reject decision with reasons.
