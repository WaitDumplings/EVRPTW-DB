# DRL action constraints: no consecutive charging-station visits

Contract ID: `drl_no_consecutive_cs_v1`.
Scope: AM-EVRPTW, EVRPTW-RL, DRL-TS and TERRAN, in training, validation and test.

## Rule

When the current service terminal is a charging station, every charging-station
action is masked. This prohibits `CS1 -> CS2`, including self-transitions; it
does **not** disable charging stations generally.

- `customer -> CS -> customer` remains available subject to the other masks.
- `CS -> depot` remains available when the existing route and resource rules allow it.
- `CS1 -> customer -> CS2` is not prohibited by this rule.
- The existing per-route prohibition on revisiting the same physical station
  remains; a depot return resets that per-route record.
- DRL-TS retains its additional paper-specific `depot -> CS` mask. This extra
  restriction is not imposed on AM, EVRPTW-RL or TERRAN.

The constraint applies to consecutive selected terminals within one vehicle
route, not to charging facilities passed geographically on a road path. It
does not merge the end of one vehicle route with the start of another.
An attempted masked move follows the existing invalid-action behavior; it
does not accrue fictitious travel distance or a new vehicle fee.

## Return feasibility

The shared slow, optimized and JIT masks use the same rule. A station-return
cache may no longer certify a forbidden multi-station chain. Pure return
certificates consist of a direct depot return or, from a customer, a return
via one available station that can itself reach the depot after full charging.

A candidate station that cannot return directly may still be admitted through
a bounded, state-dependent continuation check: after charging, reach an
unserved customer, then return directly or via one available station. This
check respects time windows, capacity, battery energy and the route's station
visit record, including the candidate station just visited. It avoids dropping
the ordinary `CS -> customer -> CS -> depot` branch merely because an old
multi-CS shortcut is no longer legal. It is a conservative lookahead, not a
complete search over all possible future customer sequences.

## Evaluation and provenance

The DRL selector first replays resource feasibility using the unchanged shared
physical verifier, then checks this action restriction. It records
`physical_verifier_passed`, `drl_policy_passed` and the contract ID separately.
A physically feasible CS-to-CS route is not a passing DRL candidate. The final
ranking remains feasible-first, then minimum registered electricity-plus-vehicle
cost; the coefficients and auxiliary shaping are unchanged.

The contract ID is recorded in environment output, manifests, checkpoints,
training/validation results and evaluation exports. Restore/resume/evaluation
reject checkpoints with absent, different or conflicting action-contract IDs.
Use fresh training outputs after pulling this revision; no dataset or shared
ID-stream regeneration is required.

This is an explicit restriction of the DRL search space, not a physical claim
that consecutive charging is impossible or always dominated. Gurobi, ALNS,
VNS-TS, their existing results and the dataset's physical schema are not
modified by this change. Cross-paradigm reporting must disclose this DRL
restriction rather than claim identical admissible route spaces.
