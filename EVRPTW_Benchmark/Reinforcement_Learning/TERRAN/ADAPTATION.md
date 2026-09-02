# TERRAN Canonical-Stage-2 Adaptation Record

## Retained method

This adapter migrates the repository's existing TERRAN implementation; it does
not replace TERRAN with a new neural baseline. The following remain unchanged:

- the attention encoder and autoregressive pointer decoder;
- the previous-node plus load/battery/time decoder context;
- POMO-style parallel trajectories;
- the actor-critic and PPO update structure;
- configurable customer-progress, repair-distance, feasible-ratio, and
  terminal auxiliary shaping.

PBRS settings are part of the method configuration and are recorded in every
checkpoint. They are not used as cross-method evaluation scores.

## Canonical benchmark adaptations

| Component | Legacy TERRAN path | Canonical Stage-2 path |
|---|---|---|
| Training source | online or consolidated legacy instances | frozen Stage-2 `view_index` and matrix families |
| Objective-facing reward | normalized route distance | normalized directed `distance_matrix_km` increment |
| Travel time and energy | optionally derived from legacy scalars | canonical released matrices |
| Charging | legacy fixed/proportional full charge | station-power, derated, full charge |
| Fleet | implementation route counter | unlimited homogeneous fleet under the common contract |
| Evaluation | environment completion | environment completion plus independent shared verifier |
| Reporting | aggregate legacy CSV | per-instance distance, feasibility, vehicle, charging, runtime, and routes |

Legacy configurations remain explicitly marked with `legacy_fixed_full` and
`legacy_derived`. They cannot silently produce canonical benchmark results.
Canonical runs must use `configs/stage2_cus100_terran.yaml` (with the scale and
terminal count changed together for other tracks) or an equivalent recorded
configuration.

## Reward boundary

The base reward minimizes directed-road distance. TERRAN's auxiliary shaping
is retained when enabled, and the checkpoint records each coefficient and its
annealing schedule. Final benchmark ordering ignores shaped return: only
complete routes accepted by the common verifier are ranked by physical
distance in kilometres.
