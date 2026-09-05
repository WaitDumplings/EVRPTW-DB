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
checkpoint. They are not used as cross-method evaluation scores. The reference
`rollout.py` contains expert-provider and route-boundary collection hooks, but
its trainer/configs do not activate them; those dormant extensions are outside
the formal Stage-2 TERRAN path and are not claimed as migrated components.

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

Formal Stage-2 training binds the registered `training.rollout_steps` value to
an environment truncation. If that horizon is reached before every customer is
served and the route has returned to the depot, the enabled terminal heuristic
adds

```text
-failure_penalty * (remaining_customers / num_customers)
```

to the final training transition (and applies the recorded PBRS annealing scale).
A trajectory completing exactly on the horizon receives the configured success
bonus instead. This shaping is not used for fixed-set benchmark evaluation.
A physical charging station is usable at most once within the current vehicle
route and becomes available again after a depot return. This route-local
anti-cycle rule is inherited from the designated legacy environment; it is not
a global station-copy limit. Station input uses full-charge time divided by the
operating horizon, and the distance reward uses one deterministic scale
estimated only from the frozen training pool.
