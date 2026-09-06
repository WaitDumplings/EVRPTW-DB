# DRL-TS Adaptation Record

## Publication and code status

This baseline follows Jinbiao Chen, Huanhuan Huang, Zizhen Zhang, and Jiahai
Wang, *Deep Reinforcement Learning with Two-Stage Training Strategy for
Practical Electric Vehicle Routing Problem with Time Windows*, PPSN 2022,
pp. 356--370, DOI: <https://doi.org/10.1007/978-3-031-14714-2_25>.

The full chapter supplied by the user at `/data/Maojie/ICLR/DRL_TS.pdf` was
audited on 2026-09-03. The paper states that source code is available on
request; no public author-maintained repository was verified. This directory
is therefore a paper-guided PyTorch adaptation, not official code or a claim of
numerical reproduction.

## Paper-verified design

The implementation follows the published method structure:

- a complete directed graph with asymmetric distance and travel time;
- published node features `(demand, earliest time, latest time, node type)`;
- published edge features `(distance, travel time, r-nearest indicator)`, with
  `r=10` in the paper;
- linear node/edge projections followed by two edge-aware GAT layers with
  simultaneous node and edge updates, BatchNorm, ReLU, and skip connections;
- GRU route memory; dynamic `(time, remaining capacity, remaining battery)`
  context; edge-aware multi-head glimpse; and tanh-clipped compatibility;
- Stage 1 keeps tour constraints hard while capacity, time-window, and battery
  violations are soft; Stage 2 masks all three feasibility violations;
- REINFORCE with a greedy rollout baseline.

The published experiment uses embedding dimension 128, eight heads, clipping
constant 10, Adam at `1e-4`, penalties `alpha=beta=gamma=1`, 200 epochs, 250
batches per epoch, and an equal 100/100 soft/hard split. Batch size is 128 for
C10/C20/C50 and 64 for C100. These are paper settings, not a requirement that
the EVRPTW-DB RQ launchers retain the same compute budget.

## Charging-station semantics

The paper explicitly says that recharging stations can be visited any number
of times and does not define a charging-station revisit reward. The shared
benchmark nevertheless freezes a route-local anti-cycle rule: a given physical
station may be used at most once by one vehicle route, and becomes available
again after a depot return. This is disclosed as a benchmark safety adaptation,
not attributed to the paper and not a global station-copy limit.

The paper's additional mask remains active: `depot -> station` and consecutive
`station -> station` actions are blocked because the vehicle is already full.
`DRLTSSoftConstraintEnv` and `DRLTSHardConstraintEnv` combine that rule with the
shared route-local station mask. No invented CS reward is used.

## Benchmark adaptations

| Component | Published DRL-TS | EVRPTW-DB adapter |
|---|---|---|
| Objective | Total route distance | Electricity plus dispatched-vehicle cost in the new formal track |
| Node input | demand, TW bounds, node type | same plus service duration and station power |
| Edge input | distance, time, nearest-neighbor indicator | same plus canonical path energy |
| Energy transition | fixed consumption rate times distance | released directed running-time-path energy matrix |
| Charging | paper service/recharging time and full recharge | arrival-dependent full-charge time from each station's power |
| Hard feasibility | paper capacity/TW/electricity mask | canonical safe-continuation mask plus the paper station mask |
| Fleet | finite homogeneous fleet in the paper formulation | unlimited homogeneous fleet; route count is nonbinding |
| Evaluation | generated instances; greedy or 1,280 samples | frozen Stage-2 splits, registered candidate budget, independent verifier |

Service, station power, and explicit energy are input adapters required to
represent distinctions in EVRPTW-DB. They do not add a new encoder or decoder
stage.

## Reward and normalization

For a complete solution the paper reward is negative total distance. Its
Stage-1 minimization cost is total distance plus raw lateness, capacity, and
electricity violations weighted by `alpha`, `beta`, and `gamma`; in Stage 2 it
reduces to total distance. There is no CS visit term.

The adapter preserves the three violation semantics and the two-stage
strategy, but separates the shared task contract from the DRL-TS-only soft
auxiliary. New formal runs replace the objective-facing distance term with the
common electricity-plus-vehicle cost in
[`COST_OBJECTIVE_CONTRACT_V1.md`](../COST_OBJECTIVE_CONTRACT_V1.md). For the
positive minimization cost used by DRL-TS, the shared task part is

```text
L_task = C / S_N + I[incomplete] * (b_N + lambda_u * unserved_fraction)
```

`S_N`, `b_N`, and `lambda_u` come from the frozen per-scale
[`drl_energy_vehicle_reference_scale_v2`](../configs/drl_reward_contract_energy_vehicle_v2.json)
contract and are shared with the other formal DRL methods. The rule
`b_N = Q_0.99(C_ref / S_N) + 1` is a frozen empirical calibration candidate,
not a mathematical feasibility-first guarantee. Although each frozen value is
larger than the maximum normalized cost in its 500-member calibration sample,
that is only an in-sample fact and is not an upper bound on every feasible
solution.

The method-specific Stage-1 profile is independently frozen in
[`drl_ts_soft_auxiliary_v1.json`](../configs/drl_ts_soft_auxiliary_v1.json).
For resource component `j` in capacity, time-window, and energy, it uses

```text
v_bar_j = min(component_clip,
              sum_{t in A_j} min(x_j,t, step_clip) / N)
L_soft = alpha * v_bar_capacity
       + beta  * v_bar_time
       + gamma * v_bar_energy
```

Here `N` is the fixed instance customer count, not an observed action or
applicable-transition count. The normalized raw excess `x_j,t` is excess demand
over cargo capacity, lateness over operating-horizon duration, or energy deficit
over battery capacity. Capacity applies only on customer arrivals; time and
energy apply on every valid travel transition. The frozen profile uses
`step_clip=1`, `component_clip=1`, and unit weights. Thus each training
component is bounded by 1 and extra zero-violation depot or station moves cannot
dilute an earlier violation.

Unclipped raw sums remain authoritative for feasibility and are logged with
the clipped sums, applicable-transition counts, and final unweighted
components. A structurally complete Stage-1 rollout with a nonzero raw
violation is labelled `completed_with_soft_violation`: it pays `L_soft` but not
the hard terminal failure floor. Every incomplete Stage-1 or Stage-2 rollout
pays the common terminal term exactly once, including a rollout that served all
customers but failed to return to the depot. The shared task-contract identity
and the independent method-auxiliary profile identity are both recorded in
formal provenance; changing either invalidates same-run resume.

Legacy distance configurations remain supported. Edge distance, time, and
energy inputs are divided respectively by the fixed training-pool distance
scale, operating horizon, and battery capacity. These observation/edge scales
are separate from both reward contracts.

The paper does not specify this reward or input normalization, clipping,
fixed-customer aggregation, or terminal guard. They are documented benchmark
adaptations, not paper-exact behavior, and none of the training-only terms
ranks reported routes.

## Fidelity boundary

The published architecture, two-stage semantics, reward terms, station mask,
and experiment settings have been checked against the supplied full chapter.
Differences caused by real directed-road matrices, station-specific charging,
explicit energy, normalized physical units, unlimited fleet semantics, and the
independent verifier are labelled adaptations above. Without author code, this
supports `verified_paper_guided_adaptation`, not bitwise or numerical
reproduction.
