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
| Objective | Total route distance | Sum of selected directed `distance_matrix_km` arcs |
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

The adapter preserves these four semantic terms, but uses explicit benchmark
scaling:

- directed-road distance is divided by one deterministic training-pool scale;
- excess demand is divided by vehicle cargo capacity;
- lateness is divided by the operating-horizon duration;
- energy deficit is divided by battery capacity.

Edge distance, time, and energy inputs are divided respectively by the fixed
training-pool distance scale, operating horizon, and battery capacity. The paper
does not specify this complete normalization, so it is a documented benchmark
adaptation rather than paper-exact behavior.
A separately named incomplete-rollout guard supplies a finite training signal
when the registered step budget is exhausted; it never ranks reported routes.

## Fidelity boundary

The published architecture, two-stage semantics, reward terms, station mask,
and experiment settings have been checked against the supplied full chapter.
Differences caused by real directed-road matrices, station-specific charging,
explicit energy, normalized physical units, unlimited fleet semantics, and the
independent verifier are labelled adaptations above. Without author code, this
supports `verified_paper_guided_adaptation`, not bitwise or numerical
reproduction.
