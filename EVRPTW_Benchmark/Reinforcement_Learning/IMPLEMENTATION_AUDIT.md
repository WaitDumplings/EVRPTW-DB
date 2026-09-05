# DRL Baseline Paper and Canonical-Environment Audit

Audit date: 2026-09-03. Target evaluation objective:
`min_directed_road_distance_km`.

## Evidence boundary

| Method | Paper evidence | Author code evidence | Audit verdict |
|---|---|---|---|
| AM-EVRPTW | Full ICLR 2019 paper | Official repository fixed at `c9abf41ac2f878a55b20dc7e829bc942bb999631` | Architecture/training adaptation verified |
| EVRPTW-RL | Full IEEE T-ITS accepted manuscript | No verified public repository | Paper-guided implementation verified against published equations; documented deviations remain |
| DRL-TS | Full PPSN 2022 chapter supplied by the user | No verified public repository | Paper-guided implementation verified against equations; documented EVRPTW-DB adaptations remain |
| TERRAN | User-designated CaliRoute implementation | Existing project code | Reference-code-verified Stage-2 adaptation; no independent manuscript-reproduction claim |
| Edge-DIRECT-H | Full Canadian AI 2024 paper | No verified public repository | **Blocked**: homogeneous special case plus known encoder mismatch |

The audit does not treat a publisher challenge page as a downloaded paper and
does not bypass access controls. DRL-TS was upgraded only after the user
supplied the full chapter; its lack of author code still precludes a numerical
reproduction claim.

## Method checks

### AM-EVRPTW

- The vendored graph encoder is byte-for-byte identical to the selected file at
  the fixed official commit.
- Three-layer/eight-head attention encoding, cached decoder projections,
  glimpse attention, tanh clipping, REINFORCE, greedy rollout baseline, and the
  paired baseline-update test are retained.
- EVRPTW static features, dynamic cargo/battery/time context, charging nodes,
  and canonical feasibility are documented adaptations.

### EVRPTW-RL

- Structure2Vec implements the local, global, neighbor-sum, and directed edge
  travel-time terms from the paper.
- Context attention, choice attention, and recurrent LSTM decoder follow the
  published sequence.
- The hard canonical mask replaces infeasibility penalties that become zero by
  construction. The unlimited homogeneous fleet is represented as a
  nonbinding fleet ratio. Beam search is not implemented and is not claimed.

### DRL-TS

- The supplied full chapter verifies the four published node features, three
  published edge features, simultaneous node/edge GAT updates, GRU/context
  decoder, two-stage masks, REINFORCE objective, and greedy rollout baseline.
- The paper has no station-visit reward and permits repeated station use. The
  benchmark explicitly adds a route-local anti-cycle rule (same station at most
  once per route, reset at depot) while retaining the paper's depot/station mask;
  soft training, hard training, validation, and evaluation share that rule.
- Service duration, station power, explicit canonical energy, normalized physical
  units, unlimited-fleet semantics, and independent verification are documented
  EVRPTW-DB adaptations. This is `verified_paper_guided_adaptation`, not an
  author-code or numerical reproduction.

### TERRAN

- The designated method reference is the CaliRoute `TERRAN` directory supplied
  by the user. The target `models/` tree and `pbrs.py` match that reference.
- Its Stage-2 adapter uses the shared canonical matrices, station-specific full
  charging, directed-distance reward, POMO candidates, and exact route verifier.
- Stage-2 pool/protocol/restart/reporting code and the rollout-horizon truncation
  wrapper are documented benchmark adaptations. The reference's dormant
  expert-provider and route-boundary collection hooks are not enabled by its
  trainer/configs and are not imported into the formal training path.
- This evidence supports `reference_code_verified_adaptation`; it does not claim
  an independent manuscript-level reproduction.

### Edge-DIRECT-H

- Implements the directed time-window reachability graph, two encoder stages,
  one-class vehicle context, node decoder, and greedy-baseline REINFORCE.
- The current scaled-dot-product attention and LayerNorm encoder blocks do not
  match the paper's additive GAT and BatchNorm equations. Architecture-level
  fidelity is therefore blocked, not verified.
- The original finite heterogeneous vehicle choice collapses to one class under
  EVRPTW-DB's unlimited homogeneous fleet. The one-class vehicle decoder is
  retained, with zero categorical log-probability, and the method is named
  `Edge-DIRECT-H` to avoid a false full-reproduction claim.
- The paper's cumulative-travel-time objective is deliberately replaced by the
  benchmark's requested directed-road-distance objective.

## Canonical environment and objective

All five adapters use the same Stage-2 contract:

- objective arcs: `distance_matrix_km`;
- schedule arcs: canonical directed running-time shortest paths;
- battery arcs: energy evaluated on those running-time paths;
- charging: immediate full charge using each station's power and the frozen
  derating factor; same-station reuse is route-local and resets at depot;
- feasibility: customer uniqueness, cargo, time windows, operating horizon,
  battery, depot returns, and charging-assisted return paths;
- final acceptance: independent replay by `route_validator.validate_routes`.

The audit found and fixed one shared defect: after the final customer, the
action mask previously exposed only a direct depot return. A customer admitted
because it could return through charging stations could therefore be truncated.
Reference and JIT masks now keep feasible station actions available, and the
regression test replays `depot -> customer -> charger -> depot` through the
independent verifier.

Model-facing normalization uses one deterministic training-pool distance
scale, operating horizon for time, battery capacity for energy, and full-charge
time divided by horizon for each station. Validation/test reuse the training
scale. Incomplete-rollout penalties and any published auxiliary shaping are
training-only. Successful candidates are selected by minimum independently
verified directed-road distance.
