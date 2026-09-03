# AM-EVRPTW Adaptation Record

## Upstream method retained

The baseline is adapted from Kool, van Hoof, and Welling, *Attention, Learn to
Solve Routing Problems!* (ICLR 2019).  The authors' implementation is available
at <https://github.com/wouterkool/attention-learn-to-route> under the MIT
license.  The implementation review used upstream commit
`c9abf41ac2f878a55b20dc7e829bc942bb999631`.

The following method components are retained:

- attention-based graph encoder with multi-head self-attention, residual
  connections, feed-forward layers, and batch normalization;
- autoregressive attention decoder with cached node projections;
- multi-head glimpse followed by tanh-clipped single-head compatibility logits;
- stochastic policy-gradient training with a deterministic greedy rollout
  baseline;
- exponential baseline warmup with beta 0.8 during the first upstream epoch;
- paired-test rollout-baseline replacement at the end of an upstream epoch.

The selected upstream graph-encoder source and its license are vendored under
`third_party/attention_learn_to_route/`.  The EVRP-TW decoder adapter is local
code because upstream AM does not define this problem.

## Benchmark adaptations

| Component | Upstream AM | AM-EVRPTW |
|---|---|---|
| Task | Euclidean TSP/CVRP and related problems | Canonical directed-road EVRP-TW |
| Final cost | Euclidean route length | Sum of `distance_matrix_km` arcs |
| Static node input | coordinates plus problem-specific scalar | coordinates, normalized volume demand, TW, service, charging power, and node type |
| Dynamic decoder context | previous node plus remaining capacity for CVRP | previous node plus used cargo, used battery fraction, and current time |
| Feasibility | upstream problem state and mask | shared EVRPTW action mask |
| Charging | absent | immediate station-dependent full charging |
| Travel physics | Euclidean norm | exported directed distance/time/energy matrices |
| Fleet | CVRP depot returns | each depot return closes one route; unlimited homogeneous fleet |

The objective-facing training cost remains raw directed-road distance in km.
Shared inputs use the documented benchmark normalization (per-instance
coordinate min-max, capacity/horizon fractions, and relative station power);
that complete normalization is an adapter choice because upstream AM has no
EVRPTW state. No auxiliary step reward is added to successful AM rollouts. An
explicitly configured incomplete
rollout penalty provides a finite training signal when a rollout is truncated;
it is not used to rank solutions and is reported as an adaptation rather than
an upstream AM component.

## Fidelity statement

`AM-EVRPTW` is an adaptation of the published AM architecture and training
procedure, not an official author implementation of EVRP-TW.  We do not claim
that its numerical results reproduce any table in the ICLR 2019 paper, because
the paper did not study EVRP-TW or the released city-road instances.
