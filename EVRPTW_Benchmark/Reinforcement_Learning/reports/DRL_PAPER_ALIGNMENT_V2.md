# DRL paper-alignment audit v2

Audit date: 2026-09-03. Benchmark objective:
`min_directed_road_distance_km`.

## Release decision

The adapters are not a set of exact paper reproductions. Formal training remains
blocked by the existing signed launch gate. A model is called paper-aligned only
for components supported by a full primary source or fixed author code. Necessary
changes for EVRPTW-DB are labelled benchmark adaptations; missing evidence is
labelled unresolved.

| Model | Evidence | Architecture | Normalization and reward | Status |
|---|---|---|---|---|
| AM-EVRPTW | Full paper and official code at `c9abf41ac2f878a55b20dc7e829bc942bb999631` | Encoder is byte-identical to the selected upstream file; local decoder is an EVRPTW adaptation | Paper route length becomes raw directed-road km; unfinished penalty is adapter-only | verified adaptation |
| EVRPTW-RL | Full accepted manuscript | Structure2Vec, two attention stages and LSTM checked against equations | Paper distance and station penalty retained; physical km is scaled explicitly; fleet and negative-battery penalties are zero under benchmark semantics | verified paper-guided adaptation |
| DRL-TS | Full PPSN chapter supplied by the user | Node/edge GAT, GRU/context decoder, masks and training equations checked; EVRPTW-DB feature additions documented | Paper distance and three Stage-1 violations retained with benchmark scaling; no CS penalty | verified paper-guided adaptation |
| TERRAN | User-designated CaliRoute reference code | Target model tree and PBRS implementation checked against project source | Shared Stage-2 normalization is documented; distance/PBRS and remaining-customer horizon penalty are named separately | reference-code verified adaptation |
| Edge-DIRECT-H | Full paper; no author code | Current scaled-dot-product/LayerNorm encoder does not match the published additive-GAT/BatchNorm encoder | Travel-time reward was intentionally changed to directed distance | **blocked and excluded from formal methods** |

## Shared observation normalization

The shared environment currently emits:

- coordinates: per-instance, per-axis min-max to `[0,1]`;
- demand and load: divided by cargo capacity;
- time windows, service time and current time: divided by operating-horizon
  duration after subtracting the working start where applicable;
- battery used/remaining: divided by battery capacity;
- station power: divided by the maximum eligible station power in that instance;
- edge matrices for EVRPTW-RL/DRL-TS/Edge-DIRECT-H: each physical matrix divided
  by its own finite per-instance maximum.

These are benchmark input adapters. No reviewed paper prescribes this complete
normalization for real city-road matrices. They must not be described as
paper-exact normalization.

## Reward contracts

### AM-EVRPTW

For a complete rollout, training cost is exactly the sum of selected canonical
`distance_matrix_km` arcs. An incomplete rollout adds the separately named
training-only guard `P_km * (1 + unserved_fraction)`. The guard never ranks
reported solutions.

The official AM rollout baseline uses exponential warmup with beta 0.8 during
its first paper epoch and challenges the greedy baseline at a paper-epoch
boundary. In the fixed-update benchmark path a paper epoch is 2,500 optimizer
updates; the smaller benchmark field `training_epochs` is only a historical
name for optimizer updates.

### EVRPTW-RL

The paper reward is represented as the negative of a minimization cost:

`distance_km / distance_scale_km + 0.3 * station_visits`.

The benchmark hard safe-continuation mask prevents negative battery, so the
paper's negative-battery penalty is zero. Unlimited homogeneous vehicles make
the paper's excess-fleet penalty zero. Truncated rollouts receive a separately
named training-only incomplete guard. The physical-distance scale is the
instance median depot-customer-depot repair distance; this is a benchmark
adapter needed to keep the published 0.3 station term dimensionally meaningful,
not a normalization stated in the paper.

EMA is used for optimizer steps 1--1,000. The greedy baseline is initialized at
step 1,000 and challenged every 100 optimizer steps thereafter, using the
configured one-sided paired test.

### DRL-TS

The full chapter defines reward as negative total distance and Stage-1 cost as
total distance plus lateness, capacity, and electricity violations. The three
published weights are all 1; Stage 2 reduces to total distance. The paper defines
no station-visit or station-revisit reward. Stations can be visited any number of
times, but its tour mask forbids selecting a station immediately after the depot
or another station.

The adapter keeps those semantic terms and station mask. Distance and each
violation are scaled by benchmark physical units, and incomplete rollouts receive
a separately named training-only guard. These scaling and guard terms are
documented adaptations, not paper-exact normalization.

### TERRAN

The model, PPO path and PBRS formula are inherited from the project code. For
the Stage-2 adapter, the base reward remains negative normalized directed
distance. PBRS components remain separately logged. If the registered rollout
horizon is exhausted, the project failure heuristic adds

`-failure_penalty * remaining_customer_fraction`.

This term is training-only and does not affect verifier-based evaluation.

### Edge-DIRECT-H

The paper uses negative travel time and a finite heterogeneous fleet. The
benchmark version changes the objective to directed-road distance and collapses
vehicle choice to one homogeneous class. More importantly, the current encoder
uses scaled dot-product attention and LayerNorm, whereas the paper specifies
additive attention and BatchNorm. It is not paper-faithful and remains outside
the frozen formal model set.

## What requires new evidence

DRL-TS no longer has a manuscript blocker because the user supplied the full
chapter. Author code would still be required for a numerical-reproduction claim,
but not for the documented paper-guided adapter. TERRAN no longer has a
manuscript blocker because the user designated the
CaliRoute implementation as its method authority; this supports a
reference-code adaptation claim, not an independent paper-reproduction claim.
Edge-DIRECT-H needs an encoder rewrite and still cannot reproduce heterogeneous
vehicle selection under the benchmark's homogeneous unlimited fleet. None of
these gaps may be filled by inferred equations or guessed hyperparameters.
