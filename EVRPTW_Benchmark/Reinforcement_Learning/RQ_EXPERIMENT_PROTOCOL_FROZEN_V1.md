# EVRPTW-DB research-question experiment protocol (frozen v1)

> [!IMPORTANT]
> **Historical contract notice (2026-09-04):** the blocked-launch status,
> three-seed counts, and gate table below describe the original frozen design.
> The active single-seed v13 execution metadata is maintained in
> `configs/drl_rq_protocol_frozen_v1.yaml` and `scripts/rq_v1/README.md`:
> direct formal launch is user-authorized, with 24 training jobs and a
> 5,000-epoch minimum / 10,000-epoch hard cap.

Status: **method frozen; formal training launch blocked**
Scope: four registered learning methods, classical solver evidence, and the
released Stage-2 dataset
Formal training jobs covered after gate release: **72**

This document is the reviewer-facing experiment contract. It supersedes the
`fixed_logical_epochs` candidate as the intended formal protocol, but it does
not relabel or alter any historical pilot evidence. In particular,
`configs/drl_experiment_protocol_v1.yaml` remains a non-release implementation
and timing prototype until the gates at the end of this document are closed.

## Frozen research questions

### RQ1 — Scale

RQ1 asks how feasibility, solution quality, online runtime, memory, and the
availability of reliable references change as the customer scale grows.

1. Same-scale training and evaluation:
   `TrainN -> TestN` for Cus50, Cus100, Cus500, and Cus1000. Cus100, Cus500, and
   Cus1000 use T1/T2/T3; Cus50 is the small-scale compatibility experiment and
   is primarily reported in the appendix.
2. Cross-scale deployment:
   `Train100 -> Test2000`, `Train500 -> Test2000`, and
   `Train1000 -> Test2000`. Cus2000 is an operational-scale proxy and is never
   a training scale or a claim of complete real-world scale coverage.
3. Reference availability:
   the experiment is titled **Reference Availability and Stability of
   Best-Observed Solutions**. It makes no uncertified ground-truth or canonical
   optimality claim.

Raw distance is not compared across scales as if the tasks had identical city
geometry. Primary outcomes are verified feasibility, time to first feasible
solution, online solve/inference time, peak memory, and distance relative to a
versioned post-hoc best-observed solution.

## Best-observed-solution contract

For instance `i`, the post-hoc reference is

```text
z_BKS[i] = min verified z[i, method, seed, decoding, time_budget]
```

Only complete solutions that pass the common canonical verifier and come from
pre-registered methods, seeds, decoding settings, and budgets are eligible.
The registered contributors are Gurobi, ALNS, VNS-TS, AM-EVRPTW, EVRPTW-RL,
DRL-TS, and TERRAN.

The final BKS manifest is generated only after all registered evaluations have
finished. It records instance ID, objective, method, seed, decoding/budget, and
solution ID. It is not available to training, hyperparameter selection, or
checkpoint selection.

Frozen terminology:

```text
Canonical lower bound: unavailable
Canonical optimality gap: unavailable
Ground-truth terminology: prohibited at uncertified scales
Primary quality reference: verified post-hoc BKS
Gurobi incumbent after canonical replay: feasible upper bound
Gurobi ObjBound/MIPGap: copy-limited MILP diagnostic only
```

If a DRL method improves the Gurobi incumbent, the permitted conclusion is that
it found a better verified feasible solution under the registered budget. The
copy-limited Gurobi diagnostic is not a canonical EVRPTW certificate.

RQ1C reports verified feasible rate, Gurobi time to first incumbent,
5 min/30 min/1 h/2 h incumbents, no-incumbent rate, gap to post-hoc BKS, BKS
provenance and improvement with budget, and feasible-objective disagreement.

## RQ2 — Training-support coverage

RQ2 asks whether access to a broader portion of the released, observable
training support improves learning and generalization under a fixed data
budget. It does not claim coverage of the complete real-world population.

Core conditions at Cus100 are:

```text
Random-10%-support
Coverage-10%-support
Full-support
```

`Random-1%-support` is optional. Percentages denote the allowed parent-family
support, not an epoch coverage rate or the fraction actually observed during
training. Full-support does not require a full traversal.

AM-EVRPTW and TERRAN are the two registered RQ2 methods. Full-support reuses
their RQ1 main-training checkpoints. The two additional support conditions add
12 training runs over three seeds.

Coverage support is selected as follows:

1. compute descriptors only from the Full-support training pool;
2. use only pre-solver features, never solution quality;
3. standardize with Full-support training statistics;
4. select parent families by deterministic farthest-first/k-center;
5. resolve ties with a frozen seed;
6. never access validation or test descriptors during selection.

Coverage is reported by mean and P95 nearest-support distance in the frozen
descriptor space. Audits also report allowed and realized parent-family counts,
city by day-type strata, family revisit P50/P90/max, per-family sample counts,
and mean/P95 coverage radius. Nested scale views are never independent support
units.

## RQ3 — Euclidean-to-directed-road transfer

RQ3 compares only `E -> G` and `G -> G`. `G + Inject -> G` is deleted. This is
a representation/deployment-gap experiment, not a new graph-model claim.

Terminal coordinates and identities, depot/customer/charging-station
membership, demand, service time, time windows, battery, capacity, charging
power, and split membership are held fixed.

For nodes `i,j`:

```text
d_E[i,j] = d_E[j,i] = straight-line distance from the fixed coordinates
e_E[i,j] = rho * d_E[i,j]
t_E[i,j] = d_E[i,j] / v_E[day_type]
```

`rho` is the same vehicle energy rate used by the Graph track. For day type
`q` (weekday or weekend), the Euclidean speed is frozen from a deterministic,
pre-registered set of training-only OD pairs:

```text
v_E[q] = sum(d_E[i,j]) / sum(t_G[i,j])
```

The OD pair IDs, pair count, frozen seed, final speeds, and train-only/leakage
audit are written to a manifest. A single speed per day type is shared across
training cities so the T3 unseen city contributes no calibration information.
Validation/test data and their results can never recalibrate Euclidean speed.

Euclidean pre-generation feasibility is reported. Any paired instance that is
infeasible under E follows a pre-registered handling rule; it is not regenerated
or tuned after observing test outcomes.

G-to-G checkpoints reuse RQ1 training. The E-to-G arm adds 12 Cus100 training
runs: four methods by three seeds.

## EV-specific integrity audit

The audit is solution-conditioned and introduces no new optimization model.
For every registered verified route, consecutive charging-station visits are
treated as a block. Each block is removed, its adjacent non-CS nodes are joined
with the directed matrix arc, customer order is fixed, battery constraints are
disabled, and travel time, waiting, service, and time-window feasibility are
replayed. The route is not re-optimized.

The audit reports:

- solutions and instances containing charging-station visits;
- charging-station visits per route;
- minimum arrival state of charge and normalized battery slack;
- charging time divided by route duration;
- distance reduction after removing charging blocks; and
- route-conditioned energy-necessary charging-visit rate.

The final metric means only: given the fixed customer order and route-prefix
state, bypassing this charging visit would violate the battery constraint.

The formal name is **Route-level infinite-battery counterfactual**. It must not
be called an infinite-battery optimum, an optimal battery ablation, or a
globally necessary charging visit.

## Frozen training fairness

The primary comparison is data-matched. For scale `N`, the budget is customer
exposure:

```text
B_N = N * sampled_base_instance_count
```

Within a scale and seed, all four methods consume the same pre-generated,
deterministic base-instance ID stream. Sampling is stratified by the frozen
training strata, with replacement, and contains no validation/test IDs.
Physical microbatches may differ by architecture. A common logical batch is
used within a scale, with gradient accumulation when the safe physical
microbatch is smaller.

Each run reports base-instance and customer exposures, optimizer steps,
physical microbatch, logical batch, native trajectories/multi-starts, training
GPU-hours, throughput, environment transitions, and peak GPU memory. Equal
epoch counts and equal memory utilization are not fairness requirements.

The secondary comparison is compute-matched. The same training run saves both
fixed-exposure checkpoints and 6 h/12 h/24 h/terminal wall-clock checkpoints.
For runs finishing before a wall-clock checkpoint, the terminal model is not
retroactively represented as having consumed the larger compute budget.

Checkpoint selection uses validation only, and test is run once after protocol
freeze. Evaluation first compares verified feasibility, then distance on a
common feasible cohort. Incomplete scores are supplementary. Bootstrap samples
parent families as clusters; paired method comparisons reuse the same sampled
family clusters, and training seeds are not treated as additional independent
test instances.

## Training matrix

```text
RQ1 main:       4 methods * 4 scales * 3 seeds = 48
RQ2 coverage:   2 methods * 2 extra supports * 3 seeds = 12
RQ3 E -> G:     4 methods * 1 scale * 3 seeds = 12
Core total:                                          72
```

Cus2000 evaluation, T1/T2/T3 evaluation, and cross-scale transfer do not add
training jobs. The energy-relaxed directed VRPTW model is not part of this
protocol.

## Formal-launch gates

All eight gates are hard requirements. As of this freeze, formal launch remains
blocked.

| Gate | Requirement | Current status |
|---|---|---|
| G1 | customer-exposure budget for every scale | PILOT REQUIRED |
| G2 | shared deterministic stratified with-replacement ID stream | IMPLEMENTATION REQUIRED |
| G3 | exposure and GPU-hour checkpoints | IMPLEMENTATION REQUIRED |
| G4 | common logical batch for every scale | PILOT REQUIRED |
| G5 | per-method physical microbatch and OOM policy | PILOT REQUIRED |
| G6 | RTX A6000 Cus1000 memory/runtime pilot | PILOT REQUIRED |
| G7 | Euclidean distance/time/energy generation manifest | IMPLEMENTATION REQUIRED |
| G8 | parent-family clustered paired-bootstrap aggregation | IMPLEMENTATION REQUIRED |

No formal 72-run manifest may be marked enabled, and no formal training may be
started, until a versioned gate report marks G1--G8 PASS. Historical timing and
fixed-logical-epoch pilots remain evidence only and must never be relabelled as
formal RQ results.
