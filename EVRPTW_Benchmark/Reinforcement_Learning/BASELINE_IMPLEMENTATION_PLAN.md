# Learning Baseline Implementation Plan

This document freezes the implementation order and the comparison boundary for
the learning baselines in EVRPTW-B.  It distinguishes original method design,
benchmark adaptation, and shared evaluation semantics.

## Canonical comparison contract

All methods solve the same homogeneous-fleet, unlimited-fleet EVRP-TW task.
The canonical objective is the total directed-road distance of a complete,
feasible solution.  The shared environment and verifier define travel time,
energy, time-window, cargo, full-charging, and route-closure semantics.

Training rewards remain method-specific.  An implementation may retain the
original paper's auxiliary shaping, penalty terms, curriculum, or value
baseline, but its objective-facing term is adapted to distance.  Every such
change is recorded in the method's `ADAPTATION.md`.  Reward values are not
compared across methods; complete solutions are compared only after shared
verification.

## Frozen implementation order

1. **AM-EVRPTW.** Adapt the authors' MIT-licensed Attention Model code.  Preserve
   the attention encoder-decoder and REINFORCE with greedy rollout baseline.
   Add only EVRP-TW input features, dynamic state, and the shared action mask.
2. **EVRPTW-RL.** Paper-guided reimplementation of the native single-stage
   EVRPTW method by Lin, Ghaddar, and Nathwani.  No official implementation has
   been verified at freeze time.
3. **DRL-TS.** Paper-guided reimplementation of the native two-stage method by
   Chen et al.  Preserve its edge-aware graph attention and two-stage training
   strategy.  The publisher labels code as available on request; no public
   author repository was verified at freeze time.
4. **TERRAN.** Migrate the existing repository implementation to the same
   canonical environment and reporting contract without changing its model
   architecture or auxiliary shaping. Canonical Stage-2 training and verified
   evaluation are implemented; legacy configurations remain explicitly marked.
5. **Edge-DIRECT (conditional).** Add only after the four required baselines are
   reproducible.  Its heterogeneous setting must be restricted explicitly to
   the benchmark's homogeneous special case.

LEHD is intentionally excluded: its supervised labels and Random Re-Construct
procedure require a new EVRP-TW algorithm rather than a data/state/mask adapter.

## Source registry

| ID | Paper | Publication | Primary source | Code status |
|---|---|---|---|---|
| `am_evrptw` | Kool, van Hoof, and Welling, *Attention, Learn to Solve Routing Problems!* | ICLR 2019 | <https://openreview.net/forum?id=ByxBFsRqYm> | Authors' repository: <https://github.com/wouterkool/attention-learn-to-route>, MIT, inspected at `c9abf41ac2f878a55b20dc7e829bc942bb999631` |
| `evrptw_rl` | Lin, Ghaddar, and Nathwani, *Deep Reinforcement Learning for the Electric Vehicle Routing Problem with Time Windows* | IEEE T-ITS 2022 | <https://doi.org/10.1109/TITS.2021.3105232> | Paper-guided reimplementation unless an author repository is subsequently verified |
| `drl_ts` | Chen et al., *Deep Reinforcement Learning with Two-Stage Training Strategy for Practical Electric Vehicle Routing Problem with Time Windows* | PPSN 2022 | <https://doi.org/10.1007/978-3-031-14714-2_25> | Publisher: source available on request; no public author repository verified; paper-guided reimplementation |
| `terran` | TERRAN | Existing project method | Project paper and repository history | Existing code; canonical-adapter migration required |
| `edge_direct` | Mozhdehi, Mohammadizadeh, and Wang, *Edge-DIRECT* | Canadian AI 2024 | <https://arxiv.org/abs/2407.01615> | No public author repository verified; conditional for the homogeneous track |

Downloaded PDFs are working materials and are not committed.  The source URLs,
code commit, license, and method-specific adaptation decisions are committed.

## Conditional Edge-DIRECT decision

Edge-DIRECT is not part of the required four-method implementation. Its three
signature components are a time-window reachability graph, edge-enhanced
attention over travel-time/energy features, and a vehicle-selection decoder for
a finite heterogeneous fleet. The first two transfer naturally. The third does
not: under the canonical homogeneous unlimited-fleet task, selecting among
identical vehicles is unidentifiable and collapses to a constant distribution.

Therefore Edge-DIRECT may enter the table only after a separately registered
homogeneous-special-case contract is approved. That contract must retain the
vehicle decoder as a documented degenerate one-class module or omit it and name
the result as an ablation; it must not silently describe a newly designed fleet
controller as the published method. No such result is required for the core
benchmark table.

## Per-method acceptance checklist

Before a method is admitted to a benchmark table, its directory must contain:

- `README.md`: reproducible train/evaluate commands;
- `ADAPTATION.md`: original design, benchmark changes, and known deviations;
- source and license provenance;
- deterministic smoke tests on a canonical instance;
- shared-verifier replay of exported routes;
- a frozen training seed set and decoding budget;
- explicit reporting of complete-solution rate, feasible-solution rate,
  distance, vehicle count, runtime, and memory.
