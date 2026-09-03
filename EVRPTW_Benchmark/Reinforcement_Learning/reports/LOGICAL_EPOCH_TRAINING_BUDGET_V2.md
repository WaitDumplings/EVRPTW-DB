# Reviewer-facing logical epoch training budget v2

Date: 2026-09-02
Dataset: released Stage-2 v7 train/validation artifacts
Hardware checked here: 4 × NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each
Status: implementation and RTX 2080 Ti smoke PASS; full launch remains gated

## Motivation

The v1 fixed-exposure protocol used one physical GPU batch as one epoch. It
matched customer exposure across methods, but produced method-dependent epoch
counts such as 25 versus 125 at Cus50. Those counts were computationally valid
but reviewer-facing epoch comparisons were unnecessarily ambiguous.

V2 defines an epoch independently of a method's maximum safe physical batch.
Within one scale, all four methods now receive the same:

- logical epoch count;
- environments per logical epoch;
- total environment count;
- customer exposure.

This is an executable change, not a relabelling of old optimizer steps.

## Frozen scale budgets

| Scale | Logical epochs | Environments / epoch | Total environments | Customer exposures | Train-pool fraction |
|---|---:|---:|---:|---:|---:|
| Cus50 | 100 | 100 | 10,000 | 500,000 | 10% |
| Cus100 | 200 | 25 | 5,000 | 500,000 | 10% |
| Cus500 | 500 | 2 | 1,000 | 500,000 | 10% |
| Cus1000 | 1,000 | 1 | 1,000 | 1,000,000 | 20% |

Sampling uses one deterministic seed-shuffled prefix of the frozen train pool
without replacement. No job traverses the complete training index.

Cus1000 intentionally has twice the customer exposure of the other scales
because one environment is the indivisible minimum and 1,000 logical epochs
were selected explicitly. This difference must be disclosed rather than
described as equal exposure across all scales.

## Logical and physical batches

The logical batch determines one epoch and one outer optimizer update for the
three REINFORCE methods. A logical batch larger than a model's physical limit is
split into exact-divisor micro-batches with gradient accumulation.

| Method | Scale | Physical micro-batch | Logical batch |
|---|---|---:|---:|
| AM-EVRPTW | Cus50 | 100 | 100 |
| EVRPTW-RL | Cus50 | 100 | 100 |
| DRL-TS | Cus50 | 50 | 100 |
| TERRAN | Cus50 | 100 | 100 |
| AM-EVRPTW | Cus100 | 25 | 25 |
| EVRPTW-RL | Cus100 | 25 | 25 |
| DRL-TS | Cus100 | 5 | 25 |
| TERRAN | Cus100 | 25 | 25 |
| AM-EVRPTW | Cus500 | 2 | 2 |
| EVRPTW-RL | Cus500 | 2 | 2 |
| DRL-TS | Cus500 | 1 | 2 |
| TERRAN | Cus500 | 2 | 2 |
| All methods | Cus1000 | 1 | 1 |

The committed method-specific batch calibration remains a safety upper bound.
Formal physical batches are equal to or below those tested limits. TERRAN keeps
its native PPO inner-update semantics; the common logical epoch controls the
number of frozen Stage-2 environments presented to the method.

## Evidence and audit trail

Cus50, Cus100 and Cus500 were each executed as a four-method concurrent wave at
two logical epochs. All 12 jobs:

- completed without OOM;
- reported `budget_mode=fixed_logical_epochs`;
- reported the exact requested epoch count;
- matched `logical epochs × environments per epoch × customer count`;
- produced the expected optimizer-step count for each method's native update
  rule.

The three REINFORCE trainers now persist `logical_epoch_history.jsonl`. Each
row records logical epoch, environment/customer exposure, physical
micro-batches, optimizer step, loss, distance, feasibility, rollout steps,
cap-exhaustion rate, transitions, memory and wall time. TERRAN already records
the corresponding per-epoch evidence in `logs/train_log.csv`.

A post-change DRL-TS smoke produced exactly two history rows. Each row contained
100 environments assembled from two 50-environment physical micro-batches, and
the optimizer-step counter advanced exactly once per logical epoch.

Checked-in manifest reproducibility and the relevant logical-epoch tests pass.

## RTX 2080 Ti timing estimate

The estimate combines same-configuration 2-epoch and 10-epoch concurrent
measurements:

```text
estimated_seconds(E) = T2 + (E - 2) × (T10 - T2) / 8
```

| Method | Cus50 / 100 epochs | Cus100 / 200 epochs | Cus500 / 500 epochs |
|---|---:|---:|---:|
| AM-EVRPTW | 5.65 min | 5.48 min | 24.38 min |
| EVRPTW-RL | 4.21 min | 4.75 min | 26.16 min |
| DRL-TS | 6.87 min | 14.39 min | 115.87 min |
| TERRAN | 15.48 min | 32.53 min | 141.22 min |

For the current static 11-slot assignment, the 36 RTX 2080 Ti formal training
jobs total about 19.85 aggregate GPU-hours with an estimated 2.95-hour training
makespan. This is deliberately conservative single-seed linear extrapolation,
not a scheduling guarantee.

The measurements include only a 2-view final validation. The formal 500-view
validation is unchanged, remains serial per instance, and is not included in
the 2.95-hour figure. Cus1000 remains assigned to RTX A6000 and needs separate
memory/runtime evidence.

## Reviewer-facing sufficiency discipline

Epoch count alone is not evidence of convergence. Formal reporting must include:

- logical epoch count and environments per epoch;
- customer exposure and environment transitions;
- native optimizer-step count;
- per-epoch training curves;
- final held-out validation result;
- rollout-cap exhaustion curves;
- three frozen seeds.

If curves remain materially improving at the frozen endpoint, the global
budget must be revised before any test metric is viewed. It must not be extended
per method after observing T1/T2/T3.

## Remaining gates

- `pilot.full_runtime_budget_approved=false` remains frozen.
- RTX A6000 Cus1000 pilot evidence is still required.
- Formal 500-view validation time must be measured or optimized.
- Full training cannot start until the new timing and convergence protocol are
  explicitly reviewed and approved.

The v1 fixed-exposure report and the historical 100-full-pass FAIL remain in
the repository as provenance; neither has been deleted or relabelled as PASS.
