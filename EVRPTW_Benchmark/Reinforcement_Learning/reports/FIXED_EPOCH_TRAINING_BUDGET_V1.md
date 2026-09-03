# Fixed-epoch training budget v1

Date: 2026-09-02
Dataset: released Stage-2 v7 train/validation artifacts
Hardware checked here: 4 × NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each
Status: implementation and RTX 2080 Ti smoke PASS; full launch remains gated

> Superseded for formal launch: this report permanently preserves the first
> fixed-exposure implementation and its timing evidence. Formal manifests now
> use reviewer-facing logical epochs shared across methods; see
> `LOGICAL_EPOCH_TRAINING_BUDGET_V2.md`.

## Decision

Formal training no longer traverses every train view and no longer uses a
full-data-pass count. Each `method × scale × seed` job receives the same
target of 500,000 customer exposures:

```text
target_environments = 500000 / customer_count
training_epochs = ceil(target_environments / physical_batch_size)
actual_environments = training_epochs * physical_batch_size
```

Each epoch consumes one physical batch from a deterministic, seed-shuffled
prefix of the frozen train pool. Sampling is without replacement inside the
job. The target is 10% of one historical full-pass customer exposure at every
scale, so the fixed job never traverses the complete train index.

The same literal epoch count would be unfair because the registered physical
batches differ by method. Equality is therefore enforced on customer exposure,
with only the unavoidable final-batch rounding shown below.

## Frozen formal budgets

| Method | Scale | Num env / epoch | Epochs | Actual envs | Customer exposures |
|---|---|---:|---:|---:|---:|
| AM-EVRPTW | Cus50 | 400 | 25 | 10,000 | 500,000 |
| EVRPTW-RL | Cus50 | 120 | 84 | 10,080 | 504,000 |
| DRL-TS | Cus50 | 80 | 125 | 10,000 | 500,000 |
| TERRAN | Cus50 | 400 | 25 | 10,000 | 500,000 |
| AM-EVRPTW | Cus100 | 100 | 50 | 5,000 | 500,000 |
| EVRPTW-RL | Cus100 | 32 | 157 | 5,024 | 502,400 |
| DRL-TS | Cus100 | 24 | 209 | 5,016 | 501,600 |
| TERRAN | Cus100 | 200 | 25 | 5,000 | 500,000 |
| AM-EVRPTW | Cus500 | 6 | 167 | 1,002 | 501,000 |
| EVRPTW-RL | Cus500 | 2 | 500 | 1,000 | 500,000 |
| DRL-TS | Cus500 | 1 | 1,000 | 1,000 | 500,000 |
| TERRAN | Cus500 | 50 | 20 | 1,000 | 500,000 |
| AM-EVRPTW | Cus1000 | 1 | 500 | 500 | 500,000 |
| EVRPTW-RL | Cus1000 | 1 | 500 | 500 | 500,000 |
| DRL-TS | Cus1000 | 1 | 500 | 500 | 500,000 |
| TERRAN | Cus1000 | 1 | 500 | 500 | 500,000 |

Training rollout horizons remain scale-aware: Cus50=80, Cus100=140,
Cus500=600, and Cus1000=1200. Validation and test retain their complete dynamic
horizon.

## RTX 2080 Ti execution evidence

All four methods completed fixed-epoch runs at Cus50, Cus100 and Cus500. Each
result reported `budget_mode=fixed_training_epochs`, the exact requested epoch
count, and exact `num_env × epochs × customer_count` exposure. No run OOMed.
The full owned DRL test suite passed 50/50, and regenerated manifests passed the
checked-in reproducibility check.

The timing model combines a same-configuration one-epoch control with a 4- or
8-epoch pilot:

```text
estimated_seconds(E) = T1 + (E - 1) × (Tn - T1) / (n - 1)
```

For runner-managed pilots, measured process overhead is added separately.

| Method | Scale | Formal epochs | Estimated training minutes / seed |
|---|---|---:|---:|
| AM-EVRPTW | Cus50 | 25 | 7.32 |
| EVRPTW-RL | Cus50 | 84 | 5.86 |
| DRL-TS | Cus50 | 125 | 6.17 |
| TERRAN | Cus50 | 25 | 11.57 |
| AM-EVRPTW | Cus100 | 50 | 6.46 |
| EVRPTW-RL | Cus100 | 157 | 5.57 |
| DRL-TS | Cus100 | 209 | 9.51 |
| TERRAN | Cus100 | 25 | 14.15 |
| AM-EVRPTW | Cus500 | 167 | 8.57 |
| EVRPTW-RL | Cus500 | 500 | 27.87 |
| DRL-TS | Cus500 | 1,000 | 67.54 |
| TERRAN | Cus500 | 20 | 40.90 |

Across the 36 RTX 2080 Ti formal training jobs, the current static 11-slot
assignment is estimated at 10.57 aggregate GPU-hours and 1.57 hours training
makespan. These figures include the 8-view pilot validation embedded in the
measurements, not the formal final 500-view validation. The final validation is
still serial per instance and must be measured or optimized separately before
the number is presented as end-to-end wall time.

Cus1000 is assigned to the two RTX A6000 GPUs and is not extrapolated from the
2080 Ti evidence.

## Remaining gates

- The 500-view final validation remains unchanged and still selects the
  checkpoint without using test data.
- RTX A6000 Cus1000 batch/memory/runtime evidence is still required.
- High rollout-cap exhaustion seen in the v2 rollout pilot remains a diagnostic
  warning for affected method-scale pairs and must stay in formal telemetry.
- `pilot.full_runtime_budget_approved=false` remains frozen; full training
  cannot start until the new runtime evidence is reviewed and explicitly
  approved.

The historical 100-full-pass FAIL is retained in
`RTX2080TI_ROLLOUT_BUDGET_PILOT_V2.md`; it has not been deleted or re-labelled.
