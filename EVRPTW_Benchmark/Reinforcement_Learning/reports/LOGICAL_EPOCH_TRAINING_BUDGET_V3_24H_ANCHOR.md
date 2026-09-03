# 24-hour Cus1000 anchored logical-epoch budget v3

Date: 2026-09-03
Runtime budget ID: `drl_rq_runtime_budget_v2_cus1000_24h_anchor`
Status: candidate implemented; formal launch remains blocked by G1--G8

## Derivation

The planning anchor is one Cus1000 training job on one RTX A6000:

```text
1000 logical epochs × 1 environment/epoch × 1000 customers
= 1000 environments
= 1,000,000 customer exposures
≈ 24 hours (planning target supplied before pilot/test evaluation)
```

The 24-hour value is not treated as measured throughput or an enforced
timeout. The A6000 pilot must measure all four models before formal launch.
The scientific primary budget remains data-matched customer exposure; wall
clock is a reported secondary resource axis.

## Scale schedule

All methods at the same scale and seed consume the same deterministic stream.
The anchor is propagated to the other scales by holding customer exposures at
one million, while retaining the preregistered 100/200/500/1000 epoch counts.

| Scale | Logical epochs | Environments/epoch | Total environments | Customer exposures | Train-pool draw fraction |
|---|---:|---:|---:|---:|---:|
| Cus50 | 100 | 200 | 20,000 | 1,000,000 | 20% |
| Cus100 | 200 | 50 | 10,000 | 1,000,000 | 20% |
| Cus500 | 500 | 4 | 2,000 | 1,000,000 | 20% |
| Cus1000 | 1,000 | 1 | 1,000 | 1,000,000 | 20% |

One logical epoch is exactly one outer optimizer update. Therefore the
effective batch is equal to environments/epoch. Physical microbatches remain
architecture dependent and exact gradient accumulation restores the effective
batch.

## Effective and physical batches

| Method | Cus50 | Cus100 | Cus500 | Cus1000 |
|---|---:|---:|---:|---:|
| AM-EVRPTW | 200/200 | 50/50 | 4/4 | 1/1 |
| EVRPTW-RL | 200/100 | 50/25 | 4/2 | 1/1 |
| DRL-TS | 200/50 | 50/10 | 4/1 | 1/1 |
| TERRAN | 200/200 | 50/50 | 4/4 | 1/1 |

Each cell is `effective batch / physical microbatch`. All physical values are
at or below the previously measured safe caps. Because some effective batches
changed, the complete concurrent pilot is still required before launch.

## Planning-time consequences

The older v2 2080Ti measurements used half as many customer exposures. A
first-order linear extrapolation gives the following training-only estimates;
they exclude 500-view validation, checkpoint I/O, failures and restarts.

| Method | Cus50 | Cus100 | Cus500 | Cus1000 |
|---|---:|---:|---:|---:|
| AM-EVRPTW | 11.3 min | 11.0 min | 48.8 min | 24 h planning target |
| EVRPTW-RL | 8.4 min | 9.5 min | 52.3 min | 24 h planning target |
| DRL-TS | 13.7 min | 28.8 min | 3.86 h | 24 h planning target |
| TERRAN | 31.0 min | 1.08 h | 4.71 h | 24 h planning target |

With twelve Cus1000 jobs and two A6000 GPUs, 24 hours per job implies a
theoretical six-wave A6000 makespan of about 144 hours (six days), before
validation and failures. It does not imply that all Cus1000 jobs finish in one
day.

## Launch discipline

- Do not edit the formal gate to start jobs.
- Run all 20 pilots and collect measured wall time, peak memory, rollout-cap
  telemetry, finite loss, checkpoint, and independent verifier evidence.
- If a Cus1000 model exceeds the 24-hour estimate, stop for one global budget
  review before viewing T1/T2/T3; do not silently reduce only that method.
- Cus2000 remains evaluation-only.
- No SHA-256 or per-file content scan is part of this protocol.
