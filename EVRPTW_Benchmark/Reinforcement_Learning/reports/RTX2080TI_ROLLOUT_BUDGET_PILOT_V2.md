# RTX 2080 Ti scale-aware rollout pilot v2

Date: 2026-09-02
Executable commit: `797f24523e55cf66d98322e1ee1a3c9784896b82`
Hardware: 4 x NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each
Environment: `maojie`, PyTorch 2.5.1+cu121
Dataset: released Stage-2 v7 train/validation artifacts

> Historical runtime evidence: the 100-full-pass extrapolation below is
> permanently retained, but it is no longer the active training protocol.
> The active candidate uses a fixed 500,000 customer-exposure budget per
> method/scale/seed job; see
> `FIXED_EPOCH_TRAINING_BUDGET_V1.md`.

## Scope

This pilot revalidates the frozen scale-aware training rollout budgets:

| Scale | Training rollout steps |
|---|---:|
| Cus50 | 80 |
| Cus100 | 140 |
| Cus500 | 600 |
| Cus1000 | 1200 |

The cap applies only to training. Validation and test retain the full dynamic
environment horizon. Cus1000 remains assigned to RTX A6000 and was not tested
on this host.

Cus100 and Cus500 were run as four-method concurrent waves, matching a four-GPU
server's expected I/O and CPU contention. Cus50 AM-EVRPTW, EVRPTW-RL and DRL-TS
were run concurrently while TERRAN Cus500 was active; TERRAN Cus50 ran alongside
TERRAN Cus500. Outputs are calibration-only artifacts under
`/tmp/drl_rollout_budget_pilot_797f245*`.

## Measured pilot results

All twelve RTX 2080 Ti method-scale pilots completed without OOM.

| Method | Scale | Batches | Seconds/batch | Peak allocated GiB | Mean steps | Cap-exhausted rate |
|---|---|---:|---:|---:|---:|---:|
| AM-EVRPTW | Cus50 | 8 | 18.67 | 2.92 | 70.70 | 59.94% |
| EVRPTW-RL | Cus50 | 8 | 6.42 | 4.04 | 79.38 | 89.06% |
| DRL-TS | Cus50 | 8 | 4.54 | 5.01 | 61.30 | 12.81% |
| TERRAN | Cus50 | 8 | 27.29 | 8.18 | 61.75 | 4.02% |
| AM-EVRPTW | Cus100 | 4 | 12.05 | 2.32 | 119.58 | 67.75% |
| EVRPTW-RL | Cus100 | 4 | 10.47 | 3.71 | 139.74 | 96.88% |
| DRL-TS | Cus100 | 4 | 7.72 | 5.67 | 138.34 | 89.58% |
| TERRAN | Cus100 | 4 | 31.77 | 6.94 | 117.64 | 4.10% |
| AM-EVRPTW | Cus500 | 8 | 14.34 | 2.60 | 546.71 | 85.42% |
| EVRPTW-RL | Cus500 | 8 | 23.97 | 4.53 | 600.00 | 100.00% |
| DRL-TS | Cus500 | 8 | 16.19 | 4.80 | 572.50 | 12.50% |
| TERRAN | Cus500 | 8 | 119.92 | 7.45 | 553.05 | 0.31% |

The memory result is green for the frozen RTX 2080 Ti batches. The scientific
horizon result is not green for every method: AM-EVRPTW and EVRPTW-RL, and
DRL-TS at Cus100, frequently exhaust the cap during untrained-policy rollouts.
The telemetry must be retained during formal training to determine whether the
rate falls as the policy improves. These caps must not be described as complete
rollouts for those method-scale pairs.

## Runtime extrapolation under the candidate 100-pass protocol

The table uses measured training seconds per batch, the frozen physical batch,
all released training views, and 100 full data passes. It excludes periodic
500-view validation, checkpoint I/O, failures and restarts, so it is a training
lower bound rather than a scheduling promise.

| Method | Scale | Batches/pass | Hours/pass | Hours/100 passes |
|---|---|---:|---:|---:|
| AM-EVRPTW | Cus50 | 250 | 1.30 | 129.6 |
| EVRPTW-RL | Cus50 | 834 | 1.49 | 148.6 |
| DRL-TS | Cus50 | 1,250 | 1.58 | 157.6 |
| TERRAN | Cus50 | 250 | 1.90 | 189.5 |
| AM-EVRPTW | Cus100 | 500 | 1.67 | 167.4 |
| EVRPTW-RL | Cus100 | 1,563 | 4.54 | 454.4 |
| DRL-TS | Cus100 | 2,084 | 4.47 | 447.1 |
| TERRAN | Cus100 | 250 | 2.21 | 220.6 |
| AM-EVRPTW | Cus500 | 1,667 | 6.64 | 664.2 |
| EVRPTW-RL | Cus500 | 5,000 | 33.29 | 3,328.7 |
| DRL-TS | Cus500 | 10,000 | 44.98 | 4,498.4 |
| TERRAN | Cus500 | 200 | 6.66 | 666.2 |

Periodic 500-view validation every five passes is expected to add roughly
5--15% for shorter jobs and a smaller percentage for the slow Cus500 jobs.
The full 2080 Ti training allocation contains 36 jobs (four methods, three
scales, three seeds). With the current static 11-slot manifest, the slowest
slot extrapolates to about 5,142 training hours, or 214 days, before periodic
validation. Even perfect rebalancing cannot beat the aggregate-work lower
bound of about 3,020 hours, or 126 days, on eleven RTX 2080 Ti GPUs.

The corresponding approximate makespans from the same measurements are:

| Full data passes | Current 11-slot makespan, training only |
|---:|---:|
| 1 | 2.1 days |
| 3 | 6.4 days |
| 5 | 10.7 days |
| 10 | 21.4 days |
| 100 | 214 days |

These figures exclude the twelve Cus1000 jobs assigned to the two RTX A6000
GPUs. Their runtime remains unverified until the A6000 pilot is run.

## Decision

- Memory gate on RTX 2080 Ti: PASS for the tested batches.
- Runtime gate for 100 full data passes: FAIL / not operationally reasonable.
- Rollout completeness at the proposed caps: diagnostic warning for the
  high-exhaustion method-scale pairs above.
- Full training launch: remains blocked pending a globally frozen data-pass or
  exposure budget and the RTX A6000 Cus1000 pilot.

