# EVRPTW-RL TR17/TR18 stability validation — 2026-09-14

Fix branch: `cus100-evrptw-stability-4-2-20260914`, based on `24382c9`.
CPU checks ran in an isolated worktree. No existing GPU experiment was stopped or modified.

## Change and scope

The opt-in `--graph-aggregation mean` divides both neighboring-node messages and off-diagonal edge-time aggregation by `max(N - 1, 1)`. It addresses attention saturation observed with the repeated complete-graph sums. No learned layers, objective coefficients, station restrictions, auxiliary coefficients or data splits were changed. This is an explicit adaptation of the published sum equation; label new experiments `EVRPTW-RL (mean-aggregation adaptation)`. Legacy/default `sum` behavior remains intact.

## Reproducible CPU checks

- 108 core CPU tests cover default compatibility, mean gradients, activation checkpointing, checkpoint signatures and actual two-rank Gloo training.
- 54 final deployment/diagnostic/stop-helper tests passed (38 + 6 + 10). No test sent signals to real training processes.
- For seeds 1/1234/2301 at 121/551 nodes, default sum outputs were bitwise equal to base commit `24382c9`. Legal-logit spans were at most 2.39e-7 and the tested encoder gradients were zero. Mean spans were 0.084–0.435, with nonzero local/neighbor gradients.
- Old single/distributed training signatures with activation checkpoint stride 0/1 were unchanged. Changed aggregation is signed and rejected during incompatible resume.
- A real production CLI run on CPU, batch 1, n-traj 8, H240/H360, completed six updates, the shortened EMA→greedy schedule, baseline probes at 4/6 and two fixed 10-instance validations. The deployment `audit_probe()` passed against its actual logs/checkpoints. This is integration evidence, not the formal batch-200/n-traj-30 GPU run.

## Fixed training-pool diagnostic cohorts

Each pair starts from fresh weights with seed 1234, shares ordered update instances and reward contracts, and uses three fixed training instances disjoint from its updates. Batch is 1; train/evaluation horizons are 240/360. The six-step checks use n-traj 30. A longer Road check uses n-traj 8 for 50 updates, so its numbers must not be compared directly with the n-traj-30 rows. No test set is used. The dedicated probe does not use validation data.

Cost below is the mean USD objective **only over complete-and-feasible instances**. Policy effectiveness counts multi-state checks with distinguishable legal logits and a finite, nonzero encoder gradient, rather than merely changed parameter bytes.

| Source | Aggregation | Updates / n-traj | Cost before → after | Feasible before → after | Effective states after |
|---|---|---|---:|---:|---:|
| Road | sum | 6 / 30 | 3422.76 → 3422.76 | 3/3 → 3/3 | 0/15 |
| Road | mean | 6 / 30 | 3388.25 → 1801.68 | 3/3 → 3/3 | 12/12 |
| Euclidean | sum | 6 / 30 | 11129.68 → 11129.68 | 2/3 → 2/3 | 0/18 |
| Euclidean | mean | 6 / 30 | 9799.02 → 13362.46 | 2/3 → 3/3 | 16/16 |
| Road | sum | 50 / 8 | 4408.01 → 4408.01 | 3/3 → 3/3 | 0/15 |
| Road | mean | 50 / 8 | 3829.49 → 1200.51 | 3/3 → 3/3 | 12/12 |

Euclidean mean after six updates changes the feasible subset from two instances to three. Its overall feasible-only mean increases because the newly feasible instance costs 23970.59. On the two instances feasible both before and after, mean cost changes from 9799.02 to 8058.40. This small diagnostic establishes useful policy changes; it does not establish full-validation superiority or convergence. Do not compare means without their feasibility counts.

Full numeric summaries, cohort IDs, data/profile/source hashes and per-instance outcomes are retained in [stability_cpu_validation.json](stability_cpu_validation.json). Local detailed diagnostic traces remain under `EVRPTW_Benchmark/results/evrptw_stability_20260914/`. Profile checkpoints supply the frozen configuration only; the learning probes initialize new weights.

## Remote 2080ti_4_2 deployment

Use [README.md](README.md) and `evrptw_rl_stable.sh`. It runs new `TR18_stable_mean` on physical GPU 2 (Road100) and `TR17_stable_mean` on physical GPU 3 (Euclidean100), retaining effective batch 200, seed 1234, n-traj 30, H240/H360, the original two-million-entry streams, frozen reward contracts and 500-instance validation schedule. It starts fresh parameters and optimizer state.

GPU memory has **not** been measured for this revision. With both cards free, the launcher tests physical batch 200 and reduces it if OOM or measured process peak exceeds 10.3 GiB; effective batch remains 200. It records whether the measured peak reaches 9.5–10.3 GiB. Each source must finish the six-update actual GPU training/baseline/validation checks and the multi-state policy diagnostic before either formal job begins. Failures stop deployment for inspection.

Original TR17/TR18 outputs are retained. The explicit stop helper verifies process identity and backs up already-saved checkpoints before TERM; unsaved current progress is not checkpointed by stopping. The optional stop helper has not been run against the user’s jobs during this repair.

## Direct retraining option

The subsequent `retrain.sh` entry starts formal training immediately on two free GPUs (default Road GPU0 / Euclidean GPU1), as requested. It skips the optional GPU probes above, retains physical/effective batch 200, and records calibration as skipped. CPU validation above remains applicable; no new GPU-memory measurement is claimed. The updated deployment suite passes 45 tests, including the actual worker path with mocked process spawning and frozen GPU/skip settings.
