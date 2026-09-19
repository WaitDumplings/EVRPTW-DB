# EVRPTW-RL Road Cus500 after-AM validation

Date: 2026-09-13 (America/Los_Angeles).
Worktree: `/data/Maojie/ICLR/cus500-evrptw-rl-after-am`.
Branch: `cus500-evrptw-rl-after-am-20260913`, based on `23c9082`.

## Completed checks

245 CPU tests passed with the local `maojie` Python, without allocating a CUDA device:

| Area | Passed | Evidence |
|---|---:|---|
| New EVRPTW-RL distributed protocol | 6 | Two-rank Gloo, global gradient/EMA, warmup copy, greedy baseline, probes, exact resume before/after transition, signed configuration rejection |
| Existing distributed adapters/helpers | 42 | AM, RRNCO-EV and DRL-TS distributed regression plus edge-row gather |
| Native EVRPTW-RL and shared objective/diagnostics | 86 | Native model, cost objective, REINFORCE diagnostics and paper alignment contract |
| Deployment | 48 | Fixed GPU 0/1, complete command, data/stream contract, locks and resume checks |
| Watcher | 20 | AM success/failure, normal process exit, PID reuse, GPU wait, source/request changes, restart without duplicate submission, actual first update, inherited daemon lock and heartbeat |
| Automatic calibration | 43 | Largest safe measured batch, two-rank finite memory, OOM versus other failure, unchanged rollout/model/objective, finite checkpoints and changed weights, baseline transition, GPU recheck before each child, termination of own probe only |

The calibration suite also ran the real EVRPTW-RL model on a tiny CPU fixture with two Gloo ranks, the formal station auxiliary and independent verifier. Its six-update outputs passed the production checkpoint audit: EMA on updates 1–2, actor-to-baseline copy after update 2, greedy baseline on 3–6, probes at 4/6, and two validations of ten instances. This validates integration, not Cus500 performance or GPU memory.

Read-only validation against the actual live AM launch request passed: same host, physical GPU UUIDs, seed, Road release, objective/reward files, 1700/2550 rollout and 30 trajectories. CPU data preparation confirmed 10,000 train / 500 validation views, 551 nodes per instance, and disjoint train/validation view and family identities. The generated deterministic stream could be read back and reused with identical contracts. Shell syntax and `git diff --check` passed.

## GPU measurement deferred until AM completes

AM is still using GPU 0/1 and TERRAN uses GPU 2/3. No active training was stopped or modified for these checks. **The formal EVRPTW-RL batch has not yet been measured on CUDA.** Template batch 1 is a placeholder; the watcher first waits for successful AM completion and release of GPU 0/1.

The queued calibration then searches from batch 4, brackets/doubles/bisects using full 1700/2550 rollout and 30 trajectories, and runs a six-update EMA-to-greedy confirmation. Each GPU's NVIDIA-SMI training-process peak must be positive and no greater than 10.3 GiB. It attempts to reach 9.5–10.3 GiB by increasing useful batch size; if integer batch sizes leave a gap, the report explicitly records the largest measured safe batch and whether both ranks reached the target band. Non-OOM failures block formal launch.

Calibration outputs are disposable. Formal training starts fresh from seed 1234 with the measured batch, the original 1000-update EMA warmup, native learning rate 1e-3, min/max 5000/10000 epochs, and 500-instance validation every 100 epochs. It never resumes a calibration checkpoint.

Runtime evidence will be saved under `EVRPTW_Benchmark/results/cus500_evrptw_rl_20260913/`:

- `watcher/request.json` and `watcher/status.json`: frozen dependency/source identities, active heartbeat, and handoff status.
- `calibration/report.json`: actual per-rank peaks, selected batch, finite/changed model checks, and baseline transition confirmation.
- `calibrated_config.json`: final formal configuration, generated only after successful calibration.
- `launchers/local_after_am_gpu01/evrptw_rl/`: formal command, source/environment/data identities and launcher status.
- `runs/evrptw_rl_road_cus500_seed1234/`: training/validation metrics, best/latest checkpoints and logs.

The watcher exits with `handed_off` only after the formal run has completed at least one update. This status does not claim that the full experiment has finished. Source changes while queued are rejected; keep this dedicated worktree frozen.
