# TR17 / TR18 EVRPTW-RL stable retraining

`evrptw_rl_stable.sh` starts only **Road Cus100 TR18 on physical GPU 2** and
**Euclidean Cus100 TR17 on physical GPU 3**. Both are fresh runs with
`--graph-aggregation mean`. This is a documented numerical-stability adaptation;
legacy checkpoints and the original `full.sh` retain `sum` semantics.
See [STABILITY_VALIDATION_REPORT.md](STABILITY_VALIDATION_REPORT.md) for the
recorded CPU checks, limitations and reproducible diagnostic results.

The old runs must first be preserved and stopped explicitly on 2080ti_4_2. This
entry rejects occupied GPU 2/3 and never stops another process. Completed DRL-TS
TR03/TR04 and work on GPU 0/1 are outside this launch.

Use an isolated checkout so other running experiments keep their source files:

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git worktree add /data/Maojie/ICLR/cus100-evrptw-stability-4-2 origin/cus100-evrptw-stability-4-2-20260914
cd /data/Maojie/ICLR/cus100-evrptw-stability-4-2
conda activate maojie
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/stop_old_evrptw_rl.py --stop
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/evrptw_rl_stable.sh
```

The optional stop command above first verifies the old TR17/TR18 process identities
and backs up already-saved checkpoints and metadata under the original
`repair_backups/` directory, then sends SIGTERM only to those verified processes.
Unsaved in-progress work is not checkpointed by stopping. The original output
files remain available. Omit `--stop` to inspect without stopping anything.

The launcher reuses existing data and frozen artifacts from the sibling
`EVRPTW-DB` checkout when they are absent in this worktree. It checks the original
SHA256 values and records the resolved absolute paths. Explicit overrides are
also supported:

```bash
export CUS100_ROAD_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823
export CUS100_SYNTHETIC_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911
export CUS100_ARTIFACT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus100_20260911
```

It first checks that both assigned GPUs are idle, locks their physical UUIDs,
checks frozen data/reward/stream files, and captures the actual source. Each
source then receives a disposable six-update CUDA smoke run: two EMA updates,
four greedy-baseline updates, baseline probes at 4/6, and two fixed validations
at 3/6 (10 instances each). The smoke additionally checks multi-state action
distinction and useful encoder gradients. Both sources must pass before either
formal run starts. Smoke checkpoints are never reused for formal training.

The initial physical batch is the old **200**, which previously used about 10 GiB
but **has not yet been measured for this stable architecture on 2080ti_4_2**.
OOM or a measured peak above 10.3 GiB triggers smaller physical microbatches,
increments of 10 near 200. Effective batch stays **200**, preserving the original
2,000,000-entry training stream and customer exposure budget. The measured
allocation and whether it falls in 9.5–10.3 GiB are recorded. A lower peak is
reported honestly; the launcher does not allocate unused tensors to fill VRAM.
Non-OOM errors, unchanged smoke validation outputs, or ineffective policy
diagnostics stop the deployment for inspection.

Formal settings remain seed **1234**, n-traj **30**, rollout **240/360**, validation
**500 every 100 epochs**, minimum **5000**, maximum **10000**, patience **5** after
5000, AdamW **1e-3**, and native **1000-update EMA → greedy** baseline. The shared
verified economic objective and source-specific frozen reward scales are
unchanged; the existing station auxiliary remains `0.3 * legal_station_visits / N`.

New default output root:
`EVRPTW_Benchmark/results/cus100_evrptw_stable_20260914`, with runs
`TR18_stable_mean` and `TR17_stable_mean`. Use `--output-root /absolute/new/path`
for another fresh attempt. There is no resume option. Existing original outputs
are never changed.

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/evrptw_rl_stable.sh --mode status
```

The usual status-table script also works after setting `CUS100_OUTPUT_ROOT` to
the new root. Live logs are in `launchers/2080ti_4_2/launcher.log`, then each new
run's `stdout.log`; calibration logs are retained under `calibration/`.
