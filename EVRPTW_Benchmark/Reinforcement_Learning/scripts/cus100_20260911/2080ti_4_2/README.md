# TR17 / TR18 EVRPTW-RL stable retraining

## Direct retraining on an idle server

On a server with the existing Cus100 data and two free GPUs, update the code and
start both fresh runs directly:

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git switch cus100-evrptw-stability-4-2-20260914
git pull --ff-only
conda activate maojie
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/retrain.sh
```

`retrain.sh` defaults to **physical GPU 0: Road Cus100 TR18**, **physical GPU 1:
Euclidean Cus100 TR17**. To select another free pair, specify the GPUs in
**Road, Euclidean** order:

```bash
EVRPTW_GPUS=2,3 bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/retrain.sh
```

This direct entry skips GPU calibration and starts from seed 1234 with physical
and effective batch **200**. It retains the previous batch setting to target
roughly 10 GiB; **GPU memory has not been measured for this stable architecture
on the destination server**. It does not automatically reduce batch. It checks
that the selected GPUs are free and rejects occupied cards. No old tasks need
to be stopped on the idle server, and no worktree or backup step is required.

The model uses `--graph-aggregation mean`, a documented numerical-stability
adaptation. Legacy checkpoints and the original `full.sh` retain `sum` semantics.
These fresh runs do not load old checkpoints or change original output files.

## Data and training settings

The launcher reuses the existing frozen data, reward files and training streams,
checks their original SHA256 values, and records resolved absolute paths. If
needed, set the existing deployment locations explicitly:

```bash
export CUS100_ROAD_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823
export CUS100_SYNTHETIC_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911
export CUS100_ARTIFACT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus100_20260911
```

Formal settings remain seed **1234**, n-traj **30**, rollout **240/360**, validation
**500 every 100 epochs**, minimum **5000**, maximum **10000**, patience **5** after
5000, AdamW **1e-3**, and native **1000-update EMA → greedy** baseline. Effective
batch 200 preserves the original 2,000,000-entry training stream and exposure
budget. The shared verified economic objective and source-specific frozen
reward scales are unchanged. The station auxiliary remains
`0.3 * legal_station_visits / N`.

Default output root: `EVRPTW_Benchmark/results/cus100_evrptw_stable_20260914`, with
runs `TR18_stable_mean` and `TR17_stable_mean`. Use `--output-root /absolute/new/path`
for another fresh attempt. Existing nonempty output is rejected; this entry has
no resume option.

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/retrain.sh --mode status
```

Status includes current epoch, exact validation costs, feasibility and time
since the last training-log write. Launcher logs are in
`launchers/2080ti_4_2/launcher.log`; each run has `stdout.log` and `stderr.log`.
`CUS100_STABLE_OUTPUT_ROOT` changes the default output location. The old status
script can read these runs after setting its `CUS100_OUTPUT_ROOT` to the new root.

## Optional guarded entry

`evrptw_rl_stable.sh` retains the original guarded workflow, defaulting to
physical GPU **2 for Road / 3 for Euclidean**. It first performs disposable CUDA
calibration and learning checks on both sources, then starts both fresh formal
runs only if the checks pass. It verifies six training updates, the EMA-to-greedy
transition, finite and effective training gradients, two fixed validation
outputs, and multi-state action distinction with encoder gradients. Smoke
checkpoints are not reused for formal training.

This optional workflow starts from physical batch 200 and can reduce the
physical microbatch if measured memory exceeds 10.3 GiB or CUDA runs out of
memory. Effective batch stays 200. Measured peaks and whether they reach
9.5–10.3 GiB are recorded; other failures stop the guarded launch for inspection.
This calibration is **not run by `retrain.sh`**.

For a deliberate restart on the old occupied server, the separate
`stop_old_evrptw_rl.py` helper can inspect old TR17/TR18 identities without taking
action. Its explicit `--stop` option verifies those identities, backs up
already-saved checkpoints and metadata, then sends SIGTERM only to the verified
processes. Unsaved in-progress work is not checkpointed by stopping. This helper
is optional and is not part of the idle-server direct-retraining steps above.

The recorded CPU evidence and its limits are in [STABILITY_VALIDATION_REPORT.md](STABILITY_VALIDATION_REPORT.md).
