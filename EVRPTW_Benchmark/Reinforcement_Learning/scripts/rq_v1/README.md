# DRL RQ server launch bundles

These bundles launch the formal training queues directly. The pilot queue and
the local-pilot launch gate were removed on 2026-09-04; no pilot command or
pilot completion artifact is required before `full.sh`.

Defaults are derived from the checked-out repository. Dataset discovery uses
the repository-relative `EVRPTW_Dataset/Instances_v2` tree, with optional
`EVRPTW_RESTORE_ROOT` as a secondary search root. No machine-specific absolute
data path is committed.

## Active single-seed four-scale configuration

Runtime budget: `drl_rq_runtime_budget_v10_min5000_max10000_tailval50`.
This is a fresh candidate budget: start it with `full.sh`; do not use a v8 or v9
checkpoint as a v10 resume source. `resume.sh` is for interruption recovery within
the same v10 job and commit.

The manifests contain seed 1234 only. The 2080 Ti servers own Cus50/Cus100;
the two RTX 6000 Ada GPUs own Cus500/Cus1000.

| Scale | Hardware | Minimum epochs | Hard cap | Environments/epoch | Maximum environments | Maximum customer exposures |
|---|---|---:|---:|---:|---:|---:|
| Cus50 | RTX 2080 Ti | 5,000 | 10,000 | 1,024 | 10,240,000 | 512,000,000 |
| Cus100 | RTX 2080 Ti | 5,000 | 10,000 | 256 | 2,560,000 | 256,000,000 |
| Cus500 | RTX 6000 Ada | 5,000 | 10,000 | 64 | 640,000 | 320,000,000 |
| Cus1000 | RTX 6000 Ada | 5,000 | 10,000 | 2 | 20,000 | 20,000,000 |

Physical batches use exact gradient accumulation divisors of the logical batch:

| Scale | AM | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|
| Cus50 | 1,024 | 128 | 128 | 128 |
| Cus100 | 256 | 64 | 32 | 128 |
| Cus500 | 8 | 16 | 8 | 64 |
| Cus1000 | 2 | 2 | 2 | 2 |

AM and TERRAN use 100 training trajectories on Cus500/Cus1000. EVRPTW-RL and
DRL-TS use one because sample-100 exceeded memory even at physical batch 1.
Validation and test use stochastic best-of-100 decoding on 500 fixed validation
views. Validation runs every 250 epochs through epoch 5,000, then every 50 epochs.
Early stopping is disabled through epoch 5,000; after that, ten consecutive
non-improving validations stop the run, with a hard cap of 10,000 epochs and an
earliest stop at epoch 5,500. `best.ckpt`, `checkpoint_selected.pt`, and
`best_within_5000.ckpt` identify the primary fixed-budget selection.
`best_overall.ckpt` records the best state including optional tail training. The
validation instance set and per-instance candidate seeds are fixed across all
checkpoints; test remains independent and never selects a checkpoint. DRL-TS
always switches from soft to hard training after epoch 2,500, independent of the
10,000-epoch cap.

There are 24 formal jobs total. Server counts are 8, 5, 3, and 8 for
`2080ti_4_1`, `2080ti_4_2`, `2080ti_3_1`, and `a6000_2_1`, respectively.
The Ada queue contains the eight large-scale jobs, split evenly across its two
GPUs.

## Launch and inspect

Activate a Python environment containing the project dependencies, then run the
bundle for the current server:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/full.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/status.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/resume.sh --seed 1234
```

`full.sh` prepares deterministic shared artifacts if necessary and launches the
formal per-GPU queues through `nohup`/`setsid`. `status.sh` is read-only.
`resume.sh` resumes only jobs with complete resume evidence. Launcher provenance
records the actual Python executable, environment, branch, and commit. The
scripts do not perform per-file SHA-256 hashing.

Standalone artifact preparation accepts the same seed:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/prepare_artifacts.sh --seed 1234
```

If the restored dataset is elsewhere, override its parent root:

```bash
export EVRPTW_RESTORE_ROOT="../../../evrptw_runtime"
```
