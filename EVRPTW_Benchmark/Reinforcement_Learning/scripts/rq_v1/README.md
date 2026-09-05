# DRL RQ server launch bundles

These bundles launch the formal training queues directly. The pilot queue and
the local-pilot launch gate were removed on 2026-09-04; no pilot command or
pilot completion artifact is required before `full.sh`.

Defaults are derived from the checked-out repository. Dataset discovery uses
the repository-relative `EVRPTW_Dataset/Instances_v2` tree, with optional
`EVRPTW_RESTORE_ROOT` as a secondary search root. No machine-specific absolute
data path is committed.

## Active single-seed four-scale configuration

Runtime budget: `drl_rq_runtime_budget_v12_min5000_max10000_tailval50`.
This is a fresh candidate budget: start it with `full.sh`; do not use a v8, v9,
v10, or v11 checkpoint as a v12 resume source. `resume.sh` is for interruption
recovery within the same v12 job and commit.

The manifests contain seed 1234 only. The 2080 Ti servers own Cus50/Cus100;
the two RTX 6000 Ada GPUs own Cus500/Cus1000.

| Scale | Hardware | Minimum epochs | Hard cap | Environments/epoch | Maximum environments | Maximum customer exposures |
|---|---|---:|---:|---:|---:|---:|
| Cus50 | RTX 2080 Ti | 5,000 | 10,000 | 1,024 | 10,240,000 | 512,000,000 |
| Cus100 | RTX 2080 Ti | 5,000 | 10,000 | 256 | 2,560,000 | 256,000,000 |
| Cus500 | RTX 6000 Ada | 5,000 | 10,000 | 64 | 640,000 | 320,000,000 |
| Cus1000 | RTX 6000 Ada | 5,000 | 10,000 | 2 | 20,000 | 20,000,000 |

Physical batches use exact sample-weighted gradient accumulation. REINFORCE
jobs may use a smaller final remainder microbatch; TERRAN keeps exact divisors:

| Scale | AM | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|
| Cus50 | 1,024 | 224 | 132 | 256 |
| Cus100 | 256 | 68 | 34 | 128 |
| Cus500 | 8 | 16 | 8 | 64 |
| Cus1000 | 2 | 2 | 2 | 2 |

The 2026-09-04 Cus1000 boundary sweep is recorded in
[`RTX6000_ADA_CUS1000_MEMORY_CALIBRATION_V2.md`](../../reports/RTX6000_ADA_CUS1000_MEMORY_CALIBRATION_V2.md).
Larger method-specific batches could not simultaneously satisfy the 40--45 GiB
target, the common-exposure contract, the even-batch constraint, and the formal
deadline; batch 2 is therefore intentional rather than an uncalibrated default.

AM and TERRAN use 100 training trajectories on Cus500/Cus1000. EVRPTW-RL and
DRL-TS use one because sample-100 exceeded memory even at physical batch 1.
Validation and test use stochastic best-of-100 decoding on 500 fixed validation
views. Validation runs every 250 epochs through epoch 5,000, then every 50 epochs.
Early stopping is disabled through epoch 5,000; after that, ten consecutive
non-improving validations stop the run, with a hard cap of 10,000 epochs and an
earliest stop at epoch 5,500. `best.ckpt`, `checkpoint_selected.pt`, and
`validation_summary.json` are the formal aliases for `best_overall.ckpt` and
`validation_summary_overall.json`: they identify the best state across the
complete run, including optional tail training. `best_within_5000.ckpt` and
`validation_summary_within_5000.json` preserve the fixed-minimum selection as
separate evidence. The validation instance set and per-instance candidate seeds
are fixed across all checkpoints; test remains independent and never selects a
checkpoint. DRL-TS always switches from soft to hard training after epoch 2,500,
independent of the 10,000-epoch cap.

TERRAN Cus1000 has a manifest-level PPO hyperparameter override of
`num_minibatches=1` and `ppo_step_chunk_size=736`. Batch 2, 100 training
trajectories, three PPO epochs, and the registered exposure budget are
unchanged. With two base environments, the minibatch override reduces Adam
updates from six to three per logical epoch; the larger step chunk reduces
loss-evaluation/backward slicing within each minibatch. The override is not
applied to TERRAN Cus500 or to any other method or scale. Manifests call the
10,000-epoch outer budget `planned_logical_epochs`; TERRAN's native Adam-step
count is recorded separately at runtime as `optimizer_steps_total`.

There are 24 formal jobs total. Server counts are 8, 5, 3, and 8 for
`2080ti_4_1`, `2080ti_4_2`, `2080ti_3_1`, and `a6000_2_1`, respectively.
The Ada queue contains the eight large-scale jobs, split evenly across its two
GPUs. Current rollout limits are Cus50=65, Cus100=120, Cus500=580, and
Cus1000=1200.

### Cus1000 priority profile on A6000

The generated `a6000_2_1/cus1000_jobs.jsonl` is a scheduling-only projection of
the same four canonical Cus1000 jobs. It changes no scientific field. GPU 1 runs
TERRAN; GPU 0 runs DRL-TS, EVRPTW-RL, then AM-EVRPTW sequentially. Launch only
this profile—not the eight-job `full.sh` queue—when prioritizing Cus1000:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/cus1000_full.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/cus1000_status.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/cus1000_resume.sh --seed 1234
```

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

## Runtime optimizations (2026-09-04)

New launches use rollout-local static caches, final-only route export during
online validation, and compact TERRAN observations. The v12 logical budgets,
seeds and best-of-100 evaluation are unchanged. Physical batches and rollout
limits use the current post-optimization calibration. The TERRAN Cus1000 PPO
override documented above is the only update-schedule change.
See [performance implementation and verification](../../PERFORMANCE_OPTIMIZATION.md)
for equivalence tests, timing boundaries, an optional idle-GPU diagnostic and
cross-commit resume precautions. No formal training was launched by this patch.

The complete 2080 Ti evidence is in
[`RTX2080TI_PER_JOB_MEMORY_CALIBRATION_V4.md`](../../reports/RTX2080TI_PER_JOB_MEMORY_CALIBRATION_V4.md).
Ada revalidation instructions are in
[`RTX6000_ADA_MEMORY_CALIBRATION_HANDOFF_V2.md`](../../reports/RTX6000_ADA_MEMORY_CALIBRATION_HANDOFF_V2.md).
