# DRL RQ server launch bundles

The pilot queue remains removed: there is no extra two-epoch pilot phase in a
formal run. The current reward-contract revision is protected by an explicit
fail-closed authorization gate. The 16 RTX 2080 Ti Cus50/Cus100 jobs and the two
previously approved A6000 TERRAN jobs are authorized; every other A6000 job
remains blocked until separately approved.

Defaults are derived from the checked-out repository. Dataset discovery uses
the repository-relative `EVRPTW_Dataset/Instances_v2` tree, with optional
`EVRPTW_RESTORE_ROOT` as a secondary search root. No machine-specific absolute
data path is committed.

## Active single-seed calibrated-scale configuration

Runtime budget: `drl_rq_runtime_budget_v13_am5_min5000_max10000_tailval50`.
This is a fresh candidate budget.  After explicit formal authorization, start
it with `full.sh`; do not use a v8, v9, v10, v11, or v12 checkpoint as a v13
resume source. `resume.sh` is only for interruption recovery within the same
v13 job, commit and reward/auxiliary contracts.

### All-method electricity + vehicle-cost restart

All four methods now use `rivian_energy_vehicle_cost_v1`:
`cost_USD = 0.1341 * (100 / 257) * distance_km + 33.56 * vehicles_started`.
Fixed vehicle cost is charged once on a valid departure from the depot.
Validation candidates and checkpoints are selected feasible-first, then by
minimum total cost; raw distance and both cost components remain separate.
See [`COST_OBJECTIVE_CONTRACT_V1.md`](../../COST_OBJECTIVE_CONTRACT_V1.md)
for parameters, accounting, normalization and historical-comparison boundaries.
The shared task-reward contract is
`drl_energy_vehicle_reference_scale_v2`; TERRAN additionally uses `gamma=1.0`
in both returns and PBRS. The other methods retain their native auxiliary
shaping and training-stage schedules. Architectures, data and budgets are unchanged.
All four trainers use AdamW with explicit decoupled `weight_decay=0.01`;
their existing method-native learning rates remain unchanged. The optimizer
name and weight decay are recorded in each manifest, launcher provenance and
checkpoint configuration/arguments.

All four formal trainers also write an observational `reward_diagnostics.jsonl`
inside their run output directory. It records normalization provenance,
training-cost/reward components, advantage distributions and pre-clipping
gradient statistics without changing training. See
[`TRAINING_REWARD_DIAGNOSTICS.md`](../../TRAINING_REWARD_DIAGNOSTICS.md)
for units, method-specific interpretation, sampling and resume boundaries.

This contract revision has independent frozen training-reference calibrations
for Cus50, Cus100, Cus500 and Cus1000. Cus50 uses the compatibility training
view index; the other scales use the core training view index. Each scale uses
its own deterministic 500-view, 10-city, weekday/weekend-stratified reference
cohort; denominators are never copied across scales.

On each RTX 2080 Ti server, start its complete authorized queue from scratch
without loading an older checkpoint:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<2080-server>/full.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<2080-server>/status.sh --seed 1234
```

The four canonical bundles contain 8, 5, 3 and 8 jobs for `2080ti_4_1`,
`2080ti_4_2`, `2080ti_3_1` and `a6000_2_1`, respectively. The three 2080 Ti
bundles are fully authorized. The general A6000 bundle still contains all eight
large-scale jobs for planning, but only its dedicated two-job TERRAN queue is
currently authorized.
For Cus1000 only on the two-GPU large-scale server, use the following instead of
its full queue; do not launch both queues together:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/cus1000_full.sh --seed 1234
```

Each run writes under
`$EVRPTW_OUTPUT_ROOT/<representation>/<condition>/<method>/<scale>/seed_1234/<git-commit>/`.
The new commit provides a new output root; old results remain untouched. Shared
artifacts are still under the unchanged v13 budget, so this update does not
require dataset or stream regeneration. Only use `resume.sh`
to recover an interrupted run of this same revision and commit. A mismatched
objective or gamma/reward contract, or a fresh launch over existing training
history, is rejected. The launcher passes the same versioned objective JSON to
every method and records its resolved values in each job's provenance.

The local CPU regression suite excludes vendored `reference_materials`.
`tests/test_rq_server_environment.py` additionally requires Linux utilities
(`flock`, GNU `realpath -m`) on non-Linux hosts. Local CPU validation does not
establish GPU convergence or a full Linux server launch. Verification commands
and platform boundaries are recorded with each implementation change.

The enabled manifests contain seed 1234 only. RTX 2080 Ti servers own the
Cus50/Cus100 jobs; the two RTX 6000 Ada GPUs own Cus500/Cus1000.

| Scale | Runtime status | Hardware | Minimum epochs | Hard cap | Environments/epoch | Maximum environments | Maximum customer exposures |
|---|---|---|---:|---:|---:|---:|---:|
| Cus50 | calibrated / authorized | RTX 2080 Ti | 5,000 | 10,000 | 1,024 | 10,240,000 | 512,000,000 |
| Cus100 | calibrated / authorized | RTX 2080 Ti | 5,000 | 10,000 | 256 | 2,560,000 | 256,000,000 |
| Cus500 | enabled | RTX 6000 Ada | 5,000 | 10,000 | 64 | 640,000 | 320,000,000 |
| Cus1000 | enabled | RTX 6000 Ada | 5,000 | 10,000 | 2 | 20,000 | 20,000,000 |

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

AM uses 5 training trajectories on every scale; TERRAN uses 100. EVRPTW-RL and
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
`num_minibatches=1` and `ppo_step_chunk_size=720`. Batch 2, 100 training
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

Once the explicit gate is open, `full.sh` prepares deterministic shared
artifacts if necessary and launches the formal per-GPU queues through
`nohup`/`setsid`. While the gate is closed it fails before spawning a trainer.
`status.sh` is read-only.
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
online validation, and compact TERRAN observations. The v13 logical budgets,
seeds and best-of-100 evaluation are unchanged. Physical batches and rollout
limits use the current post-optimization calibration. The TERRAN Cus1000 PPO
override documented above is the only update-schedule change.

TERRAN's encoder Dropout modules are retained for checkpoint-key compatibility
but use `p=0`, so PPO rollout and update log-probabilities are evaluated under
the same deterministic policy. Training rollouts encode immutable instance
features once and reuse that state across decoder steps. On Cus1000 this reduced
the measured training-epoch wall time from 22.70 s to 18.32 s on average while
leaving the 42,319 MiB PPO peak unchanged. Formal runs must start fresh from the
post-change commit rather than resume a pre-change checkpoint.
See [performance implementation and verification](../../PERFORMANCE_OPTIMIZATION.md)
for equivalence tests, timing boundaries, an optional idle-GPU diagnostic and
cross-commit resume precautions. No formal training was launched by this patch.

The complete 2080 Ti evidence is in
[`RTX2080TI_PER_JOB_MEMORY_CALIBRATION_V4.md`](../../reports/RTX2080TI_PER_JOB_MEMORY_CALIBRATION_V4.md).
Ada revalidation instructions are in
[`RTX6000_ADA_MEMORY_CALIBRATION_HANDOFF_V2.md`](../../reports/RTX6000_ADA_MEMORY_CALIBRATION_HANDOFF_V2.md).
