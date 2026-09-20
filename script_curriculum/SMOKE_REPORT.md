# Cus100 stage-1 smoke and batch measurements

Date: 2026-09-20. Host: 4 × RTX 2080 Ti, PyTorch 2.5.1+cu121, float32.

All ten archived source policies loaded successfully and completed GPU training,
backpropagation, checkpoint saving and validation with the D_time monetary objective.
Each smoke validation uses **2 instances × 30 candidates**, not the full 500-instance
formal validation. Full runs retain 2000 additional epochs and validate every 100.

| Model | G batch | G peak MiB | G tested epochs | E batch | E peak MiB | E tested epochs |
|---|---:|---:|---:|---:|---:|---:|
| AM-EVRPTW | 208 | 9877 | 3 | 108 | 10319 | 1 |
| EVRPTW-RL | 240 | 9698 | 3 | 200 | 10276 | 3 |
| DRL-TS | 44 | 9878 | 3 | 44 | 10126 | 3 |
| TERRAN | 384 | 10102 | 1 | 384 | 10158 | 1 |
| RRNCO | 82 | 9703 | 3 | 50 | 10331 | 1 |

**Default update (2026-09-20):** RRNCO G now uses batch **72** by user request.
The batch-82 row above records the original measurement, not the current default;
no new GPU memory measurement is claimed for batch 72. RRNCO E remains batch 50.

These are physical instance batches on one GPU, each with 30 trajectories.
Peaks are device-level `nvidia-smi` samples at one-second intervals, including
display/CUDA overhead; very brief peaks may be missed. G and E differ in route
lengths, so they use separate batch defaults. Short successful runs are not a
guarantee that every later random batch fits; the launcher exposes `--batch-size`
and reports OOM as failure rather than silently changing the experiment.

**Rejected profile:** EVRPTW-RL E batch 224 hit CUDA OOM during backward.
Its final default is 200, which passed the three-epoch recheck. The rejected run
is retained in `smoke_runs.csv`; it is not a usable experiment result.

All completed smoke validation records were checked for the D_time objective,
exact requested stage epoch count and validation cadence. Independent CPU tests
cover exact actor loading, reset state, preserving hard-stage DRL, baseline/critic
handling, CLI/data paths and supervisor failure handling. The final combined
targeted suite passed **63 tests**; wider existing objective/reward/CLI suites
also passed during implementation. Shell syntax and `git diff --check` passed.

Local raw evidence directories:

- `/data/curriculum_stage1_smoke_20260920`
- `/data/curriculum_stage1_batch_profile_20260920`
- `/data/curriculum_stage1_evr_e_retest_20260920`

Only source code, configuration, checkpoint identities and this compact report
are committed. Training data, checkpoint binaries and smoke outputs remain local.
No full 2000-epoch experiments were launched by this validation.
