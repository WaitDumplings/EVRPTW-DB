# AM Cus100 → Cus500 two-GPU curriculum smoke

Date: 2026-09-20. Both training ranks used RTX 2080 Ti GPUs 0 and 1.
The existing EVRPTW-RL processes on GPUs 2/3 were left running.

Source: stage-1 Road Cus100 best epoch 1300, SHA256
`6ae497332b98eefb507c63d21ff5e713a4bfeb71594259872ad2cbbf77ba2ea1`.
All runs use D_time monetary cost, 30 trajectories per instance,
train/validation action caps 1700/2550, float32, and synchronized NCCL training.

| Phase | Batch per GPU | Global batch | Updates | GPU 0 peak MiB | GPU 1 peak MiB | Mean update seconds |
|---|---:|---:|---:|---:|---:|---:|
| ema | 4 | 8 | 3 | 3691 | 3524 | 2.519 |
| greedy | 4 | 8 | 2 | 3657 | 3492 | 4.117 |
| ema | 12 | 24 | 5 | 10071 | 9924 | 5.475 |
| greedy | 12 | 24 | 3 | 9985 | 9920 | 7.234 |

**Default: batch 12 per GPU / global 24.** All four smoke runs completed with
exit code 0 and finite final model tensors. Actor tensors changed and AdamW step
counters matched the exact additional epoch count. Saved provenance records
source Cus100 → target Cus500, source epoch/hash, and reset state. Two-rank
worker/topology records and independently verified validation outputs were saved.

Smoke validation is only 4 instances per check (all 4/4 feasible), run every
update. It is an execution test, not a formal 500-instance benchmark. Formal
validation remains every 100 updates on 500 instances with 30 candidates each.

EMA profiles use the actual baseline defaults. Greedy profiles explicitly set
`--baseline-warmup-epochs 0 --steps-per-epoch 2 --baseline-eval-size 2` only in
the diagnostic command to exercise the later baseline path and its paired test;
formal training retains its 2500-update EMA warmup and 64-instance baseline probe.

Memory is device-level nvidia-smi sampling at one-second intervals, including
display and CUDA/NCCL overhead; very brief peaks can be missed. These short runs
do not guarantee the peak of every future random batch. No AMP or dummy GPU
allocation was introduced. Detailed raw paths and exact timing are in smoke_runs.csv.

The combined launcher/cross-scale regression suite passed 75 tests. Shell syntax
and git diff whitespace checks passed. Raw smoke outputs remain local in:

- `/data/curriculum_cus500_am_smoke_20260920`
- `/data/curriculum_cus500_am_b12_smoke_20260920`

The source checkpoint binary and dataset are external assets, not committed files.
