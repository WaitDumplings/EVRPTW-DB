# RTX 6000 Ada Cus1000 memory calibration v2

Date: 2026-09-04

Hardware: 2 x NVIDIA RTX 6000 Ada Generation, 49,140 MiB per GPU, driver
545.23.08

Executable revision: `f4d106876490e42d77d802f41289fb66dca7024a`

## Scope

This calibration tested the four `G / Full-support / Cus1000 / seed 1234`
training jobs after the repository update. Every passing candidate completed two
real optimizer updates, checkpoint handling, and an eight-view validation using
`sampling` with 100 candidates. Training trajectory counts matched the intended
formal runs:

- AM-EVRPTW: 100;
- TERRAN: 100;
- EVRPTW-RL: 1;
- DRL-TS: 1.

AM-EVRPTW and EVRPTW-RL used their post-warmup baseline paths so the memory
measurement represented mature training. DRL-TS crossed both its soft and hard
stages. Peaks below are whole-process peaks sampled through NVML, not only
PyTorch allocator peaks.

This was an exploratory boundary test. The eight-view cohort proves that the
training/validation path executes, but it is not a substitute for a 500-view
formal safety gate.

## Measured boundary

| Method | Batch | Peak (GiB) | Result | Wall time (s) | Interpretation |
|---|---:|---:|---|---:|---|
| AM-EVRPTW | 4 | 42.186 | PASS | 53.690 | Strict memory target reached |
| EVRPTW-RL | 4 | 35.061 | PASS | 74.120 | Below target |
| EVRPTW-RL | 5 | 44.811 | PASS | 72.213 | Target reached, but batch is odd |
| EVRPTW-RL | 6 | 47.482 | OOM | 12.856 | No safe even target batch |
| DRL-TS | 2 | 38.684 | PASS | 99.209 | Closest safe integer batch |
| DRL-TS | 3 | 47.469 | OOM | 11.826 | No integer target batch |
| TERRAN | 88 | 42.438 | PASS | 1181.204 | Target reached, runtime is prohibitive |

The raw job results are under
`/tmp/evrptw_cus1000_memory_calibration/G/Memory-calibration` on the calibration
host. Failed upper candidates were retained there.

TERRAN batch 88 took 569.536 seconds for epoch 1. Epoch 2, including the
eight-view validation, took 595.890 seconds. At this rate, 5,000 and 10,000
epochs require approximately 33 and 66 days of training respectively, before
the full formal validation schedule is added.

## TERRAN `n_traj=50` follow-up

A follow-up on 2026-09-05 tested whether halving TERRAN's training trajectory
count would permit a larger, faster batch. Validation/test remained at
sample-100.

| Batch x train trajectories | Process peak (GiB) | Result | Timing |
|---|---:|---|---|
| `4 x 50` | 1.738 | PASS | 62.2 s mean pure-training time per epoch |
| `152 x 50` | 44.842 | PASS | 761.3 s pure training; 781.1 s including 8-view validation |
| `168 x 50` | 47.346 | OOM | Failed in PPO loss evaluation after 602.5 s |

`4 x 50` and the formal `2 x 100` both generate 200 trajectories per epoch,
but they are not equivalent. The former uses four independent graph instances
instead of two and triggers four PPO minibatches rather than two; with three PPO
epochs this means 12 Adam updates instead of 6. Its measured epoch was about
twice as slow as the approximately 30.3-second `2 x 100` short-run mean.

At the memory target, `152 x 50` generates 7,600 trajectories per epoch. This
is fewer than the 8,800 trajectories of `88 x 100`, yet its pure-training epoch
was 34% slower than the 569.5-second `88 x 100` epoch because it encodes and
steps many more independent graphs. The corresponding 5,000/10,000-epoch
training-only estimates are approximately 44/88 days. It would also consume
760M/1.52B customer exposures, 76 times the formal batch-two exposure at the
same epoch checkpoint.

Therefore `n_traj=50` does allow a larger numerical batch, but exchanging
trajectories for base-instance batch is not a speed optimization in the current
TERRAN implementation. The passing batch 152 result is only a one-epoch,
eight-view boundary result and sits close to the upper memory limit; it is not
a 500-view formal safety qualification.

## Why the target batches are not formal batches

The registered Cus1000 protocol has a common logical/effective batch of 2. A
physical microbatch must divide that common effective batch, so 2 is already
the largest admissible physical batch. Raising only a physical cap cannot make
it 4, 5, or 88.

Making the exploratory batches method-specific would change the amount of
training data consumed at the same epoch number:

| Method | Exploratory batch | Exposure at epoch 5,000 | Exposure at epoch 10,000 | Relative to batch 2 |
|---|---:|---:|---:|---:|
| AM-EVRPTW | 4 | 20M | 40M | 2x |
| EVRPTW-RL | 5 | 25M | 50M | 2.5x |
| DRL-TS | 2 | 10M | 20M | 1x |
| TERRAN | 88 | 440M | 880M | 44x |

Those runs would no longer compare four methods at a common Cus1000 exposure.
Keeping an even physical set `{4, 4, 2, 88}` under one effective batch would
require an effective batch of at least 88, increasing every method's exposure
44-fold while EVRPTW-RL and DRL-TS would still miss the strict memory target.
Using EVRPTW-RL batch 5 raises the least common effective batch to 440 and still
does not create a DRL-TS batch in the target interval.

Artificial allocation, repeated filler instances, or architecture changes are
not accepted ways to fill unused VRAM because they add no comparable training
signal.

## Formal decision

Keep `logical_batch = effective_batch = physical_batch = 2` for all four
Cus1000 methods. This preserves identical deterministic stream exposure and is
the only configuration compatible with the current scientific contract and
deadline.

The active runtime budget is
`drl_rq_runtime_budget_v12_min5000_max10000_tailval50`:

- minimum 5,000 and maximum 10,000 epochs;
- validation every 250 epochs through epoch 5,000;
- validation every 50 epochs after epoch 5,000;
- early-stop patience of 10 post-minimum validations;
- earliest possible early stop at epoch 5,500;
- selection checkpoint `best_overall.ckpt`.

For context, the observed TERRAN batch-2 non-validation epoch mean was about
30.9 seconds, while a 500-view sample-100 validation was about 0.81 hours. The
estimated TERRAN wall time is therefore about 3 days at the earliest possible
early stop and about 7.6 days at the 10,000-epoch cap. Unlike batch 88, this is
compatible with the September 18 deadline.
