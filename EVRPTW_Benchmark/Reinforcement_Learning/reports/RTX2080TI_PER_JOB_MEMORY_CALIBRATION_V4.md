# RTX 2080 Ti per-job memory calibration v4

Date: 2026-09-05

Status: historical, frozen and non-launchable. All 16 jobs assigned to the
three RTX 2080 Ti bundles at the time completed the required calibration run;
formal training remained stopped. This statement describes the 2026-09-05
evidence only. As of 2026-09-06, Cus50/Cus100 have independent frozen reference
calibrations and the signed gate authorizes the 16 current RTX 2080 Ti jobs.
The historical inventory remains disabled and cannot be executed as a current
formal manifest.

## 2026-09-06 current-contract smoke addendum

After reward/FFP integration, all four Cus50 methods completed two training
epochs followed by the full fixed 500-view, best-of-100 validation. All four
returned exit code 0 and their verifier summaries passed. This was an isolated
`/tmp` smoke run, not a formal training result.

| Method | Physical batch | Peak process GPU (GiB) | Wall time (s) | Train+validation |
|---|---:|---:|---:|---|
| AM-EVRPTW | 1,024 | 4.66 | 159.05 | PASS |
| EVRPTW-RL | 224 | 8.09 | 181.65 | PASS |
| DRL-TS | 132 | 8.76 | 189.67 | PASS |
| TERRAN | 256 | 9.91 | 322.64 | PASS |

The smoke ran from executable base `f688fcd` plus the pending four-scale
reward/manifest changes recorded by the commit that adds this addendum. TERRAN
batch 256 remains below the 11,264 MiB device limit; its next exact-divisor
step is 128 and would materially underuse the card.

## Frozen calibration contract

- Hardware: four local NVIDIA RTX 2080 Ti cards with 11,264 MiB each.
- Formal seed, model, reward, normalization, data representation and data
  stream in effect at the time are preserved as historical evidence. They are
  not asserted to match the current reward contract.
- Current rollout limits: Cus50=65, Cus100=120, Cus500=580, Cus1000=1200.
- Every calibration runs exactly two logical training epochs.
- DRL-TS runs one soft-stage epoch and one hard-stage epoch.
- Every calibration then evaluates the fixed 500-view validation cohort with
  sampling and 100 candidates per view.
- Peak process GPU memory is sampled through `nvidia-smi` every 0.2 seconds.
- A calibration PASS means process exit 0, two completed training epochs and
  one complete 500-view/100-candidate validation record. It is not a claim that
  a model trained for only two epochs has converged or is feasible.

The executable base was commit `40e39b2`; the calibration ran with the exact
working-tree changes recorded by the commit that adds this report. No formal
training was launched from the calibration tree.

## Selected physical batches

| Scale | AM-EVRPTW | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|
| Cus50 | 1,024 | 224 | 132 | 256 |
| Cus100 | 256 | 68 | 34 | 128 |

REINFORCE methods use sample-weighted gradient accumulation. A final smaller
microbatch is permitted, so physical batches need not divide the logical
batch. TERRAN retains exact divisors because one PPO logical epoch currently
uses a fixed-size vector environment for every physical rollout.

## Complete per-job evidence

Peak values below are process-level GiB (`bytes / 2^30`), not only live tensor
allocation. Every row completed two training epochs and validation 500 x 100.

| Representation / condition | Method | Scale | Batch | Peak GiB | Exit | Calibration |
|---|---|---:|---:|---:|---:|---|
| G / Full-support | AM-EVRPTW | Cus50 | 1,024 | 2.635 | 0 | PASS |
| G / Full-support | EVRPTW-RL | Cus50 | 224 | 8.094 | 0 | PASS |
| G / Full-support | DRL-TS | Cus50 | 132 | 8.762 | 0 | PASS |
| G / Full-support | TERRAN | Cus50 | 256 | 9.910 | 0 | PASS |
| G / Full-support | AM-EVRPTW | Cus100 | 256 | 1.973 | 0 | PASS |
| G / Full-support | EVRPTW-RL | Cus100 | 68 | 7.986 | 0 | PASS |
| G / Full-support | DRL-TS | Cus100 | 34 | 8.818 | 0 | PASS |
| G / Full-support | TERRAN | Cus100 | 128 | 8.338 | 0 | PASS |
| E / Full-support | AM-EVRPTW | Cus100 | 256 | 1.973 | 0 | PASS |
| E / Full-support | EVRPTW-RL | Cus100 | 68 | 7.986 | 0 | PASS |
| E / Full-support | DRL-TS | Cus100 | 34 | 8.818 | 0 | PASS |
| E / Full-support | TERRAN | Cus100 | 128 | 8.338 | 0 | PASS |
| G / Random-10%-support | AM-EVRPTW | Cus100 | 256 | 1.973 | 0 | PASS |
| G / Random-10%-support | TERRAN | Cus100 | 128 | 8.338 | 0 | PASS |
| G / Coverage-10%-support | AM-EVRPTW | Cus100 | 256 | 1.973 | 0 | PASS |
| G / Coverage-10%-support | TERRAN | Cus100 | 128 | 8.338 | 0 | PASS |

The independent verifier was executed during every validation. Three
two-epoch TERRAN G runs had `verifier_summary_passed=false`; this means their
short-training checkpoints did not produce fully feasible validation routes.
It is retained as model-quality evidence and does not invalidate the memory
calibration, whose purpose is to exercise the entire validation path.

## Boundary decisions and unavoidable exceptions

- AM-EVRPTW is already at physical batch = logical batch (Cus50 1,024 and
  Cus100 256). Its 2--3 GiB use cannot be raised by batch tuning without
  changing the common logical data budget or the model, so it remains below
  the requested band.
- EVRPTW-RL Cus50 batch 224 and Cus100 batch 68 are the selected values. Their
  peaks are approximately 8.09 and 7.99 GiB.
- DRL-TS batch 144/Cus50 reached 9.793 GiB and batch 36/Cus100 reached 9.351
  GiB. They were reduced to 132 and 34, producing 8.762 and 8.818 GiB.
- TERRAN Cus100 batch 128 produces 8.338 GiB. For Cus50, the admissible
  neighboring exact divisors are 128 and 256; there is no divisor between
  them. Batch 256 completed train and validation at 9.910 GiB. This is retained
  because batch 128 substantially underutilizes the card, but it is explicitly
  outside the nominal 8--9 GiB target and leaves about 1.1 GiB device reserve.

No padding tensors, artificial caches, reduced validation cohort, changed
candidate count, changed reward, or changed logical batch were used merely to
make memory utilization look uniform.

## Reproduction tools

- `configs/drl_rq_2080ti_memory_calibration_inventory_v1.json` freezes the 16
  historical source rows independently of the current formal queues.
- `scripts/build_2080ti_memory_calibration_manifest.py` renders audit-only,
  disabled rows from that inventory. It no longer reads the active manifests.
- `scripts/run_2080ti_memory_calibration.py` explicitly rejects those
  historical rows. A new executable calibration requires calibrated Cus50 and
  Cus100 reward terms and a new versioned manifest contract.

Rendering writes only to the requested output path and cannot launch training.


## Post-v4 AM trajectory-5 revalidation (2026-09-05)

After the hardware-invariance review, AM-EVRPTW training was changed from one
trajectory on RTX 2080 Ti and 100 trajectories on the large-scale Ada jobs to
five trajectories per base instance on every scale. The 2080 Ti physical and
effective batches were not changed.

Five affected 2080 Ti job variants completed two logical training epochs and
the full fixed 500-view, sample-100 validation with the independent verifier:

| Job variant | Batch | AM train trajectories | Process peak (GiB) | Result |
|---|---:|---:|---:|---|
| Cus50 G Full-support | 1,024 | 5 | 4.658 | PASS |
| Cus100 G Full-support | 256 | 5 | 3.465 | PASS |
| Cus100 E Full-support | 256 | 5 | 3.465 | PASS |
| Cus100 G Random-10%-support | 256 | 5 | 3.465 | PASS |
| Cus100 G Coverage-10%-support | 256 | 5 | 3.465 | PASS |

No CUDA OOM occurred. All five runs exited zero, completed exactly two training
epochs, recorded one 500-instance/100-candidate validation row, and completed
the verifier. The retained formal setting is therefore AM training trajectories
= 5 under runtime budget
`drl_rq_runtime_budget_v13_am5_min5000_max10000_tailval50`.
