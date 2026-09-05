# RTX 2080 Ti per-job memory calibration v4

Date: 2026-09-05

Status: all 16 jobs assigned to the three RTX 2080 Ti bundles completed the
required calibration run. Formal training remained stopped.

## Frozen calibration contract

- Hardware: four local NVIDIA RTX 2080 Ti cards with 11,264 MiB each.
- Formal seed, model, reward, normalization, data representation and data
  stream are preserved.
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

- `scripts/build_2080ti_memory_calibration_manifest.py` builds one disposable
  four-job wave from the checked-in formal manifests.
- `scripts/run_2080ti_memory_calibration.py` materializes the exact first two
  logical epochs, runs jobs on isolated GPU slots, samples process memory and
  checks the complete validation contract.

These tools write only to their requested disposable output root. They do not
modify formal checkpoints or launch the long-running queues.
