# RTX 2080 Ti sample-100 memory revalidation v3

Date: 2026-09-04

Executable commit under observation: `31b18b054e4e20db1b0c41859cce9318216d3682`

Hardware: 4 x NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each

Runtime budget: `drl_rq_runtime_budget_v6_scale_aware_gpu_val500_es3`

## Decision

The active Cus50/Cus100 physical batches remain approved for the 2080 Ti
servers. No batch reduction is required. All publication validation and test
jobs use seeded stochastic decoding with 100 candidates per instance.

The operational process-memory ceiling is 10 GiB per 11 GiB card. The largest
retained two-update pilot peak is 8.879 GiB (DRL-TS Cus50). During the live
formal run, after every Cus50 method had completed at least one 500-view,
sample-100 validation, a 30-second one-Hz device sample observed a maximum of
9,100 MiB. No CUDA OOM or failed verifier was observed.

## Active batches

Physical batches are method-specific memory controls. Effective batches remain
identical within a scale; exact gradient accumulation preserves one common
logical update.

| Scale | Effective batch | AM | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|---:|
| Cus50 | 1,024 | 1,024 | 128 | 128 | 128 |
| Cus100 | 256 | 256 | 64 | 32 | 128 |

TERRAN additionally uses 100 training trajectories per base environment. The
other three adapters use one sampled training rollout. All four methods use 100
candidates at validation and test time.

## Retained two-update pilot evidence

Each listed job executed two real optimizer updates and a sample-100 validation
through the independent verifier. `job_result.json` reports process GPU memory
sampled by the job supervisor, while `training_result.json` separately records
the PyTorch allocator peak.

| Method | Scale | Physical batch | Process peak | Result |
|---|---|---:|---:|---|
| AM-EVRPTW | Cus50 | 1,024 | 7.559 GiB | PASS |
| EVRPTW-RL | Cus50 | 128 | 4.721 GiB | PASS |
| DRL-TS | Cus50 | 128 | 8.879 GiB | PASS |
| TERRAN | Cus50 | 128 | 5.393 GiB | PASS |
| TERRAN | Cus100 | 128 | 8.361 GiB | PASS |

The remaining Cus100 method pilots were measured before task redistribution;
their committed aggregate range is 6.32--8.49 GiB. Their individual disposable
calibration directories were intentionally not treated as release artifacts.
The current launcher README and manifest tests freeze the approved batches so a
later manifest rebuild cannot silently revert them.

## Live formal-run evidence

The four Cus50 jobs were deliberately left running while this audit was made.
At the audit point they had completed the following fixed-cohort validations:

| Method | Completed validation checks | Latest epoch | Views | Candidates/view | Verifier feasibility |
|---|---:|---:|---:|---:|---:|
| AM-EVRPTW | 5 | 250 | 500 | 100 | 100% |
| EVRPTW-RL | 4 | 200 | 500 | 100 | 100% |
| DRL-TS | 3 | 150 | 500 | 100 | 100% |
| TERRAN | 1 | 50 | 500 | 100 | 100% |

The 30-second device-memory sample was stable at:

| GPU / method | Used memory | Below 10 GiB |
|---|---:|---|
| GPU 0 / AM-EVRPTW | 7,810 MiB | yes |
| GPU 1 / EVRPTW-RL | 4,842 MiB | yes |
| GPU 2 / DRL-TS | 9,100 MiB | yes |
| GPU 3 / TERRAN | 5,530 MiB | yes |

This sample is supporting live evidence, not a replacement for the supervisor's
whole-job peak. The formal jobs are still in progress, so terminal peak values
will be taken from their final `job_result.json` files.

## Scope and limitations

- This is a memory and execution revalidation, not a claim that training has
  converged.
- Cus500 and Cus1000 are not approved on 2080 Ti and remain assigned only to
  the two 48 GiB RTX 6000 Ada GPUs.
- A larger 2080 Ti physical batch is not authorized while the current formal
  run is active. Testing one would require stopping useful work and would not
  change the within-scale scientific budget.
- Any later change to trajectories, rollout horizon, model width, precision,
  candidate count, or validation batching invalidates this memory evidence.

