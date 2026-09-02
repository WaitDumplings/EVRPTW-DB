# RTX 2080 Ti batch calibration v1

Date: 2026-09-02  
Hardware: 4 x NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each  
Environment: `maojie`, PyTorch 2.5.1+cu121  
Dataset: released Stage-2 v7 training/validation artifacts  
Scope: AM-EVRPTW, EVRPTW-RL, DRL-TS, and TERRAN at Cus50/Cus100/Cus500

> Superseded for launch: the scale-aware training rollout budgets were changed to
> Cus50=80, Cus100=140, Cus500=600, and Cus1000=1200 after this calibration.
> The measurements below remain provenance for the previous horizon only. Batch
> candidates, especially TERRAN Cus500 (450 to 600 steps), require a new memory
> and throughput pilot before any full job is approved.
>
> The replacement measurement is recorded in
> `RTX2080TI_ROLLOUT_BUDGET_PILOT_V2.md`.
> The full-data-pass runtime language is also historical: formal manifests now
> use the fixed 500,000 customer-exposure budget documented in
> `FIXED_EPOCH_TRAINING_BUDGET_V1.md`.

## Selection discipline

- Calibration used training views only. No T1/T2/T3 metric was read.
- The same selected batch is frozen for every seed and every RTX 2080 Ti
  server for a given method and scale.
- Candidates were exercised with real forward, rollout, backward, optimizer,
  checkpoint, and (for TERRAN) independent validation/verifier paths.
- Cus500 candidates and DRL-TS soft/hard phases were stressed over multiple
  consecutive batches. A candidate above 9.5 GiB allocated memory was rejected
  even if it happened to finish.
- A larger batch is not selected merely because it allocates more memory.
  A safety margin is retained for harder batches and CUDA allocator variance.
- Cus1000 remains assigned to RTX A6000 and was not calibrated on this host.

## Frozen RTX 2080 Ti batches

| Method | Cus50 | Cus100 | Cus500 |
|---|---:|---:|---:|
| AM-EVRPTW | 400 | 100 | 6 |
| EVRPTW-RL | 120 | 32 | 2 |
| DRL-TS | 80 | 24 | 1 |
| TERRAN | 400 | 200 | 50 |

Physical and effective batch are equal in protocol v1. These are
method-specific throughput settings; the matched scientific budget remains
the same complete data-pass count and therefore the same customer exposures.
Optimizer steps, environment transitions, wall time, and memory remain
mandatory report columns rather than being claimed as identical.

## Passing evidence

Peak values below are PyTorch maximum allocated bytes converted to GiB.
The job supervisor additionally samples process GPU memory with
`nvidia-smi` during formal pilots.

| Method | Scale | Candidate | Stress | Peak GiB | Result |
|---|---|---:|---|---:|---|
| AM-EVRPTW | Cus50 | 400 | 4 batches, seed 2345 | 7.24 | PASS |
| AM-EVRPTW | Cus100 | 100 | 4 batches, seed 3456 | 6.74 | PASS |
| AM-EVRPTW | Cus500 | 6 | 8 batches, seed 1234 | 8.51 | PASS |
| EVRPTW-RL | Cus50 | 120 | 4 batches, seed 2345 | 7.35 | PASS |
| EVRPTW-RL | Cus100 | 32 | 4 batches, seed 3456 | 6.99 | PASS |
| EVRPTW-RL | Cus500 | 2 | 8 batches, seed 2345 | 6.70 | PASS |
| DRL-TS | Cus50 | 80 | 8 soft + 8 hard batches | 6.47 max | PASS |
| DRL-TS | Cus100 | 24 | 8 hard batches, seed 2345 | 8.19 | PASS |
| DRL-TS | Cus500 | 1 | 8 soft + 8 hard batches | 5.23 max | PASS |
| TERRAN | Cus50 | 400 | 1 batch + verifier | 8.23 | PASS |
| TERRAN | Cus100 | 200 | 2 batches + verifier | 7.05 | PASS |
| TERRAN | Cus500 | 50 | 4 batches seed 1234 + 2 seed 2345 | 7.44 | PASS |

The TERRAN Cus500 b50 run was also observed at about 8,880 MiB process GPU
memory by `nvidia-smi`, below the 9.5 GiB formal gate.

## Rejected upper candidates

| Method | Scale | Candidate | Reason |
|---|---|---:|---|
| AM-EVRPTW | Cus500 | 8, 12, 16 | CUDA OOM |
| EVRPTW-RL | Cus500 | 3 | later stress batch CUDA OOM |
| DRL-TS | Cus50 | 120 hard | CUDA OOM |
| DRL-TS | Cus100 | 32 hard | 9.72 GiB, over 9.5 GiB gate |
| DRL-TS | Cus500 | 2 soft | 10.20 GiB, over 9.5 GiB gate |

These failures are calibration evidence only and occurred under `/tmp`.
They are not resumable experiment outputs or checkpoints.

## Fairness contract

The main matched comparison freezes:

1. the same released training views and seeded shuffle-cycle;
2. the same data-pass and customer-exposure budget;
3. seeds 1234, 2345, and 3456;
4. the same validation cohort and lexicographic checkpoint selection;
5. the same verified directed-road distance objective;
6. no test-set use for hyperparameter or checkpoint selection.

Trajectory multiplicity and optimizer structure remain method-native. In
particular, TERRAN uses 50 trajectories and PPO updates, so its transitions
and optimizer steps are not made artificially equal to the three REINFORCE
methods. They are recorded explicitly.

## Runtime warning for the candidate 100-pass protocol

Short stress runs extrapolate to roughly 1--16 hours for one data pass,
depending on method and scale. The slowest observed case is DRL-TS Cus500,
approximately 14--18 hours per pass across its soft/hard phases. Therefore the
candidate 100-pass budget can exceed 1,500 hours for one such seed.

This calibration does **not** authorize a full launch. The formal pilot must
produce versioned wall-time estimates and the global data-pass budget must be
reviewed once before any full job starts. If changed, it must change globally,
not by method, scale, seed, or observed test performance.
