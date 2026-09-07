# RTX 2080 Ti per-job memory calibration v5

Date: 2026-09-07

Status: PASS for the 16 jobs assigned to the three RTX 2080 Ti bundles under
`drl_rq_runtime_budget_v15_ntraj50_maxbatch_warmstart`.

## Frozen probe contract

- Exactly two logical training epochs followed by one complete validation.
- Validation: fixed 500 views, sampling, 50 candidates per view.
- Training trajectories: AM=5, EVRPTW-RL=1, DRL-TS=1, TERRAN=50.
- Rollout limits: Cus50=65 and Cus100=120; validation receives ceil(1.5x).
- Peak process GPU memory sampled through `nvidia-smi` every 0.2 seconds.
- Batch was selected independently for every method and scale; exposure
  matching across methods was intentionally not required.
- PASS means no OOM, exit code zero, two epochs recorded and a complete
  500x50 validation record. It is not a convergence claim.

## Selected effective/physical batches

| Scale | AM-EVRPTW | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|
| Cus50 | 2,304 | 336 | 144 | 480 |
| Cus100 | 800 | 96 | 40 | 280 |

## Complete train-plus-validation evidence

| Representation / condition | Method | Scale | Batch | Peak process GiB | Wall s | Exit | Verifier summary |
|---|---|---:|---:|---:|---:|---:|---|
| G / Full-support | AM-EVRPTW | Cus50 | 2,304 | 10.193 | 205.21 | 0 | PASS |
| G / Full-support | EVRPTW-RL | Cus50 | 336 | 10.158 | 126.36 | 0 | PASS |
| G / Full-support | DRL-TS | Cus50 | 144 | 9.727 | 108.44 | 0 | PASS |
| G / Full-support | TERRAN | Cus50 | 480 | 10.135 | 160.18 | 0 | PASS |
| G / Full-support | AM-EVRPTW | Cus100 | 800 | 10.322 | 202.15 | 0 | PASS |
| G / Full-support | EVRPTW-RL | Cus100 | 96 | 10.482 | 194.59 | 0 | FAIL (quality only) |
| G / Full-support | DRL-TS | Cus100 | 40 | 10.258 | 160.35 | 0 | PASS |
| G / Full-support | TERRAN | Cus100 | 280 | 9.972 | 226.24 | 0 | PASS |
| E / Full-support | AM-EVRPTW | Cus100 | 800 | 10.322 | 200.66 | 0 | PASS |
| E / Full-support | EVRPTW-RL | Cus100 | 96 | 10.482 | 198.98 | 0 | FAIL (quality only) |
| E / Full-support | DRL-TS | Cus100 | 40 | 10.258 | 160.99 | 0 | PASS |
| E / Full-support | TERRAN | Cus100 | 280 | 9.972 | 215.09 | 0 | PASS |
| G / Random-10%-support | AM-EVRPTW | Cus100 | 800 | 10.322 | 203.94 | 0 | PASS |
| G / Random-10%-support | TERRAN | Cus100 | 280 | 9.972 | 221.33 | 0 | PASS |
| G / Coverage-10%-support | AM-EVRPTW | Cus100 | 800 | 10.322 | 207.05 | 0 | PASS |
| G / Coverage-10%-support | TERRAN | Cus100 | 280 | 9.972 | 228.90 | 0 | PASS |

The two EVRPTW-RL Cus100 verifier-summary failures are retained as two-epoch
model-quality evidence. Both processes completed the full validation with no
OOM, so they do not invalidate memory calibration.

## Warm-start contract

A formal v15 job searches only the same representation, condition, method,
scale and seed under source commit `481176332c1e683f52b28777d3423d036f5c010f`.
If `best.ckpt` exists, only model weights are imported; optimizer, logical
epoch, baseline history, validation selection and early-stop state reset. A
missing exact checkpoint produces a recorded fresh start. Cross-condition,
cross-scale and cross-method fallback is forbidden.

All four exact Cus50 checkpoint formats were checked. AM, EVRPTW-RL and TERRAN
also completed two new epochs plus 500x50 validation after weight import.
DRL-TS passed direct model/reward/auxiliary/soft-stage compatibility and exact
state loading. Its two-epoch memory probe intentionally changes the soft-stage
boundary to exercise both stages, so that diagnostic is cold-started rather
than misrepresented as a compatible formal warm start.

Cus500/Cus1000 remain assigned to RTX 6000 Ada. Their current microbatches are
conservative and must be calibrated on that host under n-traj=50 before being
called maximum-utilization settings.
