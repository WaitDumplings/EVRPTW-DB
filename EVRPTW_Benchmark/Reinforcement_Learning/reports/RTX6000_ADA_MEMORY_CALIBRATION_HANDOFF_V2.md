# RTX 6000 Ada memory calibration handoff v2

Date: 2026-09-05

Target: `scripts/rq_v1/a6000_2_1` on two NVIDIA RTX 6000 Ada Generation
48-GiB GPUs.

The RTX 2080 Ti jobs have been recalibrated after the rollout-limit revision.
The Ada jobs were not executed on the local 2080 Ti host, so the active config
is deliberately marked `rtx2080ti_calibrated_rtx6000_ada_revalidation_pending`.
Do not interpret the checked-in Ada caps as post-revision measurements.

## Settings that must remain fixed

| Scale | Rollout limit | Logical batch | Validation views | Candidates |
|---|---:|---:|---:|---:|
| Cus500 | 580 | 64 | 500 | 100 |
| Cus1000 | 1,200 | 2 | 500 | 100 |

- Seed: 1234.
- Calibration: exactly two logical training epochs followed by the complete
  fixed validation run.
- DRL-TS calibration: epoch 1 soft stage, epoch 2 hard stage.
- Validation/test decoding: sampling, best of 100, independently verified.
- Keep logical batches, deterministic streams, reward, normalization, model
  definitions and trajectory counts unchanged.
- AM-EVRPTW trains with 5 trajectories on every scale and TERRAN retains 100;
  EVRPTW-RL and DRL-TS retain one training trajectory. This hardware-invariant
  AM setting supersedes the earlier Ada-only sample-100 calibration.

Current starting caps are:

| Scale | AM-EVRPTW | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|
| Cus500 | 8 | 16 | 8 | 64 |
| Cus1000 | 2 | 2 | 2 | 2 |

## Required evidence

Calibrate all eight method/scale jobs independently. Each accepted row must
record process peak memory across training and validation, PyTorch allocated
peak, wall time, exit code, two completed epochs, 500 validation instances and
100 candidates. Retain failed upper candidates.

Use a 43-GiB process ceiling on a 48-GiB card. A CUDA OOM, killed process,
missing terminal artifact or incomplete validation is a failure. A two-epoch
checkpoint may produce infeasible routes; record that separately from the
memory result, since this short run is not model-quality acceptance.

REINFORCE jobs may use a non-divisor physical batch because the common trainer
now emits an exact final remainder microbatch with sample-weighted gradients.
TERRAN must use an exact divisor of its logical batch.

After selecting all eight caps:

1. update only `physical_batch_caps` in
   `configs/drl_rq_runtime_candidates_v2.yaml`;
2. rebuild all manifests with `build_rq_server_manifests`;
3. run the complete Reinforcement_Learning test suite;
4. change the config status only after all eight full calibration rows pass;
5. commit and push the clean executable tree;
6. start `a6000_2_1/full.sh --seed 1234` from that exact commit.

The obsolete `pilot.sh` workflow must not be restored. Formal runs remain
commit-scoped, and a prior checkpoint must not be resumed across executable or
rollout-limit changes.
