# RTX 6000 Ada memory calibration handoff v1

Date: 2026-09-04

Target server bundle: `scripts/rq_v1/a6000_2_1`

Target hardware: 2 x NVIDIA RTX 6000 Ada Generation (48 GiB each)

## What was done on the 2080 Ti servers

The 2080 Ti configuration was not chosen from model size alone. Every accepted
method/scale pair ran this sequence:

```text
fixed released training stream
-> two real logical training updates
-> checkpoint write/read
-> fixed validation cohort
-> stochastic sample-100 decoding
-> independent route verifier
-> process and allocator peak-memory checks
```

The effective batch was kept common across methods within a scale. Only the
physical batch was changed; exact gradient accumulation made up the remaining
microbatches. A candidate was accepted only if it completed without OOM, its
validation/verifier passed, and the whole-process GPU peak stayed below 10 GiB
on an 11 GiB card. The final evidence is in
`RTX2080TI_SAMPLE100_MEMORY_REVALIDATION_V3.md`.

The Ada server must use the same method. Do not estimate its final batch by
scaling the 2080 batch by the VRAM ratio.

## Frozen scientific settings

Memory calibration may change only `physical_batch_caps`. It must not change:

- seed 1234;
- 1,000 logical epochs;
- the deterministic training stream;
- logical/effective batch or environments per epoch;
- Cus500 rollout horizon 600 and Cus1000 horizon 1,200;
- TERRAN training trajectories 100;
- validation/test decoding `sampling` with exactly 100 candidates;
- validation cohort size 500 for the final confirmation;
- verifier-feasible-first, minimum-directed-distance model selection.

Current Ada logical batches and conservative starting caps are:

| Scale | Logical batch | AM | EVRPTW-RL | DRL-TS | TERRAN |
|---|---:|---:|---:|---:|---:|
| Cus500 | 64 | 4 | 2 | 1 | 32 |
| Cus1000 | 2 | 1 | 1 | 1 | 1 |

## Required local procedure

Run everything from the repository root. Dataset discovery is repository
relative; if the restored dataset is under a sibling runtime tree, set only
`EVRPTW_RESTORE_ROOT` as documented by `scripts/rq_v1/README.md`.

First verify the exact GPU model, clean checkout, data, and conservative jobs:

```bash
git status --short
nvidia-smi --query-gpu=index,name,memory.total --format=csv
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/prepare_artifacts.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/pilot.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/status.sh --seed 1234
```

Do not start `full.sh` merely because the conservative pilot passes. First
calibrate each method/scale pair by increasing only its cap in
`configs/drl_rq_runtime_candidates_v2.yaml`, rebuilding the manifests, and
rerunning a fresh commit-scoped pilot. Keep one candidate change per commit so
the result is attributable.

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_rq_server_manifests
python -m pytest -q \
  EVRPTW_Benchmark/Reinforcement_Learning/tests/test_rq_server_manifests.py \
  EVRPTW_Benchmark/Reinforcement_Learning/tests/test_drl_job_runtime.py
```

The runtime intentionally requires a clean tree and stores outputs below the
executable commit. Commit the candidate before launching it; do not overwrite a
previous pilot result or manually edit `job_result.json`.

## Candidate search

Use a monotone search from the conservative cap. Each physical batch must divide
the logical batch exactly, otherwise it changes or complicates accumulation.

- Cus500 logical batch 64: admissible physical batches are
  `1, 2, 4, 8, 16, 32, 64`.
- Cus1000 logical batch 2: admissible physical batches are only `1, 2`.

For Cus500, double until the first OOM or memory-gate failure, then retain the
largest passing divisor. For Cus1000, test batch 2 once; keep 1 if it fails.
Do this independently for AM-EVRPTW, EVRPTW-RL, DRL-TS, and TERRAN. TERRAN's
100 training trajectories make its base-environment batch incomparable to the
other three models, so do not copy its cap to another method.

The Ada operational ceiling is 43 GiB per process. This leaves roughly 5 GiB
for CUDA context, allocator variability, harder graph batches, and system use.
Reject a candidate if any of the following occurs:

- CUDA OOM, exit 137/-9, or missing terminal artifacts;
- process peak above 43 GiB;
- validation or independent verifier failure;
- changed effective batch, stream, horizon, trajectories, or candidate count;
- a peak cannot be measured reliably.

The standard pilot uses two logical updates and eight validation views for a
fast gate. After the largest provisional batches are found, perform one final
confirmation with two updates and the complete fixed 500-view validation cohort
at sample-100. This last confirmation is important: validation can have a
different memory peak from training. Do not authorize `full.sh` from the
eight-view pilot alone.

For that final confirmation, set `pilot.validation_views: 500` in
`drl_rq_runtime_candidates_v2.yaml`, rebuild the manifests, run the two test
files above, and commit the config plus generated manifests. Then run
`a6000_2_1/pilot.sh --seed 1234` from the clean commit. Leave the value at 500
in the selected executable commit so its local pilot is the actual gate for the
subsequent full run; do not patch a generated `jobs.jsonl` by hand. Because
output directories are commit-scoped, earlier eight-view pilots remain intact
as evidence. Check each final `provenance.json` for
`training_epochs=2`, `validation_views=500`,
`validation_decode_type=sampling`, and `validation_candidate_count=100` before
accepting it.

For every candidate retain:

```text
provenance.json
training_result.json
validation_summary.json
job_result.json
stdout.log
stderr.log
nvidia-smi model/driver snapshot
```

The acceptance table must report method, scale, logical batch, physical batch,
microbatch count, allocator peak, process peak, wall time, validation views,
candidate count, feasibility rate, and PASS/FAIL reason. Failed upper candidates
are evidence and must not be deleted.

## Applying the selected Ada caps

After all eight method/scale pairs are confirmed:

1. write the selected values to `physical_batch_caps`;
2. keep `candidate_logical_batch` unchanged;
3. rebuild all four server manifests;
4. run the manifest/runtime tests above;
5. verify the generated Ada jobs still contain only seed 1234 and only
   Cus500/Cus1000;
6. commit and push the clean executable revision;
7. rerun the final Ada pilot under that exact commit;
8. launch `a6000_2_1/full.sh --seed 1234` only after all eight local pilots pass.

If the Ada agent changes anything other than physical batch sizing or discovers
an environment/verifier defect, it must stop and report the issue instead of
continuing the memory sweep.

