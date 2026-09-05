# DRL performance optimization — protocol unchanged

2026-09-04, branch `drl-benchmark-adapters`.
Baseline: `6ea2419bdab4c69e41009863c54a079546737b5b`.

This patch removes repeated work in training and online validation. It does not
change the then-active v10 budget, physical/logical batches, shared ID stream,
seeds,
validation cohort, best-of-100 decoding, objective, charging rules, feasibility
semantics, architecture or checkpoint selection. It launches no training.

The later v11 launch budget changes only the hard cap from 10,000 to 6,000 and
selects `best_overall.ckpt` formally; the optimizations and equivalence evidence
described here remain applicable.

## Implementation

| Component | Optimization | Semantic boundary |
|---|---|---|
| Shared online validation | Use light info throughout AM / EVRPTW-RL / DRL-TS decoding; export routes and merged sequences once at the end | Includes completed and step-budget-exhausted trajectories. Independent verifier and candidate selection remain unchanged. |
| Baselines / evaluation | Skip unused log-likelihood accumulation; avoid constructing categorical distributions for cost-only greedy decoding | Training actor likelihood remains enabled. Sampling retains its original sampler and RNG sequence. |
| AM | Cache the actual attention glimpse key/value projections within a rollout | Original weights and autograd retained; no cache across optimizer updates. |
| EVRPTW-RL | Cache static features, travel-time tensor conversion and static edge messages within a rollout | Dynamic structure-to-vector computation and recurrent state remain action-dependent. Gradients retained. |
| TERRAN evaluation | Encode static inputs once; skip unused critic, log-probability and entropy outputs | Encoder cache is eval-only. Training dropout execution is unchanged. |
| TERRAN training / PPO | Keep only model-consumed observation fields; reuse immutable static arrays; transfer static inputs once per PPO chunk | Dynamic snapshots remain distinct; same chunks, encoder/dropout calls, loss and update order. Full environment/PBRS observations remain available. |
| DRL-TS soft stage | Vectorize the same mask rules and reuse the pre-action mask | No changed penalties, action eligibility or soft-to-hard schedule. |

Model parameter names, state-dictionary keys and optimizer parameter order are
unchanged. This does **not** authorize bypassing existing commit/provenance
checks to resume an older run. Do not overwrite result directories or silently
restart experiments after pulling. Cross-commit continuation needs an explicit
provenance decision, not a change to the resume guard.

Reference paths retained for diagnostics:

- AM / EVRPTW-RL: `use_static_cache=False`.
- All three REINFORCE rollouts: `compute_log_likelihood=True` remains the
  default; common entrypoints disable it only when unused.
- TERRAN eval: `cache_static_embeddings=False`,
  `compact_observations=False`, `final_routes_only=False`.
- TERRAN training collection: `compact_observations=False` retains the
  full-buffer reference used in PPO gradient comparisons.

## Timing and verification

AM / EVRPTW-RL / DRL-TS `runtime_s` now includes static preparation/encoding;
the old timer started after setup. Final route export is also included. Do not
directly compare historical decode-only runtime with this setup-plus-decoding
metric. Internal inference time excludes environment construction/reset, data
loading and the independent verifier. TERRAN keeps batch-total and per-instance
amortized fields, with encoding and final export inside its timer.

Local result: **169 tests passed**, including logits, recurrent states,
parameter gradients, actions, RNG state, routes, objectives, completion flags,
TERRAN PPO loss/gradients, soft-mask oracle comparisons, truncated trajectories,
verifier selection and mock-clock timing boundaries. Environment: Python 3.11,
PyTorch 2.9.1, macOS CPU, one OMP/MKL thread.

Three additional server-path tests fail locally because macOS `realpath` lacks
the GNU `-m` option used by the existing Linux scripts. Those scripts are
unchanged. Vendored `reference_materials` tests are outside this project suite.
This is not an all-platform test PASS; CUDA and a restored frozen corpus were
not available locally.

Reproduce the tested local scope from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  EVRPTW_Benchmark/Reinforcement_Learning \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/reference_materials \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/tests/test_rq_server_environment.py -q
```

On Linux, omit the second `--ignore` to include the server-path tests.

## Optional server diagnostic

Use an idle GPU and the normal project environment. This diagnostic trains
nothing and does not modify job manifests or formal result directories:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.benchmark_drl_hotpaths \
  --method all --device cuda:0 \
  --customers 50 --stations 10 --batch-size 1 --candidates 16 \
  --steps 150 --repeats 3 --warmup 1 \
  --output-json /tmp/drl_hotpaths_cus50.json
```

Default inputs are synthetic fixtures, with the same randomly initialized
evaluation policy on both paths. To use actual train views, add a train index
and the matching customer scale, for example:

```bash
--dataset-path /path/to/release/generation_plan/core/train/view_index.parquet \
--customers 100
```

Cus50 uses `generation_plan/compatibility_cus50/train/view_index.parquet`.
Supply `--family-root` only for artifacts outside the normal release layout.
The diagnostic never reads test views. Increase scale/candidates only within
available memory; it does not determine a safe formal batch size.

The external timer includes environment construction, reset, setup, decoding
and final route export, with CUDA synchronization at rollout boundaries.
Independent verifier comparisons are outside that timer and timed separately.
JSON records hardware, Git provenance, dirty-tree status, IDs, seeds, repeated
timings, CUDA peak allocated memory, routes and selected-verifier equivalence.
A mismatch sets `passed=false` and exit status 1; unavailable CUDA never silently
falls back to CPU. The reference is the retained uncached/full-info computation
in this checkout, not a separate execution of the historical Git revision.

Local Cus50 and Cus100 fixture diagnostics (batch 1, 16 candidates, one warmup,
three repeats) passed equivalence for all four methods, with the largest gain
in TERRAN evaluation. These are not trained-policy quality measurements,
full-epoch timings, GPU measurements or end-to-end training-speedup guarantees.

## Deployment and remaining profiling

The existing four server `full.sh` entrypoints automatically use the optimized
paths for new launches. No new launch-script type or changed budget is needed.
The patch does not establish a shorter total training duration; measure actual
data loading, rollout, backward/PPO, validation, checkpoint I/O and peak memory
on the target server before revising schedules or physical batches.

Dataset prefetch, wider validation batching, a GPU-native environment, mixed
precision and fused attention are not included. They need separate memory,
numerical and seed-stream checks. Reducing validation candidates or training
exposures would be a protocol change, not a performance optimization.
