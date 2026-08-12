# ALNS Solver

ALNS baseline for the current CLE-backed Stage-2 EVRPTW instances. The input is
always a `view_index.parquet` plus its `materialized/families` directory. This is
also the layout produced by the CLE + instance-ID restoration workflow, so the
solver does not have a separate restored-instance code path.

The solver contract matches Stage 2 and the exact benchmark:

- objective: directed `distance_matrix_km`;
- travel time: directed `running_time_shortest_matrix_s`;
- battery use: directed `running_time_path_energy_kwh`;
- charging: `full_charge_linear_v1`, using each station's own
  `charging_power_kw` and the configured charging efficiency;
- customer demand/capacity: cm3; time and time windows: seconds at the dataset
  boundary (ALNS converts them to minutes internally).

The runner independently replays every reported incumbent under this contract
before accepting it.

Stage 2's per-customer feasibility certificate is reconstructed into complete
depot/customer/multi-hop-charger routes with the generator's full-state path
cache. The shared layer first replays the complete route set; only a passing
witness is exposed to ALNS as a singleton warm start. ALNS validates it again,
and every published incumbent still goes through independent runner replay.
Missing or stale certificates fall back to solver-side repair.

## Cus50 test run

```bash
python EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/run_alns.py \
  --dataset_path EVRPTW_Dataset/Instances_v1/us_11city/generation_plan/compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet \
  --save_path EVRPTW_Benchmark/results/CLE_EVRPTW_v1/compatibility_cus50/test1/ALNS_Solver_2h \
  --num_workers 30 \
  --time_limit_s 7200 \
  --checkpoints_s 60,300,900,3600,7200 \
  --seed 2026 \
  --skip_completed
```

`--dataset_path` may instead point at any Stage-2 directory containing one or
more `view_index.parquet` files. Use `--scales Cus100,Cus500` to filter a mixed
Core index. If the canonical sibling family directory cannot be inferred, pass
`--family_root .../materialized/families`.

Normal runs are time-budgeted. `--delta_iters` and `--max_iters` are retained
only for controlled smoke tests or ablations.

For disjoint server runs, use either an exclusive filtered range
`--start_index START --end_index END` or stable hash shards
`--shard_count K --shard_index I`. Seeds are derived from `(base_seed,
view_id)` with the recorded `blake2b_view_id_v1` scheme, so a view receives the
same seed regardless of index order, slicing, worker count, or server. At most
`--max_in_flight` tasks are queued (default: twice the worker count). Common
OpenMP/BLAS pools are fixed to one thread per worker; set
`EVRPTW_META_THREADS_PER_WORKER` before launch only for an intentional override.

The reported algorithm clock excludes dataset I/O and structural validation.
It includes schema adaptation (including certificate reconstruction), the
complete `ALNS_Solver` constructor, and `solve()`; `solve()` receives only the
remaining allowance. Thus scale-dependent initialization is inside the same
two-hour budget as search.
The effective algorithm profile, construction strategy/source/time/route count,
and full profile JSON are stored in summary and solution metadata.

## Outputs

- `alns_summary.csv`: final status and final validated incumbent per instance;
- `alns_time_trace.csv`: objective and full route at 60, 300, 900, 3600, and
  7200 seconds (or custom checkpoints), with solver/profile/seed/run-contract
  identity for standalone cross-server merging;
- `solutions/<run-contract-fingerprint>/*.pkl`: final canonical solutions;
- `solutions/checkpoints/<run-contract-fingerprint>/*.pkl`: canonical solution
  at each checkpoint that has an incumbent.

For cross-solver collection, the summary exposes the same
`benchmark_status`, `benchmark_completed`, and `has_incumbent` fields as the
Exact runner. The time trace likewise contains `benchmark_status` in addition
to the solver-facing `status` field.

Checkpoint rows are strict: a route discovered after a checkpoint is never
written into that earlier checkpoint. If the algorithm naturally terminates
before two hours with a feasible solution, its final best route is copied only
forward to later checkpoints. A run with no feasible route by the time limit is
marked `UNFINISHED_NO_INCUMBENT`.

Each result is fsynced to an append-only JSONL journal. `--skip_completed`
recovers it after interruption and remains compatible with legacy CSV-only
directories for reading. A terminal row is skipped only when its
`run_contract_fingerprint` exactly matches the new task. The contract includes
the algorithm/profile, budget and checkpoints, base and per-view seeds, search
parameters, timing scope, and portable view-index/family identity; worker
count, shard/range selection, ordering, and output path are deliberately
excluded. Data identity includes the actual family/view manifests, generation
seeds, and byte-level hashes of the family terminal index, four matrices, and
small per-view artifacts; direct and CLE-restored copies intentionally share
the same identity. Each selected family is read for hashing only once per
launch. The canonical replay policy is versioned too. Therefore a 30-second pilot, regenerated data, another
seed/profile, or changed search parameter is rerun even in the same directory.
Contract-scoped artifact directories isolate those reruns. Legacy rows without
a contract fingerprint are conservatively rerun rather than assumed equivalent.
Canonical CSVs are atomically materialized at the end;
`--csv_flush_interval N` optionally refreshes them every N instances without
per-instance full-file rewrites. Two active runners may not share a
`--save_path`: the second receives an actionable warning and is rejected. Use a
distinct directory for each solver and shard, then merge canonical CSVs during
collection.

## Tests

```bash
python -m pytest -q \
  EVRPTW_Benchmark/MetaHeuristics/tests \
  EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver/tests
```

The opt-in `MetaHeuristics/tests/runner_ab_harness.py` runs one real Stage-2
view through both solvers in single-worker and process-pool modes and compares
their deterministic solution fields.
