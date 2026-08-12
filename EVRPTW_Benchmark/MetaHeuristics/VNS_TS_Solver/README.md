# VNS-TS Solver

VNS + Tabu Search baseline for the current CLE-backed Stage-2 EVRPTW
instances. It only accepts the current `view_index.parquet` plus
`materialized/families` layout. Instances restored from CLE + instance IDs and
instances materialized directly by Stage 2 therefore use the same runner.

The resource model matches Stage 2 and the exact benchmark:

- objective: directed `distance_matrix_km`;
- travel time: directed `running_time_shortest_matrix_s`;
- battery use: directed `running_time_path_energy_kwh`;
- charging: full recharge with the visited station's individual 11/100 kW (or
  other exported) `charging_power_kw` and charging efficiency;
- customer demand/capacity: cm3; all temporal quantities: seconds.

Every incumbent is independently replayed before it is admitted to the anytime
trace.

The scalable `fast` profile first publishes the complete Stage-2 constructive
certificate as a safe infinite-fleet incumbent. It then runs a short,
deadline-bounded best-fit consolidation in which distance is used only to rank
candidates and every accepted route is checked with the exact resource model.
If that pass reaches its deadline, all remaining singleton routes are retained;
a partial customer set is never published. VNS/TS search continues from the
consolidated solution for the remainder of the wall-clock budget.

## Cus50 test run

```bash
python EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver/run_vns_ts.py \
  --dataset_path EVRPTW_Dataset/Instances_v1/us_11city/generation_plan/compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet \
  --save_path EVRPTW_Benchmark/results/CLE_EVRPTW_v1/compatibility_cus50/test1/VNS_TS_Solver_2h \
  --num_workers 30 \
  --time_limit_s 7200 \
  --checkpoints_s 60,300,900,3600,7200 \
  --seed 2026 \
  --skip_completed
```

`--dataset_path` may point at a broader Stage-2 directory; use `--scales` for a
mixed Core index and `--family_root` when the family directory is stored
separately.

Search controls include `--eta_feas`, `--eta_dist`, `--tabu_iter`,
`--predefine_route_number`, and `--search_mode fast|full`. A formal run now
defaults to wall-clock-driven distance search: omitting `--eta_dist` installs a
deterministic very large iteration ceiling, leaving the time limit in control.
An explicit small `--eta_dist` remains available for smoke tests and ablations.
The default `fast` mode uses the versioned
`vns_ts_stage2_adaptive_fast_v4` profile with deterministic scale-adaptive
bounded neighborhoods; the profile, construction strategy, policy version and
all effective limits are recorded in every output.

For disjoint server runs, use either an exclusive filtered range
`--start_index START --end_index END` or stable hash shards
`--shard_count K --shard_index I`. Seeds use the recorded
`blake2b_view_id_v1` mapping from `(base_seed, view_id)`, so they do not change
with index order, slicing, worker count, or server. At most `--max_in_flight`
tasks are queued (default: twice the worker count). Common OpenMP/BLAS pools are
fixed to one thread per worker; set `EVRPTW_META_THREADS_PER_WORKER` before
launch only for an intentional override.

The algorithm clock excludes dataset I/O and structural validation. It includes
schema adaptation, the complete `VNSTSolver` constructor, and `solve()`;
`solve()` receives only the remaining wall-clock allowance. This puts all
scale-dependent matrix/neighborhood initialization inside the same two-hour
budget as search. Summary and solution metadata record the algorithm profile,
initial strategy/source/time/route count, and effective fast-policy limits.

## Outputs

- `vns_ts_summary.csv`: final status and final validated incumbent;
- `vns_ts_time_trace.csv`: objective and full route at every requested
  checkpoint, with solver/profile/seed/run-contract identity for standalone
  cross-server merging;
- `solutions/<run-contract-fingerprint>/*.pkl`: final canonical solutions;
- `solutions/checkpoints/<run-contract-fingerprint>/*.pkl`: checkpoint
  solutions.

For cross-solver collection, the summary exposes the same
`benchmark_status`, `benchmark_completed`, and `has_incumbent` fields as the
Exact runner. The time trace likewise contains `benchmark_status` in addition
to the solver-facing `status` field.

A late incumbent is never backfilled into an earlier checkpoint. Natural early
termination forwards the final incumbent only to later checkpoints. No
feasible solution at the time limit is `UNFINISHED_NO_INCUMBENT`.

Each completed result is fsynced to an append-only JSONL journal.
`--skip_completed` recovers it after interruption and accepts legacy CSV-only
directories for reading. It skips a terminal row only when the stored
`run_contract_fingerprint` exactly matches the new task. The contract covers
algorithm/profile, budget and checkpoints, base and per-view seeds, every
requested/effective search parameter, timing scope, and portable
view-index/family identity. Data identity includes the actual family/view
manifests, generation seeds, and byte-level hashes of the family terminal index,
four matrices, and small per-view artifacts, while direct and CLE-restored
copies deliberately share the same identity. Each selected family is read for
hashing only once per launch. The canonical replay policy is versioned too. Worker count,
shard/range selection, ordering, and output path are deliberately absent. Thus
a short pilot, regenerated data, or a seed/profile/search change is rerun in the
same directory. Contract-scoped artifact directories prevent such reruns from
overwriting or masquerading as one another. Legacy rows without a fingerprint
are conservatively rerun rather than assumed equivalent. Canonical CSVs are
atomically materialized at the end;
`--csv_flush_interval N` optionally refreshes them every N instances without
per-instance full-file rewrites. Two active runners may not share one
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
