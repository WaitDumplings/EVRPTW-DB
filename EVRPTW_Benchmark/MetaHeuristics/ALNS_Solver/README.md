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

## Outputs

- `alns_summary.csv`: final status and final validated incumbent per instance;
- `alns_time_trace.csv`: objective and full route at 60, 300, 900, 3600, and
  7200 seconds (or custom checkpoints);
- `solutions/*.pkl`: final canonical solutions;
- `solutions/checkpoints/*.pkl`: canonical solution at each checkpoint that
  has an incumbent.

For cross-solver collection, the summary exposes the same
`benchmark_status`, `benchmark_completed`, and `has_incumbent` fields as the
Exact runner. The time trace likewise contains `benchmark_status` in addition
to the solver-facing `status` field.

Checkpoint rows are strict: a route discovered after a checkpoint is never
written into that earlier checkpoint. If the algorithm naturally terminates
before two hours with a feasible solution, its final best route is copied only
forward to later checkpoints. A run with no feasible route by the time limit is
marked `UNFINISHED_NO_INCUMBENT`.

## Tests

```bash
python -m pytest -q \
  EVRPTW_Benchmark/MetaHeuristics/tests/test_stage2_metaheuristics.py
```
