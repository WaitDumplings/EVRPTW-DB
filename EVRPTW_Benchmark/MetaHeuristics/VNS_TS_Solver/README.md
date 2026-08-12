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
`--predefine_route_number`, and `--search_mode fast|full`. The default `fast`
mode uses bounded candidate neighborhoods and may naturally terminate before
the two-hour cap.

## Outputs

- `vns_ts_summary.csv`: final status and final validated incumbent;
- `vns_ts_time_trace.csv`: objective and full route at every requested
  checkpoint;
- `solutions/*.pkl`: final canonical solutions;
- `solutions/checkpoints/*.pkl`: checkpoint solutions.

For cross-solver collection, the summary exposes the same
`benchmark_status`, `benchmark_completed`, and `has_incumbent` fields as the
Exact runner. The time trace likewise contains `benchmark_status` in addition
to the solver-facing `status` field.

A late incumbent is never backfilled into an earlier checkpoint. Natural early
termination forwards the final incumbent only to later checkpoints. No
feasible solution at the time limit is `UNFINISHED_NO_INCUMBENT`.

## Tests

```bash
python -m pytest -q \
  EVRPTW_Benchmark/MetaHeuristics/tests/test_stage2_metaheuristics.py
```
