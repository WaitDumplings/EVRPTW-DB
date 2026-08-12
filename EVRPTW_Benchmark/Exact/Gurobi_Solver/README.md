# Gurobi exact MILP baseline

This directory contains the small-scale exact MILP baseline for the CLE-backed
EVRPTW Stage-2 dataset. The primary input is a Stage-2 `view_index.parquet` plus
the shared `materialized/families/` store. Both supported dataset-acquisition
workflows produce this same layout:

1. direct Stage-2 generation; or
2. deterministic reconstruction from the released CLE plus instance IDs and
   family parameters.

There is no legacy pickle input fallback.

## Model contract

For every directed terminal arc, the V1 solver uses:

- `distance_matrix_km` in the minimization objective;
- `running_time_shortest_matrix_s` for time propagation and time windows; and
- `running_time_path_energy_kwh` for battery propagation.

Demand and vehicle capacity are both measured in `cm3`. The fleet is unlimited,
each customer is served exactly once, waiting is allowed, and all routes start
and end at the selected depot within the stored operating horizon.

The fixed `--cs_copies` value is part of the exact MILP expansion and defaults
to the benchmark contract value `2`. Each dummy
copy inherits the effective power of its physical charging station. If the
remaining battery on arrival at station `q` is `b_q`, full linear charging takes

```text
(battery_capacity_kwh - b_q)
-------------------------------- * 3600 seconds.
charging_efficiency * power_q_kw
```

The departure battery is then reset to full. The old instance-wide fixed
charging time is not used for Stage-2 views. `full_cs_to_depot_time_s` is an
environment/mask acceleration cache and is not substituted for explicit MILP
station visits.

`--cs_copies` must be an integer of at least one; invalid values are rejected
rather than silently changing the model expansion.

Every returned incumbent is independently replayed after optimization. The
replay checks customer coverage, directed distance, running time, waiting,
service time, time windows, volume, battery, station-specific charging time,
and return before the horizon. A Gurobi incumbent is reported as feasible only
when this replay passes.

## Stage-2 input layout

The normal release layout is:

```text
Instances_v1/us_11city/
├── generation_plan/
│   └── compatibility_cus50/
│       ├── val/view_index.parquet
│       └── test/test1_new_seed_same_cities/view_index.parquet
└── materialized/families/<family_id>/
    ├── family_manifest.json
    ├── terminal_index.parquet
    ├── matrices/
    └── views/<view_id>/
```

When the view index is inside this canonical tree, the runner infers the family
root. For a separately copied or reconstructed bundle, pass `--family_root`.
The solver deliberately does not distinguish directly generated matrices from
reconstructed matrices once they satisfy this layout and schema.

## Smoke test

From the repository root:

```bash
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src:\
EVRPTW_Benchmark/Exact/Gurobi_Solver \
python EVRPTW_Benchmark/Exact/Gurobi_Solver/run_gurobi.py \
  --dataset_path /path/to/view_index.parquet \
  --family_root /path/to/materialized/families \
  --save_path /path/to/gurobi_smoke \
  --limit 1 \
  --time_limit_s 60 \
  --checkpoints_s 60 \
  --cs_copies 2 \
  --workers 1 \
  --threads 1 \
  --verbose
```

The MILP test is skipped automatically when the current host has no usable
Gurobi license; pure checkpoint and route-replay tests still run.

## Compatibility Cus50 test run

```bash
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src:\
EVRPTW_Benchmark/Exact/Gurobi_Solver \
python EVRPTW_Benchmark/Exact/Gurobi_Solver/run_gurobi.py \
  --dataset_path \
EVRPTW_Dataset/Instances_v1/us_11city/generation_plan/compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet \
  --family_root \
EVRPTW_Dataset/Instances_v1/us_11city/materialized/families \
  --save_path \
EVRPTW_Benchmark/results/CLE_EVRPTW_v1/compatibility_cus50/test1/Gurobi_Solver \
  --time_limit_s 7200 \
  --checkpoints_s 60,300,900,3600,7200 \
  --cs_copies 2 \
  --workers 4 \
  --threads 1 \
  --skip_completed \
  --verbose
```

Use a small pilot before choosing server concurrency. `--threads 1` is the
paper-comparison contract. `--workers` controls independent Gurobi processes
and must respect the server license and memory limits.

The optional vehicle-count tie break is disabled by default so that the full
budget and every callback remain attached to the published distance objective.
If explicitly enabled, it is diagnostic only: the published final objective
and route always remain the frozen primary distance solution.

For Stage-2 inputs, `--start_index A --end_index B` selects the half-open stable
row-position range `[A,B)` in the view index. This supports disjoint server
shards without interpreting the hashed `view_id`. `--skip_completed` resumes
from an existing summary.

## Outputs

The runner incrementally writes:

- `gurobi_summary.csv`: final status, incumbent, bound/gap, Stage-2 identity,
  matrix sources, charging model, power range, and route-replay result;
- `gurobi_time_trace.csv`: the best incumbent available at each checkpoint;
- `solutions/*.pkl`: final `EVRPTWSolution` records; and
- `solutions/checkpoints/*.pkl`: checkpoint incumbent records.

The default benchmark checkpoints are exactly 60, 300, 900, 3600, and 7200
seconds (1, 5, 15, 60, and 120 minutes). Each time-trace row contains both
`objective_distance_km` and the corresponding `routes_json`; the same route is
also persisted under `solutions/checkpoints/`.

Checkpoint values are strictly causal: a route first found after checkpoint
`t` is never written at `t`. If the distance optimization ends before 120
minutes, its final best route and objective are forward-filled to every later
checkpoint with `reached_checkpoint=false` and
`source=final_after_early_stop`. This includes a proven optimum found early.

`benchmark_status` makes result completeness explicit:

- `COMPLETED_OPTIMAL`: a valid incumbent with proven distance optimality;
- `COMPLETED_WITH_INCUMBENT`: a valid time-limited incumbent;
- `UNFINISHED_NO_INCUMBENT`: the budget ended without any incumbent;
- `INVALID_INCUMBENT`: Gurobi returned a route that failed independent replay;
- `NO_FEASIBLE_SOLUTION`: infeasibility/unboundedness was concluded without a
  route.

For `UNFINISHED_NO_INCUMBENT`, objectives and routes remain empty at all
checkpoints where no incumbent existed; no artificial objective is inserted.

Every distinct checkpoint route is independently replayed before CSV and
solution output. A rejected route is removed from the feasible checkpoint
columns and is never written as a feasible checkpoint solution. Its replay
result, original objective, and original route remain available in
`route_validation_json`, `diagnostic_objective_distance_km`, and
`diagnostic_routes_json` for debugging.

`--skip_completed` skips valid completed results, proven no-solution results,
and a full `TIME_LIMIT` attempt without an incumbent. It does not skip invalid
incumbents or interrupted unfinished attempts.

`OPTIMAL` is reported only when Gurobi proves optimality. A time-limit result
with an incumbent remains a feasible time-limited incumbent and retains its
best bound and MIP gap. A run without an incumbent remains uncovered rather
than being assigned an artificial objective.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src:\
EVRPTW_Benchmark/Exact/Gurobi_Solver \
pytest -q -p no:cacheprovider \
  EVRPTW_Benchmark/Exact/Gurobi_Solver/tests
```

The regression suite includes strict checkpoint causality and forward-fill
semantics, checkpoint-route replay, an energy-infeasible direct route, a
required-CS MILP, and 11 kW versus 100 kW arrival-SOC charging.

## Environment

Install the packages in `requirements.txt`. `gurobipy` must match the installed
Gurobi major version and license. The runner records model/environment errors
per view instead of terminating the whole batch; use `--save_traceback` while
debugging.
