# EVRPTW_Benchmark

Benchmark solvers are organized by method family.

```text
Exact/Gurobi_Solver
MetaHeuristics/VNS_TS_Solver
MetaHeuristics/ALNS_Solver
Reinforcement_Learning/TERRAN
```

Exact and metaheuristic runners consume the CLE-backed Stage-2
`view_index.parquet` plus its `materialized/families` store. They expose a
common solver interface:

```python
solve(instance, config) -> solution
```

The benchmark runner will validate every returned solution with `EVRPTW_Core` before writing leaderboard metrics.


## Summary Comparison

Use `compare_solver_summaries.py` to join two solver summary CSV files by `instance_id` and compute objective, runtime, and vehicle-count differences.

```bash
python EVRPTW_Benchmark/compare_solver_summaries.py \
  --reference_summary EVRPTW_Benchmark/results/.../Gurobi_Solver/gurobi_summary.csv \
  --candidate_summary EVRPTW_Benchmark/results/.../ALNS_Solver/alns_summary.csv \
  --reference_name gurobi \
  --candidate_name alns \
  --save_path EVRPTW_Benchmark/results/.../alns_vs_gurobi.csv
```
