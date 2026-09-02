# EVRPTW_Benchmark

Benchmark solvers are organized by method family.

```text
Exact/Gurobi_Solver
MetaHeuristics/VNS_TS_Solver
MetaHeuristics/ALNS_Solver
Reinforcement_Learning/TERRAN
Reinforcement_Learning/AM_EVRPTW
Reinforcement_Learning/EVRPTW_RL
Reinforcement_Learning/DRL_TS
```

The learning baselines are being standardized under
[`Reinforcement_Learning/BASELINE_IMPLEMENTATION_PLAN.md`](Reinforcement_Learning/BASELINE_IMPLEMENTATION_PLAN.md).
Their common transition, charging, masking, and evaluation semantics are frozen
in [`Reinforcement_Learning/CHARGING_ADAPTER_CONTRACT.md`](Reinforcement_Learning/CHARGING_ADAPTER_CONTRACT.md).
The complete learning experiment matrix, thirteen-GPU allocation, and required
three-class launch-script contract are specified in
[`Reinforcement_Learning/DRL_EXPERIMENT_AND_GPU_SCRIPT_DIRECTIVE.md`](Reinforcement_Learning/DRL_EXPERIMENT_AND_GPU_SCRIPT_DIRECTIVE.md).

Exact and metaheuristic runners consume the current CLE-backed Stage-2
`view_index.parquet` plus its `materialized/families` store. The frozen input
contract is family schema `cle_evrptw_materialized_matrix_family_v3`, view
schema `cle_evrptw_materialized_view_v4`, generation contract
`stage2_construct_valid_v3`, and `parent_index_view` matrix storage. They expose a
common solver interface:

```python
solve(instance, config) -> solution
```

The benchmark runner validates every returned solution with `EVRPTW_Core`
before writing leaderboard metrics. Charging follows
`full_charge_linear_derated_v2`: station power is multiplied by the exported
`charging_power_derating_factor` (0.90 in the frozen profile).

The frozen Cus50 Test-1 launchers for Exact, ALNS, and VNS-TS are documented
in [`test_scripts/`](test_scripts/README.md).


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
