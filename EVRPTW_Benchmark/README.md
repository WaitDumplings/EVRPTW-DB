# EVRPTW_Benchmark

Benchmark solvers are organized by method family.

```text
Exact/Gurobi_Solver
MetaHeuristics/Greedy_Solver
MetaHeuristics/VNS_TS_Solver
MetaHeuristics/ALNS_Solver
Reinforcement_Learning/TERRAN
```

Each solver should eventually expose a common interface:

```python
solve(instance, config) -> solution
```

The benchmark runner will validate every returned solution with `EVRPTW_Core` before writing leaderboard metrics.
