# 2080ti_4_2 DRL queue

Hardware: 4 × RTX 2080 Ti

Assigned checkpoint jobs: 16 total;
pilot=3,
full=13.

From the repository root, activate any Python environment containing the project's required dependencies and run:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/logs.sh
```

`pilot.sh`, `full.sh`, and `resume.sh` detach with nohup.
`run.sh MODE` is the foreground/debug entrypoint. Environment paths may be
overridden before launch; committed defaults are repository-relative.

These four bundles intentionally contain no T1/T2/T3, best-of-50, or Cus2000
test jobs. Collect their `checkpoint_selected.pt`, validation, training result,
and provenance artifacts on the future central test server.
