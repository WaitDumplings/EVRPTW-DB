# a6000_2_1 DRL queue

Hardware: 2 × RTX A6000

Assigned checkpoint jobs: 16 total;
pilot=4,
full=12.

From the repository root, activate the `maojie` environment and run:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/logs.sh
```

`pilot.sh`, `full.sh`, and `resume.sh` detach with nohup.
`run.sh MODE` is the foreground/debug entrypoint. Environment paths may be
overridden before launch; committed defaults are repository-relative.

These four bundles intentionally contain no T1/T2/T3, best-of-50, or Cus2000
test jobs. Collect their `checkpoint_selected.pt`, validation, training result,
and provenance artifacts on the future central test server.
