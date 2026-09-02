# 2080ti_4_1 DRL queue

Hardware: 4 × RTX 2080 Ti

Assigned jobs: 60 total; pilot=3,
full=13,
evaluate=44.

From the repository root, activate the `maojie` environment and run:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/logs.sh
```

`pilot.sh`, `full.sh`, `resume.sh`, and `evaluate.sh` detach with nohup.
`run.sh MODE` is the foreground/debug entrypoint. Environment paths may be
overridden before launch; committed defaults are repository-relative.

Before evaluation, copy every checkpoint listed in
`assignment_summary.json.external_checkpoint_job_ids` into the same relative
`EVRPTW_OUTPUT_ROOT` location on this server. A shared output filesystem also
satisfies this requirement.
