# a6000_2_1 DRL queue

Hardware: 2 × RTX A6000

Assigned jobs: 184 total; pilot=4,
full=12,
evaluate=168.

From the repository root, activate the `maojie` environment and run:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/logs.sh
```

`pilot.sh`, `full.sh`, `resume.sh`, and `evaluate.sh` detach with nohup.
`run.sh MODE` is the foreground/debug entrypoint. Environment paths may be
overridden before launch; committed defaults are repository-relative.

Before evaluation, copy every checkpoint listed in
`assignment_summary.json.external_checkpoint_job_ids` into the same relative
`EVRPTW_OUTPUT_ROOT` location on this server. A shared output filesystem also
satisfies this requirement.
