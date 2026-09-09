# 2080ti_4_2 current benchmark queue

Hardware: 4 × RTX 2080 Ti. The queue has 5 formal jobs.

| GPU slot | Queue order | Method | Scale | Representation | Support |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | am_evrptw | Cus100 | G | Full-support |
| 1 | 0 | evrptw_rl | Cus100 | G | Full-support |
| 2 | 0 | drl_ts | Cus100 | G | Full-support |
| 3 | 0 | terran | Cus100 | G | Full-support |
| 0 | 1 | terran | Cus100 | E | Full-support |

Activate the project Python environment and run from the repository root:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/full.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/logs.sh
```

`full.sh` and `resume.sh` detach with nohup; existing launchers and GPU processes
are checked by the common runtime. The preverified stream mode and all frozen
stream checksums remain active. A changed TERRAN profile has its own recorded
path/hash and cannot silently resume an incompatible checkpoint.

The four commands `full.sh`, `resume.sh`, `status.sh`, `logs.sh` forward to [the current RQ bundle](../rq_v1/2080ti_4_2/README.md). The remaining pilot/start/run scripts and jobs.jsonl are retained historical protocol entry points. See [profile notes](../../configs/2080ti/README.md).
