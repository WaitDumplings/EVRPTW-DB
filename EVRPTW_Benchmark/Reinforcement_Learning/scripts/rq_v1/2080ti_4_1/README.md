# 2080ti_4_1 current benchmark queue

Hardware: 4 × RTX 2080 Ti. The queue has 8 formal jobs.

| GPU slot | Queue order | Method | Scale | Representation | Support |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | am_evrptw | Cus50 | G | Full-support |
| 1 | 0 | evrptw_rl | Cus50 | G | Full-support |
| 2 | 0 | drl_ts | Cus50 | G | Full-support |
| 3 | 0 | terran | Cus50 | G | Full-support |
| 0 | 1 | terran | Cus100 | G | Coverage-10%-support |
| 1 | 1 | am_evrptw | Cus100 | E | Full-support |
| 2 | 1 | evrptw_rl | Cus100 | E | Full-support |
| 3 | 1 | drl_ts | Cus100 | E | Full-support |

Activate the project Python environment and run from the repository root:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/full.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/logs.sh
```

`full.sh` and `resume.sh` detach with nohup; existing launchers and GPU processes
are checked by the common runtime. The preverified stream mode and all frozen
stream checksums remain active. A changed TERRAN profile has its own recorded
path/hash and cannot silently resume an incompatible checkpoint.

See [2080 Ti profile notes](../../../configs/2080ti/README.md) for parameters, calibration evidence and the separate A6000 stable-cost algorithm.
