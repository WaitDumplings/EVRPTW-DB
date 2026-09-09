# 2080ti_3_1 current benchmark queue

Hardware: 3 × RTX 2080 Ti. The queue has 3 formal jobs.

| GPU slot | Queue order | Method | Scale | Representation | Support |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | am_evrptw | Cus100 | G | Random-10%-support |
| 1 | 0 | terran | Cus100 | G | Random-10%-support |
| 2 | 0 | am_evrptw | Cus100 | G | Coverage-10%-support |

Activate the project Python environment and run from the repository root:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_3_1/full.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_3_1/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_3_1/logs.sh
```

`full.sh` and `resume.sh` detach with nohup; existing launchers and GPU processes
are checked by the common runtime. The preverified stream mode and all frozen
stream checksums remain active. A changed TERRAN profile has its own recorded
path/hash and cannot silently resume an incompatible checkpoint.

See [2080 Ti profile notes](../../../configs/2080ti/README.md) for parameters, calibration evidence and the separate A6000 stable-cost algorithm.
