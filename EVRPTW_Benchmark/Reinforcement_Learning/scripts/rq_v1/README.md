# DRL RQ server launch bundles

These four bundles generate training checkpoints only. All defaults are derived
from the checked-out repository; no machine-specific absolute data path is
embedded. The expected restored dataset location is:

`EVRPTW_Dataset/Instances_v2/us_11city`

Activate the `maojie` environment and run exactly one pilot launcher on each
server:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_2/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_3_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/pilot.sh
```

Each command prepares deterministic local artifacts if needed and then launches
its GPU queues through `nohup`/`setsid`. `status.sh` and `logs.sh` do not start
work. `resume.sh` resumes only committed formal checkpoints.

`full.sh` is intentionally present but blocked. It cannot start the 72 formal
runs until both the pilot gate report and a versioned G1--G8 PASS authorization
exist. Candidate exposure and batch settings are pilot inputs, not silently
accepted formal hyperparameters.

No SHA-256 or per-file content hashing is performed by these scripts.
