# Cus100 / seed 1234 / 2026-09-11

This launcher owns only ten fresh training configurations. It does not invoke the old `rq_v1/full.sh`, resume old results, start larger scales, or automatically evaluate Road T1.

| Server label | GPU | Experiment | Model | Training and validation source |
|---|---:|---|---|---|
| 2080ti_4_1 | 0 | TR02 | AM-EVRPTW | Road |
| 2080ti_4_1 | 1 | TR01 | AM-EVRPTW | TERRAN synthetic Euclidean |
| 2080ti_4_1 | 2 | TR06 | TERRAN | Road |
| 2080ti_4_1 | 3 | TR05 | TERRAN | TERRAN synthetic Euclidean |
| 2080ti_4_2 | 0 | TR04 | DRL-TS | Road |
| 2080ti_4_2 | 1 | TR03 | DRL-TS | TERRAN synthetic Euclidean |
| 2080ti_4_2 | 2 | TR18 | EVRPTW-RL | Road |
| 2080ti_4_2 | 3 | TR17 | EVRPTW-RL | TERRAN synthetic Euclidean |
| 2080ti_3_1 | 1 | TR10 | RRNCO | Road |
| 2080ti_3_1 | 2 | TR09 | RRNCO | TERRAN synthetic Euclidean |
| 2080ti_3_1 | 0 | — | Reserved | No scheduled job |

These labels are deployment roles, not asserted SSH hostnames. At startup the launcher records the actual hostname, GPU index and UUID, Python, CPU thread settings, repository commit and input hashes. It binds each trainer using a single GPU UUID. Occupied GPUs and populated output directories stop preflight; existing work is never killed.

## Preparation

The authoritative parameters live in `cus100_seed1234_jobs.jsonl`. All ten jobs in the delivered manifest passed the current engineering calibration and are enabled. See [CUS100_SMOKE_REPORT.md](CUS100_SMOKE_REPORT.md) for measured batches, memory, time estimates and initial policy-quality limitations. The builder is designed for isolated smoke/profile calls without starting the formal launcher:

```python
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.launch import build_command, load_jobs
job = next(job for job in load_jobs() if job['experiment_id'] == 'TR04')
command = build_command(job, overrides={
    'physical_batch_size': 4, 'effective_batch_size': 4,
    'training_epochs': 3, 'minimum_training_epochs': 3,
    'early_stop_start_epoch': 0, 'early_stop_patience_validations': 0,
    'validation_every_epochs': 3, 'validation_checkpoints': 1,
    'validation_views': 2, 'output_dir': '/tmp/unique_cus100_smoke_TR04',
})
```

A JSON profile can override fields by method and/or experiment ID. For example, `{"drl_ts": {"physical_batch_size": 5, "effective_batch_size": 5}}` is a schema example, not a measured recommendation. Per-experiment fields take precedence. Method-specific CLI options use the `extra_args` list.

Once the final profiles and both datasets are frozen:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.prepare_artifacts \
  --profiles /path/to/measured_batches.json --enable
```

Without `--profiles`, preparation uses the existing manifest. Without `--enable`, it leaves existing enabled flags as written. All ten jobs must declare `calibration_status: passed` before `--enable` succeeds. Preparation does not start training.

Each source has 50,000 train views and 500 validation views. The default budget is 5,000 minimum / 10,000 maximum logical epochs, validation every 100, patience five after the minimum, with 30 trajectories in training and validation. Training/validation horizon is 240/360. Method-specific physical and effective batches are explicit; the manifest records resulting exposure differences.

Streams have an exact length matching each job's budget. Equal lengths share one artifact. Different lengths are prefixes of the same source/seed permutation sequence; preparation compares every ordered ID and prefix SHA256. It records the verification in `results/cus100_20260911/artifacts/shared_stream_preparation.json`. No test index is read during preparation.

## Files required on each server

Repository code and manifest must be from the same deployment revision. Data and result artifacts are ignored by Git and require a separate copy or shared storage:

- `EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911/` in its entirety.
- `EVRPTW_Benchmark/results/cus100_20260911/artifacts/` in its entirety.
- Existing Road release `EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823/`: the core train/validation indexes and materialized families used by their Cus100 rows.

If the destination already has the same Road release, it can reuse those files. Road train has 50,000 Cus100 views in 5,000 families; validation has 500 views in 500 separate families. Every training/validation family was found on 2080ti_4_1. Their parent matrices occupy approximately 90.53 GiB, plus manifests, terminal indexes and view attributes. The parent data are stored at Cus1000 scale and sliced to Cus100; retaining those files does not launch a Cus1000 experiment. CLE routing files are not needed by the existing materialized-view loader.

For a copy between machines, run `rsync -a` for the two new directories above from the prepared server to the identical relative locations under the destination repository. Use your actual SSH hostname; the deployment labels do not resolve remote hostnames.

Optional environment overrides are `CUS100_PYTHON`, `CUS100_ROAD_ROOT`, `CUS100_SYNTHETIC_ROOT` and `CUS100_OUTPUT_ROOT`. Dataset roots may be absolute or repository-relative. Default output is `EVRPTW_Benchmark/results/cus100_20260911/`; training outputs are fresh `runs/TRxx/` directories.

## Start on the corresponding server

Activate the project's Python environment, then run the role's script from the repository:

```bash
conda activate maojie
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus100_20260911/2080ti_4_2/full.sh
```

On the three-GPU server, use `2080ti_3_1/full.sh`. On this four-GPU server, use `2080ti_4_1/full.sh`. The launcher detaches from the terminal and writes its PID and log path. There is no need to add another `nohup`. To inspect readiness before starting, append `--mode preflight`. A pending calibration or missing data causes a clear error and starts no trainer.

Check `results/cus100_20260911/launchers/<server>/status.json` and `runs/TRxx/{stdout.log,stderr.log,launch_record.json}`. A successful launcher start is not proof of successful training. Completion additionally requires the trainer's `training_result.json`, a selected checkpoint and the declared epoch range.

To stop this round, send `SIGTERM` to the recorded launcher PID. Its handler stops its own child process groups and retains saved checkpoints. No old queue or restart watcher is installed.

## Freeze and transfer one deployment package

After the final corpus, profiles and streams pass their checks, build the package on the preparation server:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.package_deployment \
  --output /data/cus100_deployment_20260911.tar.gz
```

Transfer that archive and its `.tar.gz.json` checksum sidecar to each destination. The archive includes current tracked file contents, new untracked source files, the complete synthetic corpus, new streams and source provenance. Existing Road data and training outputs are excluded. Verify the archive SHA256 against the sidecar, then extract at the existing target Git repository root (the archive does not install a Conda environment):

```bash
tar --same-permissions -xzf /path/to/cus100_deployment_20260911.tar.gz -C /data/Maojie/ICLR/EVRPTW-DB
```

The target launcher verifies its actual source against the transferred deployment's `source_version`; it never calls an unchanged Git HEAD the actual code version. A changed or provisional synthetic corpus fails preflight. Every task uses two CPU threads (`OMP`, `MKL`, `OPENBLAS`, `NUMBA`) and one explicitly bound GPU.

## Source-specific reward scaling

Synthetic tasks require a shared training-only calibration because their fleet and route lengths differ from Road. The objective coefficients remain those of `rivian_energy_vehicle_cost_v2.json`. Once the full synthetic corpus is complete, run the deterministic reference constructor on a seed-1234-derived, fixed 500-instance training cohort:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.calibrate_synthetic_reward \
  --workers 8 --output EVRPTW_Benchmark/results/cus100_20260911/artifacts/reward_synthetic_feasible4
```

This produces `reward_contract.json`; point all five E jobs at that same file. The five G jobs retain their Road v3 reward contract. The E scalar is the median complete objective of deterministic, independently verified constructor routes; its failure base is the linear 99th percentile of normalized reference cost plus one. The constructor uses the frozen synthetic single-customer witnesses, explicit current cost coefficients and fixed insertion candidate limits. No random ALNS search is run, and validation/test instances are not loaded for calibration. `--pilot --count 8` only estimates cost/runtime and never writes a formal reward contract. E launch preflight checks that the reward contract is bound to the exact completed synthetic corpus and training index.

## 2080ti_3_1 desktop GPU correction

On this server GNOME remote desktop uses GPU 0 as a C+G process. RRNCO Road now uses GPU 1 and synthetic E uses GPU 2; GPU 0 remains available to the desktop. Existing compute-process occupancy checks still apply to both selected training GPUs. Batch 50, trajectories 30, model code, data, rewards and training streams are unchanged. The measurements in the smoke report remain the original local calibration.

After the original data package has been extracted, update the experiment branch with `git pull --ff-only`, activate `caliroute`, set `CUS100_PYTHON="$CONDA_PREFIX/bin/python"`, and run `2080ti_3_1/full.sh`. Do not extract the old code package again after pulling the correction. The launcher recognizes only the exact approved deployment delta, verifies every changed source record and both job configurations, saves the old deployment metadata, and records the new source identity. Unknown source changes still fail. No dataset regeneration or large archive transfer is needed.

The live 2080ti_4_1 experiment retains its original frozen checkout; this deployment correction was prepared in a separate worktree.
