# TERRAN

TERRAN is the reinforcement-learning baseline package for EVRPTW-DB. It uses the
shared `EVRPTW_Env` Gymnasium-style environment and keeps POMO-style parallel
rollouts through the environment's `n_traj` dimension.

## Components

- `models/`: migrated TERRAN attention backbone, actor, and critic.
- `env_factory.py`: creates the shared EVRPTW environment with optional TERRAN
  reward shaping.
- `data_pool.py`: online service-territory pool for training-time instance sampling.
- `pbrs.py`: optional potential-based reward shaping switches.
- `train.py`: PPO-style TERRAN training entry point.
- `eval.py`: fixed-dataset best-of-`n_traj` sample evaluation.
- `prepare_eval_data.py`: fixed Cus15 eval-set generation.
- `smoke_test_terran.py`: verifies the model and environment interface on a
  pickle instance.


## Optional Precomputed Service-Territory Pool

Training can use a reusable service-territory pool prepared by the dataset generator:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_region_pool \
  --num-territories 1024 \
  --latent-customer-pool-size 5000 \
  --cs-candidate-pool-size 120 \
  --seed 20260525

CUDA_VISIBLE_DEVICES=0 conda run -n maojie python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus15_terran.yaml \
  --seed 1515 \
  --territory-pool-path EVRPTW_Dataset/AC_v1/ServiceTerritoryPool_1024
```

`mother_board_pool_size` remains the backward-compatible config key for the number of active service territories held by one run.
`territory_pool_path` is optional: if loading fails or the pool has fewer territories
than `mother_board_pool_size`, training automatically falls back to online
service-territory generation. The default replacement policy is `cycle`, which reuses the
precomputed pool for stale-region replacement without regenerating region
geometry.

## PBRS Switches

`PotentialRewardConfig` exposes the reward-shaping controls without modifying
the shared environment:

- `use_customer_pbrs`: served-customer progress potential using
  `gamma * Phi(s_next) - Phi(s)`.
- `use_repair_distance_pbrs`: single-customer depot-customer-depot repair
  workload potential using the same gamma potential-difference form.
- `use_feasible_ratio_pbrs`: feasible-unserved-customer ratio potential from
  the action mask. This is optional and disabled in the default PBRS configs.
- `use_terminal_heuristic`: terminal success bonus and failure penalty. This is
  an auxiliary shaping term, not strict PBRS, and is disabled by default.
- `customer_pbrs_mode`: default configs use `progress`, the strict gamma
  potential-difference form.

Evaluation should usually disable PBRS and use the base objective reward. PBRS is
intended for training only.

## Cus15 Baselines

The default Cus15 setup trains two baselines with identical architecture and
hyperparameters:

- `configs/cus15_terran.yaml`: base TERRAN, PBRS disabled.
- `configs/cus15_terran_pbrs.yaml`: TERRAN+PBRS with customer-progress,
  repair-distance progress, and terminal heuristic enabled.

Training samples online Cus15/CS3 operating days from a 32-region service-territory
pool and does not save each training instance. Evaluation uses a fixed
1000-instance AC-v1 evaluation suite and sample decoding: each instance runs `n_traj=50`
trajectories and keeps the best feasible trajectory by objective distance.


## Normalization And Training Metrics

The shared RL environment keeps physical dynamics in seconds, kilometers, kWh,
and cm3, but model-facing observations are normalized: locations are mapped to
`[0, 1]`, demand and current load are fractions of vehicle capacity, time
windows/service/current time are fractions of the operating horizon, battery
state is a fraction of battery capacity, and the model-facing capacity scalars
are `1.0`. Training rewards are distance-normalized for value-function stability;
`objective_distance_km` in `info` and eval CSVs remains the physical kilometer
objective used for benchmark comparison.

## Periodic Evaluation

`configs/cus15_terran.yaml` and `configs/cus15_terran_pbrs.yaml` run fixed-set
evaluation every `eval_interval` epochs. The default uses the fixed Cus15/CS3
eval set with POMO-8 sample decoding (`eval_n_traj: 8`). Evaluation metrics are
written into both `train_log.csv` and `eval_log.csv`:

- `eval_avg_objective_distance_km`
- `eval_avg_vehicle_count`
- `eval_feasible_rate`
- `eval_avg_runtime_s`

## Example

```bash
EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/scripts/prepare_eval_cus15.sh 200

EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/scripts/train_cus15_4gpu.sh

EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/scripts/eval_cus15.sh
```
