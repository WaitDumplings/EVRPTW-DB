# EVRPTW RL Environment

`EVRPTWVectorEnv` is the shared Gymnasium-style environment for reinforcement
learning baselines in this benchmark. It loads the canonical pickle
`EVRPTWInstance` schema and keeps the vectorized `n_traj` rollout dimension used
by POMO-style methods.

## API

```python
from evrptw_core.io import load_instance
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv

instance = load_instance("EVRPTW_Dataset/.../instance_000000.pkl")
env = EVRPTWVectorEnv(instance, n_traj=8)

obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(actions)
```

The environment follows the Gymnasium return convention:

- `reset(...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

`reward`, `terminated`, and `truncated` are arrays with shape `(n_traj,)`.

## Action Space

Node convention is shared with the benchmark solvers:

- `0`: depot
- `1..N`: customers
- `N+1..N+M`: charging stations

The action passed to `step` is an integer array with shape `(n_traj,)`. The
current feasibility mask is available in `obs["action_mask"]` and
`info["action_mask"]`.

## Canonical Travel, Energy, And Charging

Canonical runs consume the matrices exported by Stage 2:

- `running_time_shortest_matrix_s` for travel time;
- `running_time_path_energy_kwh` for energy;
- `distance_matrix_km` for the distance objective.

The environment deliberately does not infer time from a single average speed
or infer running-time-path energy from objective distance.  The compatibility
mode `matrix_mode="legacy_derived"` is diagnostic only and is not admissible in
benchmark tables.

The environment keeps legacy DRL compatibility by exposing
`obs["current_battery"]` as the consumed battery fraction since the last full
charge. `obs["remaining_battery"]` is also provided for models that prefer
remaining capacity.

Two additional observations support paper-faithful adapters without changing
the canonical transition contract:

- `obs["remaining_demand"]` is a per-node vector whose customer entries become
  zero after service;
- `obs["remaining_vehicle_ratio"]` is a nonbinding fleet-budget observation
  based on the canonical upper bound of one vehicle per customer.  It exists
  only for architectures whose original global context included the number of
  available vehicles; canonical feasibility never rejects an action because
  of this value.

Charging station actions perform immediate full charging.  Canonical mode is
`charging_mode="station_power_full"`; it uses the per-station effective power
exported by Stage 2 and the exported derating factor:

```text
charge_time_s = 3600 * energy_added_kWh /
                (station_power_kW * charging_power_derating_factor)
```

The generator has already capped station power by the reference vehicle's
AC/DC intake limit.  Legacy fixed-duration modes are retained only for
diagnostic replay and are not canonical benchmark settings.

## Route Export

`info["routes"]` stores per-vehicle routes. `info["route_sequence"]` stores the
benchmark-wide merged route sequence, for example:

```text
[[0, 3, 2, 1, 0], [0, 7, 5, 0]] -> [0, 3, 2, 1, 0, 7, 5, 0]
```
