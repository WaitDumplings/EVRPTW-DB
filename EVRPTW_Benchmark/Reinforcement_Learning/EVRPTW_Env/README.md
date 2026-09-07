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
AC/DC intake limit. The model-facing station scalar is the time required to
charge the full vehicle battery at that station, divided by the operating-day
horizon; relative power within one instance is retained only as a diagnostic
observation. Legacy fixed-duration modes are retained only for diagnostic
replay and are not canonical benchmark settings.

A charging station is masked after its first visit in the current vehicle
route. Returning to the depot closes that route and clears the station mask, so
the same physical station remains available to later vehicles. This is a
route-local anti-cycle rule, not a global station-copy limit.

The hard action mask also enforces forward-feasible-path (FFP) safety. Every
admitted customer or charging-station action has a time- and energy-feasible
continuation back to the depot. The return witness may contain multiple
charging stations, but removes every station already visited by the current
route from the entire path, including intermediate hops. A candidate station
may be used once as the witness source. The usual rule still requires a route
to serve a customer before it closes; when that rule would otherwise leave a
charger-only prefix with an empty mask, a physically feasible depot leg is
exposed as an emergency escape. Python and JIT masks implement the same
contract. If an instance is already actionless at reset under the selected
method policy, the environment records `no_feasible_action` immediately and
exposes only a depot sentinel; this lets the first rollout step report the
non-horizon failure without presenting all-`-inf` logits to the policy.

`allow_consecutive_station_actions` makes the station-transition assumption
part of that FFP contract. It defaults to `True`, preserving the shared/TERRAN
behavior and allowing a return witness with multiple unvisited stations. When
set to `False`, station-to-station actions are masked and every return witness
is restricted to a direct depot leg or `customer -> one station -> depot`.
Paper adapters that remove consecutive charging actions must select this mode
before the base mask is computed; filtering a permissive mask afterwards is
not FFP-safe.

Formal training uses one deterministic distance scale estimated only from its
frozen training pool. Distance edges use that scale, travel-time edges use the
operating horizon, and energy edges use battery capacity. Validation and test
reuse the scale persisted in the training checkpoint. Physical directed-road
kilometres remain a separate diagnostic in the new cost track.

## Active cost objective

The optional `objective_config` selects the versioned electricity-plus-vehicle
objective used by all four formal training methods. Its complete semantics
and parameter source are in
[`COST_OBJECTIVE_CONTRACT_V1.md`](../COST_OBJECTIVE_CONTRACT_V1.md).
The environment records `vehicles_started` on valid depot departures, charges
the fixed fee once there, and accumulates the electricity term from distance.
This does not change the energy matrix, masks, charging time or fleet limit.
`reward_objective_scale` normalizes the whole scalar cost; the old
`reward_distance_scale_km` still normalizes distance features. Omitting the
objective retains the explicit legacy distance behavior.

## Route Export

`info["routes"]` stores per-vehicle routes. `info["route_sequence"]` stores the
benchmark-wide merged route sequence, for example:

```text
[[0, 3, 2, 1, 0], [0, 7, 5, 0]] -> [0, 3, 2, 1, 0, 7, 5, 0]
```
