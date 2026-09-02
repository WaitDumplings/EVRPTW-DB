# Shared EVRPTW Learning-Environment Contract

## Scope

This contract standardizes the physical state transition, feasibility mask,
route export, and final evaluation used by every learning baseline.  It does
not standardize each paper's training reward.

## Canonical matrices and units

- objective distance: `distance_matrix_km` (km);
- travel time: `running_time_shortest_matrix_s` exposed through
  `EVRPTWInstance.shortest_time_matrix_s` (s);
- travel energy: `running_time_path_energy_kwh` exposed through
  `EVRPTWInstance.energy_matrix_kwh` (kWh);
- service and time windows: seconds;
- cargo demand and capacity: cubic centimetres;
- charging power: kW.

The canonical environment must not reconstruct travel time from a scalar speed
or reconstruct running-time-path energy from objective distance.

## State transitions

Customer action:

1. add directed travel distance to the objective;
2. add canonical travel time and subtract canonical path energy;
3. wait until the customer's ready time when early;
4. require service start no later than the inclusive due time;
5. add service time and cargo demand;
6. mark the customer served.

Charging-station action:

1. add directed travel distance, time, and energy;
2. begin charging immediately on arrival;
3. recharge to the battery capacity;
4. add the station-dependent linear full-charge duration

   `t_charge_s = 3600 * energy_added_kWh /
   (charging_power_kW * charging_power_derating_factor)`.

The exported station power has already been capped by the reference vehicle's
AC/DC intake limit during instance generation.  Stations have no time windows,
all vehicles are assumed compatible, port capacity is infinite, and queueing
delay is zero.  Charging time cannot overlap customer waiting or service.

Depot action closes the current vehicle route.  A subsequent vehicle starts at
the working start time with zero load and a full battery.  There is no hard
fleet-size limit and no depot charging duration in the canonical track.

## Feasibility mask

An action is exposed only when:

- the target is reachable with current energy;
- customer demand fits the remaining cargo capacity;
- customer service can start within its hard time window;
- the target transition finishes within the operating horizon; and
- after the target, a directed energy- and time-feasible path exists to the
  depot, through zero or more full-charge station visits when necessary.

This is a safe-continuation check, not a proof that all remaining customers can
be jointly served.  Charging-station revisits are allowed.  Customer revisits
are forbidden after service.

## Reward and evaluation boundary

The environment exposes primitive signals, including distance increment,
served-customer progress, feasibility status, termination, and truncation.
Each method may combine those signals according to its paper-specific reward
and auxiliary shaping.  The method's `ADAPTATION.md` must identify every change
needed to make distance the objective-facing term.

The benchmark ranks only complete solutions that pass the shared verifier.
Its canonical objective is total directed-road distance.  Training reward,
unfinished-rollout reward, and critic value are not cross-method metrics.

## Numerical and reproducibility rules

- Published matrices may be float32; dynamic accumulators and verification use
  float64.
- Time-window and capacity boundaries are inclusive within named tolerances.
- Environment reset and stochastic decoding accept explicit seeds.
- Evaluation freezes the seed set, decoding strategy, and candidate budget.
- Exported routes use node order `[depot, customers, charging stations]` and are
  replayed by the shared verifier before metrics are recorded.
