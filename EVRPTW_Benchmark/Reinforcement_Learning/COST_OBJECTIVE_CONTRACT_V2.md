# Electricity and vehicle-dispatch objective — v2

This document defines the versioned successor to the v1 electricity and
vehicle-dispatch objective. Its sole parameter source is
[`configs/rivian_energy_vehicle_cost_v2.json`](configs/rivian_energy_vehicle_cost_v2.json).
The v1 profile remains immutable so historical checkpoints and results retain
their original meaning. Activating v2 in a run requires a separately versioned
reward contract and run manifest; merely adding this profile does not relabel
an active or historical run.

## 1. Frozen objective

For total directed objective distance `D` in kilometres and dispatched
vehicles `K`, the benchmark objective is

```text
C(D, K) = 413.6331536717643 * K
          + 0.151750972762646 * D                              [USD]
```

The distance coefficient is represented by the existing objective interface as

```text
electricity_price_usd_per_kwh = 0.39
consumption_kwh_per_km        = 100 / 257
distance_unit_cost            = 0.39 * (100 / 257)
                              = 0.151750972762646               [USD/km, rounded]
vehicle_fixed_cost_usd        = 413.6331536717643               [USD/dispatch]
```

Electricity and vehicle costs share one objective and must not be normalized
independently. The coefficients are frozen study inputs for this revision; no
live tariff or mutable external parameter is consulted at runtime.

## 2. Accounting semantics

A valid depot-to-nondepot departure increments `vehicles_started` and incurs
the vehicle fee immediately, including a departure to a charging station.
Returning to the depot incurs distance cost but no second vehicle fee. A later
valid departure is a new vehicle dispatch and incurs another fee.

Every directed travelled edge contributes its objective distance, including
the final feasible depot-return edge. Reset, depot-to-depot no-ops, invalid
actions, padding and post-termination steps do not create vehicle charges.
An unfinished open trip retains the vehicle fee and distance already incurred.

The unshaped task contribution at step `t` is

```text
delta_C_t = 0.151750972762646 * delta_distance_km_t
            + 413.6331536717643
              * indicator(valid depot -> nondepot move)
base_reward_t = -delta_C_t / reward_objective_scale
```

Failure penalties, success incentives and method-specific shaping remain
separate from the reported economic objective. They require a reward contract
calibrated against this v2 objective and must not be reported as dollars.
Battery feasibility continues to use the canonical path-energy matrices; the
distance-cost proxy does not replace those physical constraints. Charging
energy must not be billed a second time on top of the distance term.

## 3. Validation and version boundary

Candidate solutions are replayed through the common resource verifier. Among
verified complete candidates, selection minimizes the v2 cost recomputed from
the route distance and depot departures. Reports retain distance, vehicles,
electricity cost, vehicle cost and total objective as distinct fields.

Checkpoint and resume compatibility requires the complete resolved objective
snapshot. A v1 checkpoint cannot be resumed or warm-started as v2 training
unless the training protocol explicitly defines a weights-only migration.
Historical metrics must remain associated with their original objective; route
rescoring under v2 is a new derived result, not a replacement training result.
