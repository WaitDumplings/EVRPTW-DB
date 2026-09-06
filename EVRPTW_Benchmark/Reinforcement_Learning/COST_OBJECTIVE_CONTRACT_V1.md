# Electricity and vehicle-dispatch objective — v1

This is the active objective for fresh AM-EVRPTW, EVRPTW-RL, DRL-TS and TERRAN
training in the formal RQ queues. It supersedes the distance-only objective for
these new runs, not the historical results or the physical dataset contract.
The sole parameter source is
[`configs/rivian_energy_vehicle_cost_v1.json`](configs/rivian_energy_vehicle_cost_v1.json).
The subsequent independent action revision
[`drl_no_consecutive_cs_v1`](ACTION_CONSTRAINT_CONTRACT_V1.md) forbids CS-to-CS
visits in all four DRL methods without changing these economic coefficients.

## 1. Objective and reference parameters

For total directed objective distance `D` (km) and dispatched vehicles `K`:

```text
C(D, K) = p_e * rho * D + c_vehicle * K                         [USD]
p_e     = 0.1341                                               [USD/kWh]
rho     = 100 / 257                                            [kWh/km]
c_vehicle = 83,900 / (10 * 250) = 33.56                         [USD/vehicle-day]
```

All four methods use identical coefficients, including all scales, support
conditions and Euclidean-training controls. The JSON records source links and
accounting assumptions. The electricity price is a frozen reference price,
not a live tariff; the capital allocation assumes ten years, 250 operating days
per year, zero residual value and no discounting. This is electricity plus
allocated purchase cost, not a full fleet total-cost-of-ownership estimate.
Driver wages, maintenance, insurance and infrastructure are not added.

The electricity term prices distance-proportional consumption. Initial battery
energy and return-to-depot travel are therefore not free. Do **not** add another
bill for energy purchased during charging visits: that would double count it.
Charging powers and durations remain part of the unchanged physical model.
In particular, battery feasibility still uses the stored canonical path-energy
matrix; it must not be replaced by `rho * objective_distance` during this update.
The cost term is the registered distance-based consumption proxy, not a claim
that the two path-specific matrix families are identical.

## 2. Vehicle dispatch and per-step reward

The canonical fleet has no fixed upper limit and permits one trip per vehicle.
A valid move from depot to a non-depot terminal increments `vehicles_started`
and charges `c_vehicle` immediately, even if that terminal is a charging station.
Returning to the depot incurs its travel electricity but no second vehicle fee.
The next valid departure uses a new vehicle and incurs another fixed fee.

No fee is charged by reset, depot-to-depot no-ops, invalid actions, padding, or
steps taken after termination. An unfinished open trip has already used a
vehicle, so its fixed fee remains in the incurred-cost ledger. A paid
charging-only trip cannot be silently dropped from cost-route export.

```text
delta_C_t = (p_e * rho) * delta_distance_km_t
            + c_vehicle * indicator(valid depot -> nondepot move)
base_reward_t = -delta_C_t / reward_objective_scale
```

The unshaped, undiscounted sum equals negative incurred total cost divided by
the scale. Existing invalid-action/unfinished penalties and method-specific
auxiliary shaping remain separate and retain their intended semantics; they
are not reported as benchmark objective values. AM's incomplete-penalty
parameter remains in km-equivalent units and is converted by `p_e * rho`
before entering the normalized cost loss. Its historical `cost_km` diagnostic
is never relabelled as dollars.

TERRAN uses `gamma=1.0` in both returns and PBRS under
`terran_undiscounted_energy_vehicle_pbrs_v1`. Encoder/decoder architectures,
native training-stage boundaries and optimizer schedules are unchanged. PPO
encoder caches retain the update-boundary invalidation introduced previously.

## 3. Reward normalization, not an extra economic weight

Each method retains its existing training-pool distance-scale procedure. The
selected reference workload is converted to cost units as a whole:

```text
reward_objective_scale = (p_e * rho) * reward_distance_scale_km
                         + c_vehicle * reference_dispatches
reference_dispatches = N for a single_customer_repair_sum scale,
                       1 for a mean/median or other single-reference scale
```

The `dataset_` prefix denotes the existing shared training-pool scale. This
positive scalar multiplies the entire objective; it does not change the
vehicle-to-distance cost ratio. Normalizers are not fit on test results. The
legacy distance track retains its previous distance normalization.

## 4. Validation, checkpoint selection and test export

For each instance, independently replay complete candidates through the common
resource verifier and select the verified candidate with minimum total cost.
Cost ordering is recomputed from exported routes and the instance's directed
distance matrix, including every depot departure; it does not trust a stale
distance-only ranking or a closed-route counter. If the environment did not
complete, a virtually appended depot in route export cannot turn the candidate
into a successful cost-track evaluation.

Checkpoint selection is lexicographic: highest verified feasibility rate,
then lowest mean verified total cost on the fixed validation cohort. Candidate
count, seeds, validation schedule and test independence remain unchanged.
Distance is a diagnostic, not the secondary checkpoint-selection criterion.

The environment, validation reports and evaluation exports distinguish:

- `objective_distance_km`: raw distance;
- `vehicles_started`: charged departures, including an open trip when present;
- `electricity_cost_usd` and `vehicle_cost_usd`: the two cost components;
- `objective_cost_usd` / `objective_value` with `objective_unit=USD`;
- verifier feasibility and served-customer counts.

Validation aggregates cost and distance over verified complete instances only.
The old `vehicle_count` field is retained for compatibility and must not replace
`vehicles_started` in the new cost formula. Checkpoints store the resolved
objective dictionary, not merely a path to a mutable config file. Evaluators
load that snapshot and reject conflicting explicit overrides. A checkpoint
without objective metadata is treated as historical distance-only, not cost.

## 5. Fresh-run and comparison boundaries

All 24 formal jobs in the four server bundles now carry the same objective
snapshot and config path. Launch preflight checks those fields against the
versioned profile. Job provenance, completion checks and same-run resume guards
include the objective; TERRAN additionally includes its gamma/reward contract.
Old distance checkpoints cannot resume as cost checkpoints. Fresh launches
refuse existing training history. A new Git commit yields a new run directory.

Dataset files, frozen ID streams, exposure budgets and queue assignments are
unchanged. Use `scripts/rq_v1/<server>/full.sh` for the new all-method run after
stopping any older queue. Use `resume.sh` only for interruption recovery within
this objective revision and the same commit. Do not import previous checkpoints.

Historical Gurobi/ALNS/VNS-TS and DRL distance-optimized solutions remain
distance-optimized evidence. Their stored routes may be explicitly rescored
under this formula, but rescoring is **not** a cost-optimized solver run and
must not silently overwrite the old metrics or be described as such.
