# CLE-EVRPTW Stage-2 instance model

This document is the technical contract for converting a City Logistics
Environment (CLE) into classical, static EVRPTW operating-day instances. The
Generator README is intentionally a quick-start guide; model definitions,
formulas, parameter provenance, assumptions, and release limitations belong
here.

## 1. Scope and claim boundary

The U.S. reference implementation is a **real-geography, public-data-integrated,
semi-synthetic benchmark**:

- CLE roads, directed topology, building locations, facility candidates, and
  charging-site evidence are grounded in public data;
- active customers, packages, service times, time windows, day-level speeds,
  and solver instances are generated from versioned statistical models;
- no generated instance is claimed to be an observed Amazon route;
- Amazon coordinates are not used to place customers in CLE cities.

The V1 problem remains compatible with the classical EVRPTW:

- one depot is selected per instance family;
- the EV fleet is homogeneous, unlimited, and initially fully charged;
- demand and capacity use volume in `cm3`;
- each active service location has at most one time window;
- public charging stations have infinite port capacity;
- charging is linear and must continue to full capacity;
- road travel times are static within an instance;
- the optimization objective is minimum physical travel distance;
- time and energy are feasibility resources, not extra objective terms;
- runtime action masks are computed by environments and are not stored.

The operating horizon is `08:00-24:00` for this profile. It is a configurable
benchmark setting, not a claim that every carrier uses the same shift.

## 2. Inputs, outputs, and execution order

Stage 2 consumes:

1. a portable CLE package;
2. the frozen benchmark contract
   `configs/cle_evrptw_stage2_v1.json`;
3. a versioned operations adapter such as
   `configs/us_reference_instance_profile_v1.json`;
4. Census block-group polygons for the customer split; and
5. aggregate Amazon ARCD calibration statistics produced by
   `scripts/analyze_amazon_arcd_statistics.py`.

The execution order is:

1. freeze complete-community train/held-out customer pools;
2. plan matrix families and lower-scale views;
3. sample day type, depot, customer superset, and nested CS set;
4. realize one static directed road state;
5. compute distance-shortest and turn-aware fastest terminal closures;
6. sample packages, demand, service time, and time windows;
7. reject and replace infeasible order draws without changing a time window;
8. store a compact one-customer feasibility certificate and CS-return cache;
9. write four parent matrices and lower-scale index views; and
10. structurally verify every family and view.

## 3. Split and scale model

### 3.1 Complete-community split

A community is:

```text
community_id = Census block group x directed road SCC
```

Complete communities, rather than individual locations, are assigned to either
the train-location pool or the held-out-location pool. The frozen target is
80/20. A held-out community is marked `training_ineligible=true`, so no
location from that community may appear in training, validation, or Test-1.

The three core tests have distinct meanings:

- Test-1: new seeds, the ten training cities, train-location pool;
- Test-2: the ten training cities, held-out communities only;
- Test-3: Jacksonville, unseen during training.

Cus2000 is a same-city unseen-scale test, not an exact-solver comparison track.
Cus50 is the compatibility and budgeted-MIP track.

### 3.2 Exposure budget and CS counts

Each training scale has five million active-customer exposures:

| Scale | Train views | CS count | Main role |
| --- | ---: | ---: | --- |
| Cus50 | 100,000 | 10 | classical compatibility |
| Cus100 | 50,000 | 20 | core |
| Cus500 | 10,000 | 50 | core |
| Cus1000 | 5,000 | 50 | core parent |
| Cus2000 | 0 | 50 | unseen-scale test |

CS counts are fixed by scale. This makes tensor shape and solver step semantics
stable while still exposing charger-density differences across scales.

## 4. Facility and customer activation

### 4.1 Depot

One release-eligible Tier-A/B depot candidate is sampled for a matrix family.
Tier C is not an instance depot pool. The depot changes across families so a
city is not represented by one fixed start location. Strict candidates receive
sampling weight 2 and optional candidates weight 1. This 2:1 weighting is a
versioned benchmark choice: it favors stronger logistics evidence without
collapsing a city to the small strict-only set. It is not an observed carrier
market share and must be included in sensitivity analysis.

Starting from the selected depot, the customer catchment begins at 40 km and
expands by 10 km, up to 100 km, until it contains at least
`max(N, ceil(1.5 N))` eligible latent locations. These are straight-line
prefilter distances; all released terminal-to-terminal matrices still use the
directed road graph. The thresholds bound the family geographically and are
development sampling parameters, not legal service radii.

### 4.2 Charging stations

The selected depot defines an expandable catchment. CS candidates are greedily
ordered against complete-community reference centroids, before daily customer
IDs are sampled. The first 10, 20, and 50 entries form nested CS sets. This
prevents charger selection from using the exact active-customer realization.

Let `d_c(q)` be the distance from community reference point `c` to candidate
charger `q`, and let `D_c` be its current distance to the nearest already
selected charger. A candidate is chosen by minimizing

```text
sum_c w_c min(D_c, d_c(q))
  + 0.25 * Q90_c[min(D_c, d_c(q))]
  + 0.10 * max_c[min(D_c, d_c(q))].
```

The first term targets weighted mean coverage; the other two discourage a
small set of poorly served communities. The weights are eligible-location
counts, not realized order counts. The 0.25/0.10 coefficients are transparent
development design constants rather than parameters estimated from AFDC.

Each station retains the AFDC charging-mode and power provenance. Effective
power is:

```text
p_effective = min(p_station, p_vehicle_cap(mode))
```

Reported station power is used when available. A missing value uses the
city-by-mode median. Official generation fails if that median is unavailable;
development pilots may use the vehicle mode cap and must record the fallback.

### 4.3 Active service locations

The CLE contains latent physical service locations, not daily orders.
Activation first selects communities and then locations within communities.
Residential-unit evidence affects activation and package multiplicity:

- a house is one ordering unit;
- an apartment location can represent multiple residential units;
- `CusN` always means `N` distinct active physical service locations, not
  `N` packages or households.

For a requested `N`, a target locations-per-community value `T` is sampled from
a clipped lognormal distribution. The current center/spread and 56--205 bounds
come from Amazon route stop-count aggregates, but `T` is used only as a
spatial-diversity prior. The number of selected communities is at least

```text
max(2, ceil(N/T), ceil(log2(N+1))).
```

Communities are sampled without replacement using eligible-location count,
depot-distance decay, and one family-level lognormal activity multiplier. They
are added until they also contain at least `ceil(1.08 N)` eligible locations.
Within those communities, a location with `u_i` residential units receives
base activation weight

```text
a_i(day) = 1 - (1 - p_day)^u_i,
```

multiplied by its community activity. Sampling guarantees at least one
location from each selected community and then fills the remaining positions
without replacement. This makes apartments more likely to be active without
changing the meaning of `CusN`.

Only the route-size descriptive statistics are directly Amazon-derived. The
community activity spread, distance decay, 1.08 buffer, and per-unit
probabilities are development cross-data calibration values. They are
auditable configuration values, not universal parcel rates.

## 5. Static road-state model

### 5.1 Three speed concepts

For every directed physical edge `e`, Stage 1 retains:

- `v_e_legal`: direction-applicable legal speed;
- `v_e_ref`: commercial-vehicle reference speed; and
- H/M/U operating mode:
  - H: motorway and trunk transfer;
  - M: urban transfer and major urban roads;
  - U: residential, service, and delivery-access roads.

OSM supplies the road type and, when present, direction-applicable legal speed.
HPMS `F_SYSTEM` is the preferred functional-class evidence when a successful
conflation exists; OSM `highway=*` is the fallback. H/M/U is the benchmark's
portable crosswalk, not a native NREL or MOVES classification.

The H/M/U reference-speed anchors come from the public NREL Fleet DNA
commercial-vehicle report and are capped by the legal speed. Fleet DNA is used
as a mode-level prior, not described as Rivian edge telemetry
([NREL report](https://doi.org/10.2172/1397153)).

### 5.2 Instance factor

EPA MOVES indexes average-speed distributions by road type, source type, and
day/time strata. V1 uses its road-type/day structure as the transferable U.S.
adapter:

```text
H   -> urban restricted access
M,U -> urban unrestricted access
```

For each matrix family, one factor is drawn for each MOVES road type under the
sampled weekday/weekend class:

```text
v_e(instance) = min(v_e(legal),
                    max(v_min, v_e(ref) * alpha(day, MOVES-road-type)))
```

M and U share the same day-level MOVES factor but retain different
`v_e_ref`, so delivery-access roads remain slower. The model deliberately
does not add unsupported corridor, physical-segment, or direction-specific
random multipliers. Directional asymmetry instead comes from the directed OSM
topology, direction-applicable legal speeds, path choice, and turns.

The MOVES database and algorithms provide the calibration strata
([EPA MOVES algorithms](https://www.epa.gov/moves/moves-algorithms)); the
numerical factor distributions currently in the JSON profile are explicitly
`development_calibration`. Before official release, they must be regenerated
from a frozen MOVES version/query and the resulting parameter table must be
published. NPMRDS can be implemented as an optional observed-speed adapter, but
it is not the default because its directional TMC coverage is licensed and
primarily covers the National Highway System
([FHWA NPMRDS overview](https://ops.fhwa.dot.gov/publications/fhwahop20028/)).

New customer, depot, and CS connectors use the U reference speed in both
directions. Connector symmetry does not change the directionality of the
physical road to which the connector is attached.

## 6. Turn-aware terminal closures

No signal-delay model is used. V1 includes only geometry-derived straight,
right, left, and U-turn penalties from the versioned operations profile.

The distance path minimizes physical distance. Its paired travel-time matrix
evaluates edge travel time and turn penalties on that exact distance-minimizing
path.

The fastest path is exact with respect to the V1 turn model. The directed road
graph is transformed to an edge-state graph. A state is the incoming directed
edge `e`; a transition from `e` to `f` is allowed only when
`head(e)=tail(f)`, with weight

```text
w(e,f) = travel_time(f) + turn_penalty(e,f).
```

Dijkstra is then run on this edge-state graph. Terminal connector and partial
projected-edge costs are added exactly at the source and destination. This
allows a slightly longer route to be selected when it has a lower
turn-inclusive running time.

## 7. Package, demand, service, and time-window models

The primary empirical source is the
[2021 Amazon Last Mile Routing Research Challenge data set](https://doi.org/10.1287/trsc.2022.1173).
It contains real route-, stop-, and package-level records, including package
dimensions, planned service time, time windows, and volumetric vehicle
capacity. Its stop coordinates are obfuscated, and it does not label stops as
houses or apartments. Consequently:

- its coordinates are not transferred to CLE cities;
- aggregate distributions are transferable priors;
- house/apartment conditioning is a disclosed cross-data model using CLE
  residential-unit evidence; and
- the benchmark does not claim observed building-type-specific parcel rates.

### 7.1 Package count and volume

For active location `i` with `u_i` residential units:

1. sample ordering units conditional on `u_i` and weekday/weekend;
2. condition on at least one ordering unit because the location is active;
3. sample extra parcels with a negative-binomial model; and
4. sample parcel volumes from a truncated lognormal distribution and sum them.

```text
demand_i = sum(volume_ij),  j=1,...,package_count_i.
```

Demand and vehicle capacity are both stored in `cm3`.

### 7.2 Service time

The stop-level conditional model is

```text
s_i = clip((beta_0
            + beta_pkg * package_count_i
            + beta_vol * demand_i) * lognormal_noise,
           s_min, s_max).
```

Amazon defines planned service time at package level, so the calibration script
sums package-level values at each stop before fitting stop-level summaries. The
package-count and volume terms make service time increase with the amount
delivered at one physical location.

### 7.3 One time window per active location

The instance first draws a day-level TW-presence probability from a
weekday/weekend beta distribution. Each active location then receives either
the full operating horizon or one sampled `strain`/`loose` interval. Window
width and center are sampled from the versioned profile.

A time window is clipped only to the declared `08:00-24:00` model support. It
is never moved, widened, or shortened using travel-time feasibility.

### 7.4 Sample-then-validate policy

Packages, demand, service time, and TW are sampled first. A constructive
one-customer certificate then checks volume, energy, TW, and return before the
horizon end. If an order draw fails, the order attributes are replaced for the
same physical location. Rejection attempts and reason counts are stored.

If a location is structurally impossible even with minimum service time, the
matrix-family attempt fails and the outer deterministic retry selects a new
family realization. The implementation never edits a sampled TW to force
acceptance.

The current bound of 64 order draws is an algorithmic fail-safe, not an
industry parameter. Exhaustion is an error, never silent truncation.

## 8. Vehicle and energy model

### 8.1 Reference vehicle

V1 freezes the Rivian Commercial Van Delivery 700. Rivian's 2025 reference
guide reports:

- cargo volume: 18.5 m3 = 18,500,000 cm3;
- EPA-estimated range: 160 mi = 257 km;
- battery pack: 100 kWh LFP;
- AC charging rate: 11 kW; and
- DC charging rate: up to 100 kW.

See the
[Rivian Commercial Van Reference Guide](https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf).
The vehicle is a reference configuration; the benchmark does not claim that
every Amazon delivery route uses the same trim or state of battery health.

### 8.2 Constant distance consumption

The classical V1 energy contract is:

```text
h = 100 kWh / 257 km
  = 0.3891050584 kWh/km

energy(P) = h * distance(P).
```

Travel speed, waiting, turn time, and auxiliary load do not change V1 energy.
This keeps the resource model linear and compatible with classical EVRPTW
exact and heuristic baselines. Constant per-distance consumption is a common
EVRPTW abstraction; see the original EVRPTW formulation by
[Schneider, Stenger, and Goeke](https://doi.org/10.1287/trsc.2013.0490) and a
later comparison of charging formulations that explicitly uses a constant
consumption rate
([Operational Research article](https://link.springer.com/article/10.1007/s12351-023-00806-5)).

This is a modeling simplification, not a claim that real EDV consumption is
speed-independent. Weather, payload, elevation, HVAC, and driving behavior can
be added by another energy adapter in a future benchmark track.

## 9. Charging and feasibility cache

At a CS with effective power `p_q`, V1 full linear charging from current
energy `b` takes

```text
charge_time(q,b) = (battery_capacity - b) / (eta * p_q) * 3600.
```

The reference profile uses `eta=1`. Therefore AFDC/vehicle-capped kW is treated
as effective battery-side charging power. This convention avoids claiming an
unobserved charger/vehicle efficiency; alternative profiles may provide an
explicit efficiency and must document whether station power is input-side or
battery-side.

For mask acceleration, each view stores
`full_cs_to_depot_time_s[q]`: depart CS `q` full and reach the depot by the
fastest energy-feasible sequence, including intermediate CS travel and
charging, excluding charging at the origin and depot. Reverse Dijkstra on the
full-state depot/CS graph permits multiple CS hops.

Energy is derived from the fastest-path distance matrix and `h`; the cache
does not use distance-shortest-path energy.

## 10. Four stored matrices

Each parent family persists exactly four `float32` matrices:

1. `distance_matrix_km`: physical distance of the distance-shortest path;
2. `distance_path_travel_time_s`: turn-inclusive time on that path;
3. `running_time_shortest_matrix_s`: exact turn-aware fastest time; and
4. `running_time_path_distance_km`: physical distance of the fastest path.

The loader derives:

```text
distance_path_energy_kwh
    = distance_matrix_km * h

running_time_path_energy_kwh
    = running_time_path_distance_km * h.
```

The consumer API still exposes both energy arrays, but they are not duplicated
on disk. Four matrices are necessary because the distance-minimizing and
time-minimizing paths can differ, while time and energy feasibility must match
the path being evaluated.

## 11. Current numerical profile audit

The JSON profile is the executable source of truth. This table makes its main
values and evidence level readable without implying that every development
coefficient is already publication-ready.

| Submodel | Current V1 value | Source/evidence level | Why it is present |
| --- | --- | --- | --- |
| Day type | weekday:weekend = 5:2 | calendar-based benchmark mixture | separates the two frozen operating-day profiles without time-of-day traffic |
| MOVES restricted-access factor | weekday mean/std 0.96/0.035; weekend 0.98/0.030 | MOVES stratum; numerical fit still development | one transferable day-level factor for H roads |
| MOVES unrestricted-access factor | weekday 0.92/0.050; weekend 0.95/0.045 | MOVES stratum; numerical fit still development | shared factor for M/U while their reference speeds remain different |
| Minimum road speed | 5 km/h | numerical safeguard, not empirical claim | prevents zero/infinite edge time |
| Connector speed | 36.403361 km/h | NREL U-mode prior | treats private access as delivery-access travel |
| Turn classes | straight <=30 degrees; U-turn >=150 degrees | geometry convention | deterministic portable angle classification |
| Turn time | right 3 s; left 8 s; U-turn 20 s | development benchmark constants | adds asymmetric maneuver cost without unavailable signal timing |
| Depot evidence weight | strict:optional = 2:1 | development sampling rule | favors strong logistics evidence while retaining depot diversity |
| Depot catchment | start 40 km, step 10 km, maximum 100 km | development sampling rule | expands only when the eligible pool is insufficient |
| Catchment pool buffer | 1.5 x requested locations | development sampling rule | leaves a nontrivial pool for stochastic activation |
| Community target | lognormal median 141.6661, sigma 0.1835, bounds 56--205 | Amazon route stop-count descriptive prior | controls spatial spread, not order total |
| Community capacity/activity | buffer 1.08; lognormal sigma 0.85; distance decay 30 km | development spatial sampling rule | avoids a single dense community dominating every family |
| Per-unit order probability | weekday 0.028; weekend 0.022 | development cross-data calibration | links CLE unit evidence to daily active orders |
| Charger coverage score | weighted mean + 0.25 p90 + 0.10 maximum | development facility-sampling rule | balances average and tail community coverage without using daily IDs |
| Extra-package model | mean 0.62194, dispersion 0.327 | Amazon-aggregate development fit | permits multiple parcels per ordering unit |
| Package volume | lognormal median 7,000 cm3, sigma 1.0, cap 300,000 cm3 | Amazon package dimensions; development fit | produces positive heavy-tailed volumetric demand |
| Service time | base 28.1626 s, 46.9063 s/package, 0.000358429 s/cm3 | Amazon planned-service aggregates; development fit | makes service depend on both count and delivered volume |
| Service noise/bounds | lognormal sigma 0.75; 5-8,007 s | development residual/bound table | preserves heterogeneity and rejects pathological tails |
| TW occurrence | weekday Beta(6.1,60.7); weekend Beta(5.9,58.5) | Amazon aggregate development fit | varies the tight-window share by operating day |
| TW interval profiles | `strain` and `loose` center/width distributions in JSON | Amazon aggregate development fit | supports one practical interval or the full horizon |
| Order retry bound | 64 | algorithmic fail-safe | fails visibly if acceptance sampling cannot finish |
| Battery/range | 100 kWh / 257 km | Rivian official guide | defines the reference battery and linear coefficient |
| Cargo volume | 18,500,000 cm3 | Rivian official guide | keeps demand and capacity in the same volume unit |
| AC/DC caps | 11/100 kW | Rivian official guide | caps AFDC station power by the vehicle |
| Charging efficiency | 1.0 effective-rate convention | explicit V1 assumption | avoids inventing an unobserved loss factor |

The Amazon-related values must be accompanied by the generated
`amazon_arcd_training_statistics_v1.json`, source hashes, fitting notebook or
script, and sensitivity table before the profile can become
`release_calibrated`. The turn and road-factor values need the same treatment:
either publish a supporting calibration or retain them as clearly labeled
configurable benchmark assumptions.

## 12. Parameter provenance and release state

| Parameter group | Source role | Current state |
| --- | --- | --- |
| Road topology/legal speed | OSM; optional HPMS | public-data grounded |
| H/M/U reference speed | NREL Fleet DNA mode prior | built-in adapter |
| Weekday/weekend road factor structure | EPA MOVES strata | structure grounded; numerical distributions still development |
| Package volume, service, TW summaries | Amazon ARCD aggregates | development calibration; frozen artifact required |
| House/apartment conditioning | NSI units + CLE type + Amazon aggregates | disclosed semi-synthetic cross-data model |
| Battery, range, cargo volume, AC/DC caps | Rivian official guide | frozen reference specification |
| Energy consumption | battery/range ratio plus classical EVRPTW linear model | frozen V1 assumption |
| Turn penalties | geometry-only configurable adapter | development parameter table |
| Full-charge policy, infinite ports/fleet | benchmark contract | frozen V1 assumption |

No profile may be marked `release_calibrated` until its source snapshot,
analysis command, fitted table, sensitivity report, and citation list are
published. Structural test success alone is not scientific validation.

## 13. Primary references

- Merchán et al. (2022), *2021 Amazon Last Mile Routing Research Challenge:
  Data Set*, Transportation Science:
  <https://doi.org/10.1287/trsc.2022.1173>
- Rivian (2025), *Commercial Van 500/700 Reference Guide*:
  <https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf>
- U.S. EPA, MOVES Algorithms:
  <https://www.epa.gov/moves/moves-algorithms>
- NREL Fleet DNA:
  <https://www.nrel.gov/transportation/fleettest-fleet-dna.html>
- Konan et al. (2017), NREL/TP-5400-65921:
  <https://doi.org/10.2172/1397153>
- Schneider, Stenger, and Goeke (2014), *The Electric Vehicle-Routing Problem
  with Time Windows and Recharging Stations*:
  <https://doi.org/10.1287/trsc.2013.0490>
- FHWA, NPMRDS overview:
  <https://ops.fhwa.dot.gov/publications/fhwahop20028/>
