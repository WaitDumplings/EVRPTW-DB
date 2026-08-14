# CLE to EVRPTW instance generation contract

This document is the authoritative scientific description of Stage 2. It
explains exactly how a portable City Logistics Environment (CLE) becomes a
classical, static EVRPTW instance and which outputs are retained for audit.
The JSON configuration files are the executable source of truth; this document
states their meaning and the claim boundary.

## 1. Benchmark scope

The U.S. reference corpus is a **real-geography, public-data-integrated,
semi-synthetic benchmark**. A CLE fixes the city boundary, directed road graph,
latent residential service locations, depot candidates, public charging sites,
and edge speed evidence. Stage 2 decides which of those latent objects are
active on one operating day and attaches observed Amazon order templates.

The released instance is not claimed to be an observed Amazon route or a
reconstruction of an Amazon service area. Amazon coordinates are not transferred
to the CLE. Amazon data supplies empirical depot-day route structure and
package/service/time-window templates; CLE geography supplies the physical
locations and directed travel network.

The canonical problem track retains classical EVRPTW assumptions:

- one depot per instance family;
- a homogeneous, initially full, unlimited EV fleet;
- demand and capacity in cubic centimetres;
- one time window per active physical service location;
- static travel time within an instance;
- real public charging sites with fixed effective power and unlimited ports;
- linear charging to full capacity;
- linear distance-based energy consumption; and
- minimum physical travel distance as the objective.

The operating horizon is 08:00--24:00 in the U.S. profile. It is configurable
and is not presented as a universal carrier shift.

## 2. Required inputs and produced artifacts

Stage 2 consumes:

1. eleven portable CLE packages produced by Stage 1;
2. Census block-group polygons for the customer-pool split;
3. Amazon Last Mile Routing Research Challenge 2021 `route_data.json`,
   `package_data.json`, and `travel_times.json` from `model_build_inputs`;
4. `configs/cle_evrptw_stage2_v1.json`, which freezes benchmark sizes, splits,
   and view relationships; and
5. `configs/us_reference_instance_profile_v1.json`, which freezes the U.S.
   vehicle, charging, connector, and feasibility adapter.

The production runner writes:

```text
EVRPTW_Dataset/Instances_v1/us_11city/
  amazon_artifacts.json
  customer_splits/<city>/
    customer_split_manifest.parquet
    community_manifest.parquet
    community_adjacency.parquet
    customer_split_report.json
  generation_plan/
    families.parquet
    views.parquet
    split_registry.json
  materialized/families/<family_id>/
    family_manifest.json
    terminal_index.parquet
    phase1_metrics.json
    phase1_observations.parquet
    phase1_region_pair_metrics.parquet
    matrices/*.npy
    views/<view_id>/*
  rejections/*.json
  reports/phase1/
    family_metrics.parquet
    stratified_metrics.csv
    rejected_attempts.parquet       # only when rejections occurred
    summary.json
```

The raw Amazon files are preprocessed once into compact Parquet artifacts.
Raw customer coordinates are never copied into the generated dataset.

## 3. Stage-2 execution order

The implementation performs the following steps in order:

1. preprocess Amazon depot-day structure and order templates;
2. freeze location pools and the directed community-adjacency graph;
3. plan parent matrix families and their lower-scale views;
4. select a physical depot access point;
5. construct a depot-centred feasible territory;
6. activate exactly the parent number of customer locations;
7. select the scale-fixed real charging-station set;
8. realize one static weekday or weekend road state;
9. compute the four parent terminal matrices;
10. attach distinct feasible Amazon order templates;
11. build lower-scale views, charging caches, and feasibility certificates;
12. verify the family and persist Phase-1 diagnostics; and
13. aggregate diagnostics across the completed corpus.

Selection is deterministic conditional on the recorded seeds. A failed family
attempt is recorded and retried with a deterministic attempt seed; it is never
silently modified after acceptance.

## 4. Amazon preprocessing

### 4.1 Station-day units

Routes are grouped by station code, calendar date, and weekday/weekend class.
Two empirical roles remain separate:

- **spatial-structure support** uses route membership and inter-stop travel
  times to describe how stops are distributed within routes; and
- **order support** uses package volume, planned service time, and time-window
  fields to create stop-level delivery templates.

This separation prevents a stop that is usable for route structure but has
incomplete order attributes from being treated as a complete order template.

### 4.2 Spatial envelope and route structure

For each station-day, the preprocessing records route stop counts and travel-
time distributions. The territory envelope `T_env` is the station-day tail
quantile specified by the frozen contract. It is used as empirical reach in
network travel time; it is not interpreted as a radius in kilometres.

Within a chosen structure source, stops are stratified by source route and by
decile of depot-to-stop travel time. The generated city must preserve both
route totals and decile totals after deterministic controlled rounding.

If one real station-day supports the requested parent size, it is used directly
and labelled `SINGLE_STRUCTURE_DAY`. If not, multiple days from the same Amazon
station and day type are combined until they support the requested scale and
the source is labelled `SAME_STATION_STRUCTURE_COMPOSITE`. Cross-station
composition is not allowed.

### 4.3 Order templates

One observed stop becomes one order template containing:

- package count;
- summed package volume in `cm3`;
- summed planned service time; and
- either one observed time window or the full operating horizon.

Package dimensions are converted to a common volume unit before summation.
Templates with required missing or invalid fields are excluded and their
attrition is reported. A source is labelled `SINGLE_ORDER_DAY` or
`SAME_STATION_ORDER_COMPOSITE`; the latter combines days only within one
station and one day type. Templates are never duplicated to reach a larger N.

## 5. Customer pools and community graph

### 5.1 Community definition

The geographic unit is:

```text
community_id = Census Block Group x directed-road strongly connected component
```

A Census block group contributes a public, stable residential geography. The
road-SCC suffix prevents locations that fall in the same Census polygon but
cannot share the same directed operational road component from being treated
as one routing community.

### 5.2 Community adjacency

`community_adjacency.parquet` is built from actual directed OSM edges that
cross between community-labelled graph nodes. Its cost is derived from the CLE
edge travel time. Transit-only road communities with no latent customers are
retained because they can connect two customer communities. This graph is used
for spatial expansion; straight-line nearest polygons are not treated as road
neighbours.

### 5.3 Leakage-free split

Complete communities, rather than individual buildings, are assigned to the
train-location pool or held-out-location pool. The target is 80/20. A held-out
community is unavailable to train, validation, and Test-1.

- Test-1: new seeds in the ten training cities, train-location pool;
- Test-2: held-out communities in the ten training cities;
- Test-3: Jacksonville, unseen during training; and
- Cus2000: same-city unseen-scale evaluation.

## 6. Family and view plan

The plan allocates work before physical selection. Day types are assigned per
city and cohort by deterministic largest-remainder allocation of the fixed 5:2
weekday/weekend mixture, followed by a seeded shuffle. A slot's day type never
changes during retry.

The core training exposure budget is five million activated customers per
scale:

| Scale | Train views | CS | Purpose |
| --- | ---: | ---: | --- |
| Cus50 | 100,000 | 10 | compatibility and budgeted MIP |
| Cus100 | 50,000 | 20 | core benchmark |
| Cus500 | 10,000 | 50 | core benchmark |
| Cus1000 | 5,000 | 50 | core parent scale |
| Cus2000 | 0 | 50 | unseen-scale evaluation |

Cus1000 parent families are partitioned into exact, disjoint lower-scale views.
Twenty Cus50 children form the parent; fixed groupings of these leaves form
Cus100 and Cus500. The union of leaves must equal the parent and siblings must
be disjoint. A Cus2000 family additionally stores a deterministic Cus1000
control view to separate city/seed effects from the scale change.

## 7. Depot selection

Stage 1 groups OSM access points that refer to the same physical logistics
facility. Grouping uses containment by a logistics/industrial land-use polygon,
or matching normalized name/operator together with matching address or touching
geometry. A facility that has no supported match remains a singleton; proximity
alone does not merge two depots.

For each family:

1. sample one eligible physical facility group uniformly;
2. select a Tier-A access point within that group when available; otherwise
   select an eligible Tier-B access point; and
3. record the evidence tier and group identifier.

Building area is retained as evidence but is not a hard `1000 m2` gate and is
not used to impose an unexplained area threshold. Tier C retail or parcel-shop
locations are not in the canonical depot pool.

## 8. Feasible territory

The selected depot is routed once to all road nodes in the selected day-type
reference state. A latent customer enters the candidate territory only if:

- it belongs to the family split pool;
- its directed road connector is valid;
- its depot travel time lies within the Amazon-derived `T_env`; and
- depot-to-customer-to-depot path energy is no greater than one full battery.

The last condition is deliberately sufficient rather than necessary. It
guarantees every activated customer has an individual direct-roundtrip energy
certificate. Charging can still be required in a multi-customer route. The
territory must contain at least N candidates; otherwise the family attempt is
rejected before spatial activation. No 40/50/.../100 km ladder or `1.5N`
straight-line pool rule is used.

## 9. Step 6: spatial customer activation

### 9.1 Target quota table

For requested parent size N, source route-by-decile counts are proportionally
downscaled only when their total exceeds N. Deterministic controlled rounding
then produces an integer table `Q[r,b]` such that:

```text
sum_b Q[r,b] = rounded route total r
sum_r Q[r,b] = rounded decile total b
sum_r,b Q[r,b] = N.
```

This is a two-margin transportation problem, not independent cell rounding.
Rows that round to zero are reported as dropped; margins and total are hard
correctness gates.

### 9.2 Region seeds

Each positive source-route row corresponds to one generated delivery region.
The first seed community is selected deterministically from an admissible
travel-time decile. Subsequent seeds maximize their minimum **symmetrized
network travel time** to existing seeds, subject to the row's available decile
quota. This discourages all generated regions from collapsing into one part of
the city.

If the exact decile lacks capacity, fallback is ordered to the immediately
nearer decile before the immediately farther decile, and the event is recorded.
The implementation does not silently search arbitrary distance bands.

### 9.3 Community growth and customer assignment

Regions grow in round-robin order on the directed community-adjacency graph.
At each step, the region requests capacity in its remaining route-decile cells.
Eligible adjacent communities are ranked deterministically by usable capacity,
decile compatibility, and seed. A community may support competing regions, but
one customer ID can be assigned only once globally.

Final customer assignment is a global min-cost flow over requested
region-decile cells and eligible customer IDs. If competition makes the first
candidate set infeasible, the affected regions expand to the next road-adjacent
communities and retry. Every expansion is counted. A completed family must
contain exactly N unique customer IDs and must satisfy every quota cell.

The full region construction has a bounded deterministic redraw count. A redraw
changes the family attempt seed and is logged; it does not edit an accepted
sample.

### 9.4 Radial control baseline

For every successful family, the generator also creates a size-matched radial
baseline from the same feasible territory. It uses the target depot-time
distribution but does not use community-contiguous growth. The baseline is not
released as a benchmark instance; it is retained to reveal selection effects in
the Phase-1 evaluation.

## 10. Charging-station selection

Charging stations are selected after the active customer geography is known.
The candidate pool contains real, compatible AFDC sites from the CLE; no
synthetic station is inserted. Fixed counts by scale are part of the benchmark
contract.

Selection has two roles:

1. a feasibility core would include stations required by the energy
   reachability certificate; under the current conservative direct-roundtrip
   customer screen this core is normally empty; and
2. remaining positions cover active customer communities, the depot-to-region
   corridors, and poorly covered parts of the activated territory.

This means stations are geographically relevant to the realized delivery area,
not sampled before the customer territory. The selection report states core
size, fill composition, candidate count, power provenance, and any fallback.
The environment, not the dataset, controls repeated CS visits during a route.

Effective charging power is:

```text
p_effective(q) = min(p_reported_or_city_mode_median(q), vehicle_mode_cap).
```

The canonical adapter accepts compatible DC and Level-2 sites. Missing station
power uses the city-by-mode median; all connectors and terminal access remain
part of the directed matrices.

## 11. Static road state and matrices

Stage 1 stores weekday and weekend reference speed for every directed physical
edge. Stage 2 chooses the column fixed by the family day type. The canonical
profile does not invent a second uncalibrated edge-randomness multiplier.
Directional asymmetry comes from OSM one-way topology, direction-applicable
legal speed, different forward/reverse paths, and turn penalties.

Customer/depot/CS connectors are bidirectional and use the city/day physical-
length-weighted median speed of delivery-access edges. The physical road itself
retains its original direction.

The parent stores four `float32` matrices:

1. shortest-distance path distance;
2. turn-inclusive travel time on the shortest-distance path;
3. exact turn-aware fastest travel time; and
4. physical distance of that fastest-time path.

Right, left, straight, and U-turn classes are derived from edge geometry. No
signal-delay model is included. Fastest-time routing uses an edge-state graph,
so a turn penalty belongs to the actual incoming/outgoing edge pair rather than
being added after shortest-path computation.

## 12. Order-template attachment

After terminal matrices exist, each parent customer is matched to one distinct
Amazon order template. A customer-template edge is admissible only when:

- template demand does not exceed vehicle volume capacity;
- the earliest service start satisfies the observed time window; and
- service plus the precomputed energy-feasible return duration finishes within
  the operating horizon.

Maximum bipartite matching must cover all customers. If a source fails Hall
coverage, the attempt is recorded and the next admissible single-day or same-
station composite source is tried. Templates are not reused, and a time window
is never shifted or widened to force feasibility. Every customer row stores
`order_template_id`, `order_station_day_id`, and `order_source_mode`. Child
views inherit the corresponding parent templates.

## 13. Vehicle, energy, and charging cache

The U.S. reference vehicle is the Rivian Commercial Van Delivery 700 profile:

- battery capacity: 100 kWh;
- cargo volume: 18.5 m3 = 18,500,000 cm3;
- reference range: 257 km;
- AC cap: 11 kW; and
- DC cap: 100 kW.

The classical energy model is linear:

```text
h = 100 / 257 kWh/km
energy(path) = h * path_distance.
```

Full charging from state `b` at site `q` takes:

```text
charge_time(q,b) = (battery_capacity - b) / (eta * p_effective(q)) * 3600.
```

For runtime feasibility checks, every view stores the fastest feasible time
from each CS, departing full, to the depot. Reverse shortest-path computation
over depot/CS full-battery states permits multiple CS hops and includes travel
and intermediate full-charge time. This is static instance data, not a stored
action mask.

## 14. Phase-1 evaluation and persisted metrics

The purpose of Phase 1 is to evaluate whether the spatial activation mechanism
does what its contract says before expensive benchmark generation is accepted.
Correctness gates and realism diagnostics are kept separate.

### 14.1 Hard correctness gates

Every family must pass:

- exactly N parent customers;
- globally unique customer IDs;
- a declared split pool;
- exact route margins and decile margins;
- exact, pairwise-disjoint nested-view partitions; and
- exact child sizes.

Failure rejects the family; it is not averaged away at corpus level.

### 14.2 Spatial diagnostics

The generator records:

- **M1 radial agreement:** normalized Wasserstein-1 distance between generated
  depot times and the Amazon structure target, compared with the radial
  baseline;
- **M2 nearest-neighbour time:** generated network nearest-neighbour
  distribution versus the Amazon reference distribution;
- **M3 within-region compactness:** generated within-region pairwise network
  time P50/P90 versus source-route references;
- **M4 region structure:** region count, region-size distribution, and routes
  dropped only by controlled rounding; and
- **M5 community concentration:** active-community count, largest-community
  share, and HHI, always shown beside the radial baseline.

M1 is the primary comparative candidate. M2, M3, and M5 are report-only in the
current pilot because cross-city topology and the obfuscated Amazon geometry do
not justify freezing universal numeric thresholds before seeing the eleven-city
results.

### 14.3 Reliability and selection-bias diagnostics

Each family additionally stores territory reserve ratio, energy-screen removal
share, decile fallback count, region redraw count, community-growth steps, and
assignment-competition expansions. Corpus aggregation reports:

- first-attempt success rate by planned family slot;
- conditional attempt success rate;
- rejection error-type counts;
- distributions of redraw and fallback behaviour; and
- all metrics stratified by city, weekday/weekend, and parent scale.

The statistical unit is an attempted parent family. Reporting the rejected
attempts prevents survivorship bias from disappearing behind successful output.
No pilot-derived numeric threshold is silently encoded as a scientific gate.

## 15. Verification and claim limits

The family verifier checks matrices, dimensions, finite values, asymmetry,
energy derivation, charging caches, per-customer feasibility certificates,
Amazon template provenance, metric-file row counts, and all Phase-1 hard gates.

The benchmark supports the claim that the generator composes public city data
and observed delivery templates through a deterministic, auditable mechanism.
It does not support claims that generated instances reproduce an individual
carrier's service territory, fleet dispatch, traffic trace, or proprietary
customer coordinates. Those boundaries are part of the method, not hidden
limitations.
