# EVRP-TW-Hierarchy-D Calibration Guideline

This document explains how to calibrate the generator to a real last-mile dataset such as Amazon Last Mile Routing, while preserving a defensible EVRP-TW-D benchmark construction.

The core interpretation is:

```text
mother board = stable city / region / delivery-station service territory
instance     = one active operating day sampled from that territory
```

The generator should use input-side statistics only. Driver sequences, realized route duration, route cost, and route quality are solution-side outcomes and should be reserved for validation, not generation rules.

## 1. Algorithm Self-Audit

### 1.1 Road Network

The current road graph is synthetic but road-network-first. Depot, community gateways, local streets, customers, and charging stations are all graph terminals or attached graph vertices. Distances are computed by road shortest paths, not direct Euclidean distances.

This is reasonable for a benchmark generator because it avoids claiming access to exact private road maps while still producing graph-dependent travel distances. The key assumption is that Amazon-like service territories can be represented by a calibrated embedded sparse road graph.

Defensible points:

- The graph is connected by construction and validated from the depot.
- Cluster gateways form a corridor/infill service-sector pattern, matching station service territories better than a full radial ring.
- Road edge length uses a stretch factor over Euclidean length to approximate road-network detours.
- The generator saves `distance_matrix_km`, which is the EVRP objective and energy basis.

Known boundary:

- This is not a real GIS road network. If exact roads are available, they can replace the synthetic graph while keeping the same mother-board / active-day architecture.

### 1.2 Customer Locations

Customers are generated as a latent address pool. The current default uses `road_oriented_community` placement:

1. sample macro community centers from Amazon-calibrated depot-stop radii and service corridors;
2. generate local street vertices around each community gateway;
3. sample micro-zone centers inside road-oriented community ellipses;
4. sample customer stops around each micro-zone with larger spread along the local road direction and smaller perpendicular spread;
5. attach every customer to the nearest local street node.

This fixes two failure modes:

- isotropic Gaussian blobs looked too clustered;
- exact street-segment sampling looked too line-like.

The current default is an A/B blend tuned against Amazon spatial statistics. It targets the Amazon depot-stop distance distribution while keeping community spread between the compact and wide alternatives.

Known boundary:

- At very small scales, a purely cluster-based daily sampler can collapse into one community. The current sampler therefore applies a sparse-day active-cluster lower bound and ensures selected cluster coverage.

### 1.3 Daily Customer Activation

Daily activation is not uniform over all latent customers. It first samples active clusters, then samples customers using cluster-level and micro-zone-level lognormal activity weights.

The active cluster count should be calibrated from station-day community-demand
scale, not set as a purely manual constant. Amazon does not directly label our
synthetic communities, so we use historical route size only as an input-side
proxy for community demand. In the Amazon training data aggregated by
`date x station`, the mean number of unique dropoff customers per route has:

| metric | value |
|---|---:|
| min | 56.0 |
| p25 | 129.3 |
| median | 144.6 |
| mean | 143.9 |
| p75 | 160.0 |
| p90 | 176.1 |
| max | 205.0 |

The default config therefore samples `target_customers_per_active_cluster` from
a clipped lognormal distribution with median `141.6661` and sigma `0.1835`.
For a fixed `num_customers`, this converts Amazon route-size statistics into a
daily active-community count. It does not prescribe vehicle count, generated
route count, route sequence, route duration, or objective value. Those remain
solver-side quantities. Cus1800 therefore activates a realistic number of
communities without leaking Amazon solution-side structure; small Cus5/Cus15
cases still activate multiple communities through the sparse-day lower bound.

This matches a station-day view: a service territory is stable, but each day activates a subset of communities and addresses.

Current behavior:

```text
Cus5   -> about 3 active clusters
Cus50  -> about 6 active clusters
Cus100 -> about 7 active clusters
Cus1800 -> about 14 active clusters
```

This gives small instances enough diversity without destroying large-scale Amazon-like locality.

### 1.4 Charging Stations

Amazon does not provide EV charging infrastructure. The generator therefore treats CS placement and activation as a modeled EV overlay.

Mother-board CS are placed on arterial/corridor edges, not inside residential micro-zones. Daily CS are selected using a graph facility-location objective over active customers:

```text
J(S) = alpha * mean customer-to-nearest-CS distance
     + beta  * p90 customer-to-nearest-CS distance
     + gamma * max customer-to-nearest-CS distance
     + eta   * redundancy penalty
     + unreachable penalty
```

This combines p-median, k-center, maximal-covering, and redundancy ideas. It is a reasonable reviewer-facing assumption because it explains why a subset of available infrastructure is relevant to a daily operating area.

Known boundary:

- CS counts are not Amazon-calibrated. They should be chosen from the intended EV benchmark regime, Solomon-style ratios, public charging assumptions, or sensitivity analysis.

### 1.5 Time Windows

Time windows are daily attributes, not static customer attributes. The generator uses:

- a beta distribution for daily TW presence rate;
- strain/loose scenario mixture for window width;
- truncated normal models for width and center;
- feasibility correction using depot-to-customer time, customer-to-depot time, service time, working start, and working end.

This is better than hard Bernoulli copying from observed stops because it creates a smooth generative distribution while preserving the observed aggregate TW rate and width/center patterns.

Known boundary:

- If a reference dataset has true appointment types, the strain/loose mixture can be replaced by a fitted mixture model.

### 1.6 Demand And Service Time

Demand is modeled as package volume in cm3. Package count uses a negative-binomial model and per-package volume uses a lognormal model. Service time is a simple regression-like model:

```text
service_seconds = base
                + beta_package * package_count
                + beta_volume  * volume_cm3
                + lognormal residual
```

This is defensible because service time is empirically correlated with package count and package volume, but still has substantial residual variation.

### 1.7 Feasibility

Mother-board validation checks structural reachability. Daily instance validation is authoritative:

- active road distances are computed after active customers and active CS are selected;
- inactive CS cannot appear in the active EV transition graph;
- greedy audit must produce a complete feasible route set before the instance is saved.

If greedy succeeds, the instance has a constructive feasible solution and the returned vehicle count is an upper bound. If greedy fails, the instance may still be feasible, but the generator rejects it conservatively and resamples.

## 2. Calibration Workflow For A New Dataset

Use this order. Do not tune all parameters at once.

### Step 1: Extract Input-Side Reference Statistics

From the real dataset, compute the following by depot/station/day whenever possible.

| Statistic | Use In Generator |
|---|---|
| daily active customer count | choose `num_customers` scenarios |
| station-level latent customer count | choose `mother_num_customers` |
| depot-to-stop distance distribution | cluster radius and service region scale |
| nearest-neighbor stop distance | address spacing and local density |
| within-route/community pairwise distance | community footprint |
| route/community bbox width, height, area | community shape and spread |
| stop spatial dispersion | community spread |
| number of route/community groups per station-day | active cluster sampling |
| package count per stop | demand package-count model |
| package dimensions / volume | volume demand model |
| service time by stop | service-time regression |
| TW presence rate | beta distribution for TW rate |
| TW width and center distributions | strain/loose TW models |
| travel time / road or Euclidean distance | congestion/effective speed |
| working start / dispatch start | day start distribution |

Do not use actual stop sequence, realized route duration, realized route cost, or driver behavior as generator rules. Historical route counts should also not be used as a required vehicle count. They can be reported as diagnostics and, at most, used indirectly to calibrate input-side community demand scale.

### Step 2: Calibrate Region Scale And Depot

Relevant config block:

```yaml
region:
  area_size_km
  depot_sampling.central_fraction
  cluster_geometry.radius_min_km
  cluster_geometry.radius_max_km
  cluster_geometry.radius_median_km
  cluster_geometry.radius_lognormal_sigma
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `area_size_km` | absolute coordinate grid | larger possible service territory | normalization range / city scale |
| `central_fraction` | depot sampling area | depot varies more within center | depot location diversity |
| `radius_median_km` | typical depot-community distance | larger depot-stop mean and median | depot-stop mean/p50 |
| `radius_lognormal_sigma` | radial heterogeneity | more near/far communities | depot-stop p10/p90 spread |
| `radius_max_km` | farthest service communities | larger depot-stop p90/max | outer service radius |
| `service_sector_angle_deg` | service-sector fan width | wider directional coverage | visual station service sector |
| `infill_probability` | off-corridor communities | less corridor-only structure | urban infill behavior |

Recommended check: match depot-stop mean, p50, and p90 first. If those are wrong, do not tune customer micro-zones yet.

### Step 3: Calibrate Cluster Count And Community Size

Relevant config block:

```yaml
region.cluster_count:
  reference_customers
  reference_clusters
  scale_exponent
  min
  max
region.cluster_assignment_alpha
```

Auto cluster count is:

```text
K(N) = ceil(reference_clusters * (N / reference_customers)^scale_exponent)
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `reference_clusters` | cluster count at reference scale | more communities | active/community count |
| `scale_exponent` | how clusters scale with customer pool | more clusters at large N | multi-scale behavior |
| `cluster_assignment_alpha` | balance of customers among clusters | larger gives more equal clusters | cluster size distribution |
| `min_customers_per_cluster` | minimum cluster size | avoids tiny latent clusters | small-scale stability |

For Amazon-like Cus1800, the current default gives about 14 active clusters per day after active sampling. For small instances, sparse-day lower bounds spread customers across several communities.

### Step 4: Calibrate Customer Density And Community Shape

Relevant config block:

```yaml
region.customers:
  cluster_spread_km
  micro_zone_size_mean
  min_spacing_km
  placement_model: road_oriented_community
  road_angle_jitter_deg
  community_long_axis_scale
  community_lateral_axis_scale
  zone_longitudinal_median_km
  zone_lateral_median_km
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `cluster_spread_km` | overall community footprint | larger community bbox/pairwise/dispersion | community bbox and dispersion |
| `micro_zone_size_mean` | stops per micro-zone | larger local address groups | micro-zone granularity |
| `min_spacing_km` | minimum address spacing | larger nearest-neighbor distance | NN distance |
| `community_long_axis_scale` | macro spread along road direction | more elongated community patches | bbox width / visual shape |
| `community_lateral_axis_scale` | macro spread perpendicular to road | wider community patches | bbox area / non-line-like appearance |
| `road_angle_jitter_deg` | local road direction variability | less uniformly aligned communities | visual naturalness |
| `zone_longitudinal_median_km` | micro-zone along-road spread | larger within-zone pairwise distance | local pairwise distance |
| `zone_lateral_median_km` | micro-zone cross-road spread | thicker local blocks | local dispersion |

Current A/B blend defaults:

```yaml
cluster_spread_km: 1.90
min_spacing_km: 0.052
community_long_axis_scale: 0.56
community_lateral_axis_scale: 0.39
zone_longitudinal_median_km: 0.15
zone_lateral_median_km: 0.07
```

Observed Cus1800 A/B blend metrics:

| Metric | A/B Blend | Amazon Target |
|---|---:|---:|
| depot_stop_mean_km | 18.169 | 18.053 |
| depot_stop_p90_km | 31.060 | 30.457 |
| nearest_neighbor_mean_km | 0.103 | 0.095 |
| community_pairwise_mean_km | 1.532 | 1.622 |
| community_bbox_area_km2 | 14.022 | 12.707 |
| community_dispersion_km | 1.107 | 1.232 |

### Step 5: Calibrate Daily Active-Customer Sampling

Relevant config block:

```yaml
active_customer_sampling:
  target_customers_per_active_cluster
  small_scale_min_active_clusters
  sparse_day_log_cluster_base
  sparse_day_log_cluster_scale
  ensure_selected_cluster_coverage
  macro_activity_lognormal_sigma
  micro_activity_lognormal_sigma
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `target_customers_per_active_cluster` | customers per active community | fewer active clusters for fixed N | active cluster count |
| `small_scale_min_active_clusters` | lower bound for tiny N | prevents Cus5 collapse | small-scale visual diversity |
| `sparse_day_log_cluster_scale` | sparse-day cluster count | more active clusters for small/medium N | Cus5/Cus50/Cus100 behavior |
| `ensure_selected_cluster_coverage` | at least one customer per selected cluster | prevents selected clusters being empty | robust active cluster count |
| `macro_activity_lognormal_sigma` | day-level cluster imbalance | more uneven cluster sizes | daily demand heterogeneity |
| `micro_activity_lognormal_sigma` | within-cluster hotspot strength | more local hotspot variation | micro-zone activity |

Practical rule:

```text
If small instances look too local, increase sparse_day_log_cluster_scale.
If large instances look too scattered, increase target_customers_per_active_cluster.
```

### Step 6: Calibrate Road Graph Shape

Relevant config block:

```yaml
region.road_graph:
  road_stretch_factor
  gateway_knn
  gateway_max_edge_km
  junction_knn
  junction_max_edge_km
  arterial_lateral_sigma_km
  local_street_factor
  local_knn
  local_max_edge_km
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `road_stretch_factor` | road detour vs Euclidean | larger road distances and travel times | road/Euclidean ratio |
| `gateway_knn` | arterial connectivity | more inter-community shortcuts | route distance realism |
| `gateway_max_edge_km` | max arterial edge | longer gateway links allowed | graph connectivity / shortcut level |
| `arterial_lateral_sigma_km` | junction deviation | less straight depot-cluster roads | visual road realism |
| `local_street_factor` | local street node count | denser local street graph | local access structure |
| `local_knn` | local connectivity | more local shortcuts | local travel distance |

Tune this after spatial stop statistics are reasonable. Overly dense graphs make distances too Euclidean; overly sparse graphs make detours too large.

### Step 7: Calibrate Charging Stations

Relevant inputs:

```bash
--mother_num_charging_stations
--num_charging_stations
```

Relevant config blocks:

```yaml
region.charging_stations
cs_activation
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `mother_num_charging_stations` | candidate infrastructure density | more possible CS coverage | infrastructure pool size |
| `num_charging_stations` | active daily CS count | easier EV feasibility | benchmark EV difficulty |
| `candidate_cs_per_customer` | local CS prefilter width | more activation options | CS reachability |
| `alpha_mean` | p-median pressure | lower average customer-CS distance | mean coverage |
| `beta_p90` | tail coverage pressure | lower p90 distance | p90 coverage |
| `gamma_max` | worst-case pressure | lower max distance | max coverage / feasibility |
| `eta_redundancy` | anti-collocation | more spread-out CS | infrastructure diversity |
| `repulsion_km` | minimum useful spacing | avoids duplicate CS | CS layout naturalness |

Because Amazon has no EV CS data, report CS settings as EV overlay assumptions or sensitivity regimes, not Amazon facts.

### Step 8: Calibrate Demand And Service Time

Relevant config blocks:

```yaml
demand
service_time
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `mean_extra_packages` | package count above 1 | larger average demand | packages per stop |
| `dispersion` | package-count skew | smaller creates heavier tail | package count tail |
| `median_cm3` | typical package volume | larger mean volume | volume per package |
| `sigma` | volume heterogeneity | heavier volume tail | package-size skew |
| `base_seconds` | base stop time | larger service times all stops | service time intercept |
| `beta_package_seconds` | package-count effect | stronger service-count correlation | service vs package count |
| `beta_volume_seconds_per_cm3` | volume effect | stronger service-volume correlation | service vs volume |
| `lognormal_noise_sigma` | unexplained heterogeneity | wider service-time distribution | service residual variance |

Fit package count and volume first, then fit service time conditional on those variables.

### Step 9: Calibrate Working Day, Congestion, And Speed

Relevant config blocks:

```yaml
day
congestion
vehicle
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `working_start_mean_h` | average start time | shifts all TW/service day later | dispatch/start distribution |
| `working_start_std_h` | start-time variation | more day-to-day variation | start-time variance |
| `working_horizon_hours` | available operating day | easier TW feasibility | working end assumption |
| `weekday_weight/weekend_weight` | day-type mix | more weekday/weekend days | calendar mix |
| `congestion.mean_factor` | effective speed multiplier | higher travel speed | implied urban speed |
| `congestion.std_factor` | speed variation | more instance-level speed variation | travel-time variance |
| `vehicle.design_speed_kmh` | nominal vehicle speed | scales travel time | base speed profile |

Current default uses EDV 700 parameters and a congestion factor so effective urban speed is below design speed.

### Step 10: Calibrate Time Windows

Relevant config block:

```yaml
time_window
```

| Parameter | Controls | Increase Effect | Primary Target |
|---|---|---|---|
| `presence_rate_beta.alpha/beta` | daily TW customer rate | higher alpha or lower beta gives more TW stops | TW presence rate |
| `presence_rate_multiplier` | global TW rate scale | more/fewer TW stops | sensitivity |
| `realistic_strain_share` | strict-window share | more narrow windows | TW tightness |
| `width_mean_h/std_h` | TW width distribution | wider/narrower windows | width quantiles |
| `center_mean_h/std_h` | TW center distribution | shifts TW earlier/later | center/start/end distribution |
| `service_time_selection_weight` | TW assigned to complex stops | stronger service-TW correlation | TW vs service relation |

Recommended fitting:

1. fit daily TW presence rate with a beta distribution;
2. split observed TW widths into strain/loose groups;
3. fit width and center distributions per group;
4. generate TW, then check feasibility rate and greedy vehicle upper bound.

### Step 11: Validate Before Large Generation

For every new calibration, run at least:

```text
10 seeds x 1000 instances for target scale if computationally possible
5-10 visual plots for each important scale
```

Validation metrics:

| Category | Metrics |
|---|---|
| spatial | depot-stop mean/p50/p90, NN distance, pairwise, bbox, dispersion |
| daily activity | active cluster count, cluster-size distribution, customer exposure |
| EV infrastructure | customer-to-nearest-CS mean/p90/max, unreachable rate |
| demand/service | package count, volume, service mean/p90, correlations |
| TW | presence rate, width distribution, center distribution, infeasible correction rate |
| feasibility | greedy success rate, vehicle upper bound distribution |
| storage/runtime | matrix size, generation seconds per instance |

## 3. Recommended Tuning Order

Use this order to avoid unstable calibration.

1. `area_size_km`, depot, cluster radius: match depot-stop statistics.
2. cluster count and active-cluster sampling: match daily community count.
3. customer placement: match NN, bbox, pairwise, dispersion.
4. road graph stretch/connectivity: match road-distance scale.
5. demand and service time: match package and service distributions.
6. day/congestion: match effective speed and working start.
7. TW: match TW rate, width, center, then feasibility.
8. CS pool and activation: choose benchmark EV difficulty, then verify feasibility.
9. run multi-seed validation and plot review.

## 4. Reviewer-Facing Wording

Use this framing:

> We model benchmark generation as a two-stage process. First, a city/region-level mother board is generated to represent a station service territory calibrated from real last-mile input-side statistics. Second, daily EVRP-TW-D instances are sampled from this fixed territory by activating a subset of customers and charging stations, then generating daily demand, service times, and time windows. This mirrors real last-mile operations, where the service territory is stable but daily orders vary.

For Amazon specifically:

> Amazon calibrates last-mile order-set statistics, including spatial density, service territory scale, package demand, service-time, and time-window patterns. EV-specific operational parameters such as battery capacity, energy consumption, charging power, and charging station availability are introduced as documented EV benchmark assumptions.

Avoid saying:

```text
Amazon provides EVRP charging data.
Amazon route sequence is used to generate instances.
Generated instances reproduce Amazon driver behavior.
```

These claims would be easy for reviewers to challenge.

## 5. Current Default Profile

The current default config is the A/B blended Amazon-like profile:

```text
region profile: mixed_amazon_station_region
placement: road_oriented_community
vehicle: Rivian_EDV_700
objective basis: distance_matrix_km
time unit for saved time fields: seconds
shortest path oracle: source_cache
```

Recent spot checks:

```text
Cus1800 / CS12: 10/10 greedy feasible
Cus100 / CS6:  5/5 greedy feasible
Cus50 / CS4:   5/5 greedy feasible
Cus5 / CS2:    5/5 greedy feasible
```

Small-scale active-cluster behavior after the latest update:

```text
Cus5:   3 active clusters
Cus50:  6 active clusters
Cus100: 7 active clusters
```
