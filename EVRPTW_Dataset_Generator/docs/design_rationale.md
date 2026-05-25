# EVRP-TW-Hierarchy-D Design Rationale

## Two-Stage Interpretation

The generator separates region-level structure from daily operating uncertainty.

- **Mother board**: a city/region/station service territory. It stores the road graph, depot, latent customer pool, community structure, and charging-station candidate pool.
- **Daily instance**: one operating day sampled from a mother board. It activates customers and charging stations, then samples demand, service time, and time windows.

This mirrors last-mile operations: the service territory is stable, while daily orders and available operating infrastructure vary.

## Road-Network-First Spatial Model

A region is generated as a connected graph embedded in a kilometer grid. Graph vertices are first-class road vertices; depot, customer access nodes, cluster gateways, and charging stations are attached to this graph.

The default spatial profile uses an Amazon-calibrated service-sector model rather than an all-around radial ring. Cluster centers are sampled from a mixture of service corridors and infill communities, with depot-stop radii calibrated to Amazon depot-stop quantiles. This matches the fact that a delivery station usually serves selected urban/suburban sectors rather than uniformly surrounding the depot.

The model has three spatial layers:

1. **Backbone/gateway layer**: depot and community gateways are connected with a sparse arterial network.
2. **Local-street layer**: each community has local street vertices connected to its gateway.
3. **Customer access layer**: latent customers are sampled inside micro-zones and attached to the nearest local street.

Cluster count is auto-scaled by

```text
K(N) = ceil(K_ref * (N / N_ref)^alpha)
```

with default `K_ref=18`, `N_ref=1800`, and `alpha=0.8`. Users may override it in the config.

## Charging Station Pool

Charging stations are placed on corridor/arterial road edges rather than inside customer micro-zones. This matches the operational assumption that public or fleet-accessible chargers are more likely on major roads and highway exits than in residential community interiors.

The mother board stores a large candidate pool. Daily instances activate a smaller subset.

## Daily CS Activation

For each active day, CS activation is solved as a graph facility-location problem over active customers. Candidate CS are pre-filtered locally, then selected greedily by minimizing

```text
J(S) = alpha * mean_i d_G(i, S)
     + beta  * p90_i d_G(i, S)
     + gamma * max_i d_G(i, S)
     + eta   * redundancy_penalty(S)
     + unreachable_penalty(S)
```

where `d_G` is road-network shortest distance. This combines:

- p-median behavior through the active-customer mean coverage term;
- k-center behavior through p90/max coverage terms;
- maximal-covering behavior through the unreachable penalty;
- infrastructure diversity through the redundancy penalty.

Inactive charging stations are never used in the daily EV shortest-time matrix.

## Shortest-Path Computation

Road edges are non-negative, so Dijkstra remains the exact shortest-path primitive. The default implementation uses an on-demand source cache: once a road node is used as a source in a daily instance, its source-to-all shortest-distance vector is cached for the current region. This avoids computing terminal pairs that are never activated by sampled days.

For heavy reuse of one region, the generator can instead precompute a terminal-to-terminal road distance matrix over depot, latent customers, and CS candidates. Daily instances then extract active submatrices by indexing. This is useful only when the fixed precomputation and memory cost can be amortized across many operating days.

This is preferable to Floyd-Warshall, which is cubic in the number of road vertices, and to A*, which helps point-to-point queries but not repeated all-terminal matrix extraction.

## Active EV Shortest-Time Matrix

For each daily instance, the generator computes a road shortest-distance matrix over:

```text
depot + active customers + active charging stations
```

It then builds an EV transition-time graph. A transition is allowed if its road distance is within battery range. Departing from a charging station includes full-charge time:

```text
time(i, j) = road_distance(i, j) / effective_speed
           + full_charge_time if i is an active CS
```

The final `shortest_time_matrix_s` is computed only on the active terminal set. For the common local-delivery case where all active terminals are mutually battery-reachable, the generator uses an O(n^2) fast path because charging detours cannot improve travel time when full charging time is positive. The benchmark charging policy is fixed full-charge at active charging stations, matching the generator and exact solver semantics.

The persisted default is storage-efficient: `distance_matrix_km` is saved because distance is the objective and the energy basis; time matrices are derived from distance and speed during generation/audit and are not saved unless requested; if saved, they use seconds. Each instance also saves `cs_time_to_depot_s`, the charging-aware CS-to-depot return time used by time-window and future-feasibility logic.

## Daily Active-Customer Sampling

Daily instances activate a subset of community clusters before sampling customers. The default active community size is no longer a single hard-coded value. It is sampled from an Amazon-calibrated clipped lognormal distribution fitted to `mean_route_unique_customers` over `date x station` records. This Amazon field is used only as an input-side proxy for local community demand scale, not as a generated vehicle count, route count, route sequence, route duration, or objective-value prior. For Cus1800, this usually produces about 12-15 active communities. This prevents each day from activating the whole latent region while keeping the daily view close to station-day operations.

## Demand, Service Time, And Time Windows

Demand, service time, and time windows are daily attributes. They are sampled after active customers are selected.

- Demand uses package-count and package-volume distributions in cubic centimeters.
- Service time is modeled as a function of package count and volume with lognormal residual variation.
- Time windows are not assigned by hard Bernoulli copying. The daily TW presence rate is sampled from a beta distribution, then constrained by reachable service envelopes. A TW upper bound is interpreted as the latest service-start time; service completion and depot return are checked separately by generator audits and solvers.

## Mother-Board Freshness

A region is discarded when repeated sampling could overfit a model to one geometry:

```text
sampled_days_from_region >= reuse_limit
or customer_exposure_rate >= threshold
or recent_mean_jaccard_distance <= threshold
```

This keeps the benchmark as a pool of regions and a pool of daily instances.

## Feasibility Semantics

Mother-board validation checks structural serviceability:

- road graph connectivity from depot;
- depot/customer/cluster battery reachability under the full CS pool;
- existence of customer and cluster candidate CS lists.

Daily instance validation is authoritative for EVRP-TW-D feasibility:

- active road distances are computed after customer and CS activation;
- active charging-aware shortest-time matrix uses only active CS;
- graph-aware greedy audit must serve all customers before the instance is saved.
