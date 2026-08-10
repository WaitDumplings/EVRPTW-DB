# CLE construction pipeline

## 1. Contract and execution graph

```text
frozen city/service boundaries + frozen OSM PBF
                         |
                         v
               directed road graph ------ HPMS/OSM legal speed evidence
                         |                           |
          +--------------+---------------+           v
          |              |               |    reference speed profile
          v              v               v
   OSM depot tags   AFDC charging   eligible physical roads
          |              |               |
          +-------+------+-------+-------+
                  |              |
       NSI residential data      |
                  |              |
      Microsoft footprint match  |
                  +--------------+
                         |
                   CLE assembler
                         |
             independent hash/semantic verifier
```

Every branch records native source fields, normalized fields, source paths,
SHA-256 hashes, row counts, and unresolved QA states before assembly.

## 2. Boundary and road graph

1. Freeze the Census Place polygon as the administrative boundary.
2. Subtract Census AREAWATER polygons to obtain the land-only service boundary.
3. Extract a directed `drive` graph from a frozen Geofabrik PBF. Preserve
   one-way direction, parallel edges, original OSM IDs/tags, and geometry.
4. Audit all weakly connected components in the exact city graph.
5. Let `G_city` be all in-boundary roads and `L_city` its largest weak
   component. Compute:

   ```text
   node_coverage = nodes(L_city) / nodes(G_city)
   physical_road_length_coverage = undirected_physical_length(L_city)
                                   / undirected_physical_length(G_city)
   ```

   The second metric deduplicates reciprocal directed representations of the
   same physical road so two-way streets are not counted twice.
6. If either coverage is below 0.99 or 0.995 respectively, test routing
   envelopes at 1, 2, 5, 10, and 20 km. Select the smallest envelope whose real
   OSM roads connect the retained city roads and pass both thresholds.
7. Mark all nodes/edges outside the service boundary `transit_only=true`. They
   can connect a route but cannot host a service location or facility.
8. Compute stable directed SCC labels. The largest directed SCC is the Stage-1
   reference service SCC; smaller SCCs remain in the road artifact for audit.

This is a weak-connectivity construction gate, not a claim that every directed
node pair is mutually reachable. Directed strong-component statistics are
still recorded, and every active Stage-2 OD pair must pass directed route
feasibility. The pipeline never makes one-way roads bidirectional and never
draws synthetic intercomponent roads.

## 3. Charging sites

1. Freeze AFDC rows with `Fuel Type Code=ELEC`, `Status Code=E`,
   `Access Code=public`, and `Country=US`.
2. Retain raw AFDC coordinates and address fields.
3. Produce optional Census address anchors and exact-address OSM charging-POI
   matches from the same PBF snapshot.
4. Resolve geometry with explicit precedence: reviewed manual override, OSM
   exact-address match, raw AFDC. Census is QA/address access only.
5. Assign a coordinate-evidence tier. Census corroboration does not convert an
   AFDC point into exact charger geometry; uncorroborated rows remain visible
   but cannot enter the default candidate pool.
6. Spatially retain sites inside the service boundary.
7. Store L2/DC-fast port counts, connector tokens, access restrictions, and
   maximum vehicle class separately. A Tesla/NACS site is not removed merely
   because the reference vehicle connector policy is unresolved.
8. Project every anchorable site to the nearest eligible physical road. Keep
   all distances; `>250 m` is a QA flag only.
9. Label the exact projection by directed SCC and quarantine it from the
   default pool unless it inherits the reference SCC.

AFDC does not consistently provide per-port power. Missing kW remains missing;
the CLE does not infer a charging curve or claim that every public site can
serve a medium-duty delivery van.

## 4. Depot candidates

Depot evidence is extracted from the same frozen OSM PBF with tags such as
`building=warehouse`, `industrial=warehouse`, `office=logistics`,
`amenity=post_depot`, accepted `depot=*`/`logistics=*` values, and relevant
industrial land use.

The evidence policy separates:

- Tier A: named carrier/logistics evidence tied to a physical warehouse,
  dispatch-function name, or explicit depot/logistics feature. Retail shipping
  counters are excluded.
- Tier B: warehouse/logistics proxy lacking verified current parcel-dispatch
  function. Included in the optional pool.
- Tier C: generic industrial or ambiguous carrier point. Excluded by default.

Area is measured for polygonal features and stored continuously. The 1,000 m2
reference only flags sensitivity; it neither creates nor deletes Tier B. Every
retained candidate is road-anchored, with long access distance flagged rather
than deleted.

## 5. Latent service locations

1. Query NSI over the exact land boundary in deterministic tiles and freeze the
   raw responses.
2. Keep ordinary residential occupancy families `RES1`, `RES2`, and `RES3`;
   exclude institutional residential families from the ordinary parcel pool.
3. Group NSI records by the first available shared structure identifier:
   `ftprntid`, `usastrucid`, `bid`, then record ID fallback.
4. Sum residential-unit evidence within the group and retain original NSI
   occupancy, footprint, floor, height, area, and mixed-use signals.
5. Classify the latent service location:

   | Class | Default evidence rule |
   | --- | --- |
   | `house` | fewer than 2 estimated units, ordinary residential fallback |
   | `manufactured_home` | `RES2` fallback when units do not imply multi-unit |
   | `small_apt` | 2-4 estimated units or lower `RES3` fallback |
   | `medium_apt` | 5-19 estimated units or mid `RES3` fallback |
   | `large_apt` | at least 20 estimated units or high `RES3` fallback |

6. Match each NSI structure point to Microsoft footprint geometry:
   - G1: NSI point is covered by a Microsoft polygon.
   - G2 candidate: nearest polygon is at most 10 m away and its area differs by
     at most a factor of four. G2 remains explicitly reviewable.
7. Do not infer apartment complexes merely because buildings are nearby. A
   future complex adapter may merge buildings only with independent parcel,
   address, parcel-boundary, or site evidence.
8. Project the resolved building boundary to the nearest eligible physical road
   and materialize a road projection node plus symmetric bidirectional access
   connector. `>200 m` is a QA flag, not a deletion rule.
9. Infer the SCC inherited by the exact directed-edge split. Retain all source
   locations, but only reference-SCC projections are default instance
   candidates. A bidirectional connector alone is not proof of road-network
   round-trip reachability.

The CLE stores `residential_units` and location type as activation covariates.
It does not store realized package count, demand, service time, or time window.

## 6. Speed profiles

Each directed physical edge keeps three separate concepts:

1. `legal_speed_kph`: direction-applicable OSM `maxspeed`, generic OSM
   `maxspeed`, high-confidence HPMS `SPEED_LIMIT` fill when OSM is missing, then
   transparent within-city class/mode/parent/global median imputation.
2. `operating_mode`: HPMS `F_SYSTEM` 1-2 -> H, 3-6 -> M, 7 -> U when a
   high-confidence normalized match exists; otherwise OSM `highway=*` fallback.
3. `reference_speed_kph`: a running-speed prior used before Stage-2 variation.

The U.S. adapter uses Table 14 of NREL/TP-5400-65921. The source reports
Average Driving Speed profile means of 48.57, 33.64, and 22.62 mph for FDNA
clusters 3, 2, and 1. Our adapter explicitly maps those descending profiles to
H/M/U; NREL does not assign H/M/U labels to OSM road edges.

```text
reference_speed(edge) = min(
    legal_speed(edge),
    NREL_FDNA_profile_speed(mode(edge)),
    optional_versioned_vehicle_cap
)
```

This is running speed on the edge, not depot-to-customer or door-to-door route
average. Stop/service time and instance-static weekday/weekend directed
variation belong to Stage 2. Pilot scenario generation is available only for
engineering QA and is off by default.

## 7. Assembly, packaging, and verification

The assembler requires all layer manifests to use the same city slug, boundary,
and operational-graph hash. It copies small canonical tables into the CLE,
references the work GraphML artifact, writes a source registry, computes every
published output hash, and records unresolved evidence gates. This first CLE is
a Generator-owned technical/debug artifact.

The verifier recomputes hashes, counts, ID uniqueness, Stage-1 inactivity,
positive finite speed/travel-time values, reference-speed legal caps, and
facility/customer release semantics. Technical verification and external
real-world validation are intentionally separate statuses.

For each Stage-2 instance, the selected depot, active customers, and permitted
charging stations must inherit the same reference SCC. The current customer
access materializer independently checks `reference road -> customer` and
`customer -> reference road` after virtual edge splitting. The Stage-2 facility
materializer must apply the same post-materialization check to the selected
depot and charging stations before matrices are accepted.

After technical verification, the packager:

1. copies the work CLE into a release staging directory;
2. copies `graph_operational.graphml` and the road manifest into `graph/`;
3. replaces absolute graph references with package-relative paths;
4. removes machine-local paths from the release source registry while retaining
   source hashes;
5. recomputes every affected output hash;
6. runs the strict portable verifier; and
7. atomically promotes the staging directory into `EVRPTW_Dataset`.

The strict verifier fails on missing packaged runtime files, absolute output
paths, paths escaping the CLE root, or graph-reference hash mismatches. Passing
that check sets `portable_package_verified=true`; it does not override open
scientific gates or set `release_eligible=true`.
