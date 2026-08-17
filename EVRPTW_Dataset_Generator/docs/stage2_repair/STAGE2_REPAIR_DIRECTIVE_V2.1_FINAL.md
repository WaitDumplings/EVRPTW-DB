# Stage-2 Repair Directive V2.1 Final

Status: frozen implementation contract  
Branch: `stage2-repair-candidate`  
Supersedes: every Stage-2 repair recommendation or directive before V2.1

This is the only normative Stage-2 repair document in the repository. Gate
failures require changes to generation; acceptance thresholds must not be
relaxed after observing pilot results.

## 1. Frozen decisions

- D-1: hybrid blocked Amazon split. `METRIC-HOLDOUT` isolates whole stations;
  `GEN-TRAIN` and `GEN-EVAL` partition the remaining station-days by usable
  template mass.
- D-2: every selected structure source has its own P99 `T_env` and decile
  edges. Global P95/P99/P99.5 are reference-only.
- D-3: canonical nested views use the deterministic region-first heuristic in
  Section 5. The earlier region-by-decile balanced partition is superseded.
- D-4: canonical running-time matrices use zero temporal turn penalties.
  Geometry turn penalties 3/8/20 seconds are an optional adapter tested
  separately and never used to generate canonical release matrices.
- D-5: for every evaluable `day_type × scale × source_mode` primary stratum,
  every M2/M3 component must satisfy
  `Q0.90(D_generated-to-holdout) <= Q0.90(D_real-to-real)`.
- D-6: `charging_power_derating_factor=0.90` and
  `p_battery=0.90*min(p_station_or_imputed,p_vehicle_cap)`.
- B-3: the direct depot-customer-depot energy screen makes the single-customer
  feasibility core empty by construction. All K charging stations are
  relevance/optimization fill, never a computed energy core.
- Day types remain frozen at weekday:weekend 5:2.

## 2. Amazon cohort contract

### 2.1 METRIC-HOLDOUT

`METRIC-HOLDOUT` contains every station-day, route, stop and template from the
three stations frozen in `configs/amazon_cohort_split_v1.json`. None may be
used for instance generation. Holdout is used only for M2/M3 reference and
real-to-real variation.

The canonical H3 search considers exactly three whole stations in the
14.5%-15.5% usable-mass window. Candidates must support both source types at
N=1000 for both day types. Ranking is deterministic:

1. maximize the minimum weekday/weekend, order/structure N=2000 day count;
2. minimize absolute deviation from 15% usable mass;
3. lexicographic station-code tuple.

The frozen result is `DCH2 + DLA9 + DSE2`.

Every declared primary stratum requires an evaluable qualified holdout gate.
If a primary stratum lacks support, `release_calibrated` remains false unless
that stratum is removed by a new explicitly scoped profile ID. It must not be
converted to report-only in the same canonical profile. Cus2000 and composite
strata are report-only.

### 2.2 Generation pools

`GEN-TRAIN` is used only by train families. `GEN-EVAL` is used by validation,
Test1, Test2, Test3 and scalability families. Structure and order sources for
one family must both come from its track's pool; there is no unrestricted-pool
fallback. A deterministic GEN-EVAL source ledger assigns station-days to
evaluation cohorts and records unavoidable reuse.

The following assertions are hard errors:

1. metric-holdout stations and generation stations are disjoint;
2. station-day ownership is pairwise disjoint and exhaustive;
3. template IDs are pairwise disjoint across the three pools;
4. route IDs are pairwise disjoint across the three pools.

`template_id` includes station, date, route and stop. Every family stores its
structure/order source IDs and whether the pair is the same station-day, same
station, or independent station-day.

Canonical Cus100/Cus500/Cus1000 require
`(SINGLE_STRUCTURE_DAY,SINGLE_ORDER_DAY)`. Retry may change source IDs but not
source mode. Composite sources are permitted only for Cus2000 scalability or
separately versioned noncanonical profiles and are always report-only.

## 3. Per-source territory

Structure source selection precedes territory construction. For a single
structure day, P99 `T_env` and decile edges are calculated from that day. For a
same-station, same-day-type composite, they are calculated from the combined
source. The depot star is cached by `(city,depot,day_type)`; source-specific
filtering and binning do not rerun Dijkstra.

The source's IDs, T_env and complete edges are stored in the family manifest.
`route_decile_histogram` uses source-specific edges. A
`territory_too_small` rejection contains the structure source ID; retry selects
a new source and rebuilds the territory.

## 4. Charging-station relevance fill

There is no core-selection or fixed-point closure stage. A candidate CS is
eligible only if it belongs to the depot's battery-feasible communicating set:
depot-to-CS and CS-to-depot must each be feasible, possibly through multiple
CS hops, under canonical zero-turn time and the frozen energy model.

Road-time selection uses the full eligible roster and a deterministic greedy
coverage objective. For customer `i` and station `s`, the replacement-time
increment is clamped for floating-point safety:

```text
delta(i,s) = max(0, min(t(i,s)+t(s,depot)-t(i,depot),
                        t(depot,s)+t(s,i)-t(depot,i)))
```

Child views reselect their K stations from the eligible parent roster; charger
indices are arbitrary and are not parent prefixes. Selected stations must
retain bidirectional energy eligibility.

Post-generation metrics include `certificate_used_cs_count`,
`solution_cs_visit_count` when a baseline solver is run, and
`binding_energy_count`. A solver-derived binding count records SOC tolerance,
solver version and time budget in `run_manifest.json`; those runtime fields do
not enter the instance hash.

## 5. Deterministic region-first nested partition

This is a deterministic priority order, not a claim of globally optimal
combinatorial fragmentation.

1. Sort regions by decreasing size; ties use a seeded region-ID rank.
2. Assign a whole region to the child with maximum remaining capacity that can
   contain it; child-index tie-break.
3. If no child can contain it, mark it split and fill children in decreasing
   remaining-capacity order.
4. Correct residual size only by moving customers within already split
   regions.
5. Within split regions, use deterministic min-cost flow to minimize decile
   margin deviation; ties use H64 rank.

Every partition has exact child sizes, disjoint children and union equal to the
parent. Each child stores region count, parent regions touched, split-region
count, fragmentation score, largest-region share, region HHI, road-community
component count and M2/M3. No single region-count threshold is a gate. Failure
of D-5 must not reactivate the superseded balanced partition.

## 6. Metrics and pairing

- M1-family: normalized W1 between a generated family's depot-time
  distribution and its assigned structure-template targets.
- M1-corpus: normalized W1 between one `scale × day_type` generated corpus and
  `F_structure(N,d)`.
- M2: distribution of each customer's outgoing network nearest-neighbor time,
  `min_{j != i} t(i,j)`.
- M3-P50/P90: P50/P90 of all ordered customer-pair directed travel times within
  each generated region; the Amazon comparison unit is a route.
- M4: construction audit including regions, sizes, rounding residuals and split
  regions.
- M5: community count, largest-community share and HHI versus a uniform
  baseline with the same view/customer count; diagnostic only.

Time values are divided by their own source T_env before W1 calculation.
Holdout days use the same deterministic scaling rule to form the same N.
Within one stratum, every generated view is paired with every qualified
holdout day. Real-to-real uses every unordered pair of distinct qualified
station-days. Pair subsampling is forbidden.

`metric_pairing_ledger.parquet` stores generated view, holdout day, pair ID,
station block, metric component and source mode. Pair dependence is handled by
station-block bootstrap. Confidence intervals are report-only in V2.1.
Jacksonville/Test2 never calibrates thresholds.

Primary gate strata are Cus100/Cus500/Cus1000. Cus50 is reported as a separate
compatibility-track gate. Cus2000 and composite strata are report-only.

## 7. Canonical operations profile

Canonical turn times are zero. After zeroing, validation checks for immediate
edge reversal at virtual access-connector split nodes. If observed, the graph
forbids topological U-turns at those nodes and records the rule; no temporal
penalty is introduced.

Charging uses `charging_power_derating_factor`, not efficiency:

```text
p_battery(q) = 0.90 * min(p_AFDC_or_imputed(q), p_vehicle_cap(q))
charge_time_s = delta_energy_kwh / p_battery_kw * 3600
```

Pilot sensitivity reports factors 0.85/0.90/0.95; canonical artifacts store
only 0.90. Missing power uses a frozen national-by-mode median. If that median
is absent, generation is a hard error; vehicle-cap fallback is prohibited.

## 8. Pilot and release discipline

All V2 outputs go only to `Calibration_v2/` and `Instances_v2/`; V1 schemas and
outputs are never reused. Default maximum attempts per family is four and
every rejection has a structured reason.

Smoke tests use a new temporary directory. Calibration pilot covers only the
ten training cities and train/validation tracks. It must not read or generate
Test2/Jacksonville. Full 7,500-family generation is forbidden until the pilot
evidence is reviewed and explicitly approved.

The pilot handoff must include commit and changed-file summary; schema table;
test commands/count/time; H3 and PF support; four leakage assertions; source
mode/reuse and matched-vs-pool bias; fragmentation and CS reports; M1-M5/Q90;
attempt/timing/RSS; archive arbitrary-child-index restore; and explicit
confirmation that prohibited V1 schema/output/fallback/selector/prefix paths
were not used.
