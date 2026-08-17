# Stage-2 pilot terminal-connectivity audit

Date: 2026-08-17

> Historical evidence captured before the connectivity repair. The code repair
> is implemented and unit-tested, but no CLE rebuild or new pilot has been run.
> Review the proposed fix and post-approval pipeline in
> `STAGE2_CONNECTIVITY_REPAIR_AND_PIPELINE_REVIEW_ZH.md`.

## Decision

The 10-city pilot was stopped after 67 materialized families and 202 rejected
attempts. The observed failures are not caused by vehicle capacity, energy
consumption, or charging power. They occur before those checks, in directed
road-access closure.

No Test2/Jacksonville instance, full 7,500-family run, archive, restore, or
commit was started.

## Reproduced failure mechanism

1. `route_depot_star()` receives the depot plus the complete split customer
   roster. One non-finite depot/customer direction rejects the entire family.
2. Charger selection receives the depot, selected customers, and every
   candidate-eligible charger in the city. One non-finite terminal pair rejects
   the entire family before the energy-feasible charger roster is formed.
3. A retry changes the random seed but not these fixed bad terminals, so retry
   is ineffective for this failure class.

## Exact audit results

| City | Audited roster | Node-roundtrip bad | Turn-aware bad |
|---|---:|---:|---:|
| Houston | 341,511 train-pool customers | 1 | not applicable |
| Phoenix | 309,217 train-pool customers | 3 | not applicable |
| Los Angeles | 2,000 candidate chargers | 1 | 7 |

All audited bad terminals were nevertheless stored as `anchor_scc_id=S0001`,
`protected_roundtrip_eligible=true`, and candidate eligible.

### Houston

- Customer `msft_nsi_msft_usbf_houston_004823097`
- Coordinate: `(-95.55138812995706, 29.900405997847752)`
- Physical edge: `0eb9bf8e70bc1cf08760`, OSM way `15311389`
- One-way `secondary_link`, projection fraction `1.0`
- Depot-to-customer unreachable; customer-to-depot reachable

### Phoenix

- `msft_nsi_msft_usbf_phoenix_000722556`
- `msft_nsi_msft_usbf_phoenix_001951371`
- Both project at fraction `0.0` to one-way physical edge
  `8385227de5fba5b19bb0` (`West Leiber Place`, OSM way `396101715`).
- `msft_nsi_msft_usbf_phoenix_001738061` projects at fraction `0.0` to
  one-way physical edge `a03351e513704e05245b` (`West Grandview Road`, OSM
  way `262969271`).
- All three are reachable from the depot but cannot return.

### Los Angeles

- Charger `afdc_113090` is node-level and turn-aware return-unreachable. It is
  projected at fraction `0.0` to one-way `primary_link` physical edge
  `49d5b8b8196dac5710ba`, OSM way `48226731`.
- Chargers `afdc_159025`, `afdc_160987`, `afdc_176457`, `afdc_176458`,
  `afdc_176459`, and `afdc_179517` are node-roundtrip reachable but cannot
  return under the canonical turn-aware topology. All six project to physical
  edge `0d7838d9782599888a0`, OSM way `48144161`.

## Contract defect

`projection_scc_id()` labels an endpoint projection from only the endpoint at
which the projection lies. Stage-2 access semantics are directional: arrival
uses the projected edge from `u` toward the projection, while departure uses
the remainder toward `v`. A fraction-1 projection can therefore inherit the
SCC of `v` even when arrival through `u` is impossible; fraction 0 has the
dual failure. `protected_roundtrip_eligible` is currently just
`anchor_in_reference_scc` and does not test these two access directions.

The Stage-1 flag also does not evaluate the canonical turn-aware line graph,
so it cannot detect the six Los Angeles turn-only failures.

## Required repair before another pilot

1. Define projection eligibility from actual inbound and outbound access
   semantics, not one inherited endpoint label.
2. Add an exact terminal-connectivity preflight against the same runtime node
   and canonical turn-aware graphs used by Stage-2.
3. Quarantine or deterministically remap bad customers/chargers before family
   generation; record IDs and reasons in an audit ledger.
4. Make fixed-roster connectivity failures non-retryable and fail before
   expensive terminal closure.
5. Rebuild/repair CLE eligibility, rerun this audit, and only then start a new
   10-city train/validation pilot.
