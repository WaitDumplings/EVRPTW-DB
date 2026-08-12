# Data sources and provenance contract

## U.S. reference sources

| Layer | Source | Required? | What the pipeline uses | Important limitation |
| --- | --- | --- | --- | --- |
| Administrative/service boundary | U.S. Census TIGER/Line Places and Area Hydrography | Included frozen assets | City proper and land-only mask | City proper is not a carrier service territory |
| Roads and OSM facilities | OpenStreetMap via Geofabrik PBF | Yes | Directed topology, tags, geometry, depots, OSM charging POIs | Tag completeness varies |
| Building geometry | Microsoft USBuildingFootprints | Yes | Polygon and area geometry | No authoritative residential type |
| Residential attributes | USACE National Structure Inventory (NSI) | Yes, queried automatically | Occupancy family, modeled units, structure identifiers and attributes | Modeled inventory, not parcel-order ground truth |
| Charging sites | NREL Alternative Fuels Data Center (AFDC) | Yes; free API key | Public/available EV site coordinates, connectors and port counts | Public site is not automatically fleet-compatible; power may be missing |
| Address anchor | U.S. Census Geocoder | Optional but recommended | Address-level coordinate QA | Not exact EVSE geometry |
| Road class/legal speed | FHWA HPMS | Optional | High-confidence functional class and missing legal-speed evidence | Requires explicit OSM conflation; local-road coverage is incomplete |
| Commercial-vehicle speed prior | NREL Fleet DNA report NREL/TP-5400-65921 | Built-in versioned profile | Three Average Driving Speed profile means | Mode-level prior, not Amazon/Rivian edge observation |
| Day/road-type speed-factor structure | U.S. EPA MOVES | Stage 2 U.S. adapter | Restricted/unrestricted access and weekday/weekend calibration strata | Current numerical factor table is still development calibration |
| Operating-day statistics | Amazon Last Mile Routing Research Challenge 2021 | Stage 2 only | Package volume, service-time and time-window aggregate calibration | Coordinates are obfuscated; no house/apartment label |
| Reference EV specification | Rivian Commercial Van Delivery 700 Reference Guide | Stage 2 U.S. adapter | 18.5 m3 cargo, 100 kWh battery, 257 km range, 11/100 kW AC/DC caps | Reference configuration, not route telemetry |
| Energy resource model | Classical EVRPTW literature | Stage 2 V1 contract | Constant `h = battery/range` and linear path-distance energy | Deliberately omits payload, weather, HVAC and speed dependence |

Official/source entry points:

- OpenStreetMap/Geofabrik: <https://download.geofabrik.de/north-america/us.html>
- Census TIGER/Line: <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>
- Census Geocoder: <https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html>
- Microsoft USBuildingFootprints: <https://github.com/microsoft/USBuildingFootprints>
- USACE NSI: <https://www.hec.usace.army.mil/confluence/nsi/>
- AFDC API: <https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/>
- FHWA HPMS: <https://www.fhwa.dot.gov/policyinformation/hpms.cfm>
- NREL report DOI: <https://doi.org/10.2172/1397153>
- EPA MOVES algorithms: <https://www.epa.gov/moves/moves-algorithms>
- Amazon ARCD paper: <https://doi.org/10.1287/trsc.2022.1173>
- Rivian Commercial Van Reference Guide:
  <https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf>
- Schneider et al. EVRPTW formulation:
  <https://doi.org/10.1287/trsc.2013.0490>

Stage-2 model details and the distinction between observed specifications,
calibrated priors, and benchmark assumptions are in
[STAGE2_INSTANCE_MODEL.md](STAGE2_INSTANCE_MODEL.md).

## Provenance rules

1. Raw source files are immutable within a profile version.
2. Routine research builds record versioned source/profile IDs and source
   references. A complete SHA-256 audit is required for the final release
   package, not for every exploratory run.
3. API responses are cached and reused. A later live response is a new source
   snapshot, not an in-place update; final-release caches are hashed.
4. Native fields are preserved beside canonical fields. Examples include OSM
   `highway`, HPMS `F_SYSTEM`, AFDC connector text, and NSI `occtype`.
5. Automatic coordinate resolution never deletes raw coordinates.
6. A missing value is not replaced with an unverifiable claim. In particular,
   missing charger power remains missing.
7. Profiles are versioned when a source snapshot, cohort, boundary, semantic
   rule, or crosswalk changes.

The profile does not require a human-written download date in every filename.
The source hash, upstream timestamp where available, API query, generation UTC,
and adapter version are the reproducibility record.

## License/release checklist

- Retain OpenStreetMap attribution and comply with ODbL for derived road data.
- Review Microsoft building-footprint license/attribution requirements for the
  released derivative.
- Preserve Census, USACE, NREL, and FHWA source notices in release metadata.
- Add an explicit code license before publishing the repository.
- Publish large source/generated artifacts through a versioned data host or
  GitHub Release with checksum manifests rather than ordinary Git blobs.
