# Data sources and provenance contract

## U.S. reference sources

| Layer | Source | Required? | What the pipeline uses | Important limitation |
| --- | --- | --- | --- | --- |
| Administrative/service boundary | U.S. Census TIGER/Line Places and Area Hydrography | Included frozen assets | City proper and land-only mask | City proper is not a carrier service territory |
| Roads and OSM facilities | OpenStreetMap via Geofabrik PBF | Yes | Directed topology, tags, geometry, depots, OSM charging POIs | Tag completeness varies |
| Building geometry | Microsoft USBuildingFootprints | Yes | Polygon and area geometry | No authoritative residential type |
| Residential attributes | USACE National Structure Inventory (NSI) | Yes, queried automatically | Occupancy family, modeled units, structure identifiers and attributes | Modeled inventory, not parcel-order ground truth |
| Charging sites | Alternative Fuels Data Center (AFDC), National Laboratory of the Rockies | Yes; free API key | Public/available EV site coordinates, connectors and port counts | Public site is not automatically fleet-compatible; power may be missing |
| Address anchor | U.S. Census Geocoder | Optional but recommended | Address-level coordinate QA | Not exact EVSE geometry |
| Road class/legal speed | FHWA HPMS | Required by the U.S. reference profile; replaceable by another country adapter | High-confidence functional class and direction-verified missing legal-speed evidence | Requires explicit OSM conflation; local-road coverage is incomplete |
| Commercial-vehicle operating-speed prior | U.S. EPA MOVES5 default database `movesdb20241112` | Compact derived profile is built in; raw SQL is optional for reproduction | National sourceTypeID 32 speed-bin distributions and hourly VMT fractions for urban restricted/unrestricted access and weekday/weekend | Not a road network, edge observation, city traffic count, or Rivian telemetry |
| Operating-day statistics | Amazon Last Mile Routing Research Challenge 2021 | Stage 2 only | Package volume, service-time and time-window aggregate calibration | Coordinates are obfuscated; no house/apartment label |
| Reference EV specification | Rivian Commercial Van Delivery 700 Reference Guide | Stage 2 U.S. adapter | 18.5 m3 cargo, 100 kWh battery, 257 km range, 11/100 kW AC/DC caps | Reference configuration, not route telemetry |
| Energy resource model | Classical EVRPTW literature | Stage 2 V1 contract | Constant `h = battery/range` and linear path-distance energy | Deliberately omits payload, weather, HVAC and speed dependence |

## Arbitrary U.S. city acquisition contract

`generate_us_city_cle.sh` automates sources with stable public download/API
interfaces: Census TIGER/Line, Geofabrik, Microsoft USBuildingFootprints, and
the FHWA 2018 HPMS FeatureServer. AFDC is also automatic, but the user must
provide a free NLR Developer Network API key when the shared national snapshot is not already
present. NSI is fetched and cached inside the existing CLE customer stage.
Amazon is not a CLE input and is therefore not downloaded by this command.

After downloading AFDC, the adapter extracts charging POIs from the selected
OSM PBF, batch-geocodes AFDC street addresses in the requested state through
the Census Geocoder, and runs the versioned coordinate resolver. The resolver
preserves the source latitude/longitude and records distinct evidence tiers.
A Census match corroborates the street-address anchor; it is not treated as an
observation of the exact charging-space or EVSE geometry. The city CLE then
applies its documented boundary, compatibility, road-projection, and SCC rules
to the resolved table.

The city name is resolved only within the named state against Census Places.
The generated `city_contract.json` records the GEOID, public URLs, paths,
profile, and output roots. The default OSM source is the state extract; an
explicit Geofabrik subregion or PBF URL is accepted only as a source adapter
override, not inferred from the city name.

The official FHWA endpoint currently exposed uniformly by state is the 2018
HPMS public spatial release. The downloader records that vintage and its
published caveats. It requests only the city window, while the matcher later
clips to the service boundary and applies the same confidence/direction rules
as the frozen cohort. A newer compatible state/FHWA service can be supplied
with `--hpms-service-url` and becomes a new source snapshot.

Official/source entry points:

- OpenStreetMap/Geofabrik: <https://download.geofabrik.de/north-america/us.html>
- Census TIGER/Line: <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>
- Census Geocoder: <https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html>
- Microsoft USBuildingFootprints: <https://github.com/microsoft/USBuildingFootprints>
- USACE NSI: <https://www.hec.usace.army.mil/confluence/nsi/>
- AFDC API: <https://developer.nlr.gov/docs/transportation/alt-fuel-stations-v1/>
- NLR developer-domain transition:
  <https://developer.nlr.gov/docs/nlr-domain-transition/>
- FHWA HPMS: <https://www.fhwa.dot.gov/policyinformation/hpms.cfm>
- EPA MOVES algorithms: <https://www.epa.gov/moves/moves-algorithms>
- EPA MOVES5 population/activity methods:
  <https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P101CUN7.TXT>
- Frozen MOVES default database:
  <https://github.com/USEPA/EPA_MOVES_Model/blob/master/database/Setup/movesdb20241112.zip>
- FHWA free-flow/off-peak definitions:
  <https://ops.fhwa.dot.gov/perf_measurement/ucr/documentation.htm>
- Amazon public dataset registry and download:
  <https://registry.opendata.aws/amazon-last-mile-challenges/>
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
- Preserve Census, USACE, NREL AFDC, EPA MOVES, and FHWA source notices in
  release metadata.
- Keep Amazon raw JSON outside Git. Preserve Amazon attribution and CC BY-NC
  4.0 terms for redistributed compact artifacts and generated data derived
  from those templates; see [AMAZON_LAST_MILE_2021.md](AMAZON_LAST_MILE_2021.md).
- Add an explicit code license before publishing the repository.
- Publish large source/generated artifacts through a versioned data host or
  GitHub Release with checksum manifests rather than ordinary Git blobs.
