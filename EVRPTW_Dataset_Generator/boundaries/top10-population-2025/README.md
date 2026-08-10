# Population-oriented ten-city boundaries

This directory freezes the 2025 U.S. Census city-proper boundaries used by the
EVRPTW-DB depot/customer cle cohort.

For every city:

- `admin_boundary.geojson` is the Census Place polygon used for graph
  membership;
- `land_boundary.geojson` is that Place minus intersecting Census AREAWATER and
  is used for facilities, customers, chargers, and map display;
- `metadata.json` records the Place GEOID, intersecting county GEOIDs, source
  archives, SHA-256 hashes, and land/water area cross-checks.

The cohort is New York City, Los Angeles, Chicago, Houston, Phoenix,
Philadelphia, San Antonio, San Diego, Dallas, and Fort Worth. City proper—not
the surrounding metropolitan area—is the service-boundary unit.

`manifest.json` is the cross-city checksum/provenance ledger. Runtime Census
archives live under `data/sources/census/tiger2025/` and are intentionally not
stored in Git.
