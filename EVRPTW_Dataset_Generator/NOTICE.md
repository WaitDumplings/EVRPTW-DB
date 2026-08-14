# Data and license notice

This repository's code license has not yet been selected. Add an explicit code
license before public release.

Generated road graphs are derived from OpenStreetMap and must retain
OpenStreetMap attribution and comply with the Open Database License (ODbL).
Boundary-provider metadata and source timestamps are stored with every city
output. If U.S. Census TIGER/Line boundaries are supplied as an override, their
source files and checksums should also be recorded.

Stage-2 calibration artifacts and generated instance fields are derived in
part from the 2021 Amazon Last Mile Routing Research Challenge Dataset,
copyright Amazon.com, Inc. or its affiliates. The upstream material is
provided under Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0). Preserve its attribution, license notice, source citation, and
a description of EVRPTW-DB's transformations when redistributing Amazon-derived
artifacts. The repository's eventual code license does not relicense the
Amazon source or its derivatives. See `docs/AMAZON_LAST_MILE_2021.md`.

Do not commit live API caches. Publish large GraphML/GPKG files through Git LFS
or versioned GitHub Release assets, together with a checksum manifest.
