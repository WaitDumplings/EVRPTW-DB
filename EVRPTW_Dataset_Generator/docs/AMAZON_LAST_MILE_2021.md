# Amazon Last Mile 2021 source contract

## Role in EVRPTW-DB

The Amazon Last Mile Routing Research Challenge 2021 data is a **Stage-2
calibration/template input**. It is not part of a CLE, and its obfuscated
coordinates are never transferred to an EVRPTW-DB city. The current adapter
uses the training-side route, stop, package, travel-time, planned-service-time,
and time-window fields to build a compact empirical artifact layer.

The three required upstream files are:

```text
model_build_inputs/
  route_data.json
  package_data.json
  travel_times.json
```

## Official download

The source is public in the Registry of Open Data on AWS. No AWS account or API
key is required, but AWS CLI must be installed.

From the repository root, download only the files used by Stage 2:

```bash
EVRPTW_Dataset_Generator/scripts/download_amazon_last_mile_2021.sh
```

The default destination is:

```text
EVRPTW_Dataset_Generator/data/sources/amazon-last-mile-2021/
  License.txt
  Readme.txt
  model_build_inputs/
    route_data.json
    package_data.json
    travel_times.json
```

An alternative destination may be supplied as the first argument. For an
already downloaded snapshot, set `AMAZON_MODEL_BUILD_INPUTS` to its
`model_build_inputs` directory.

The upstream command for downloading the complete challenge corpus is:

```bash
aws s3 sync --no-sign-request \
  s3://amazon-last-mile-challenges/almrrc2021/ ./data/
```

EVRPTW-DB does not need the complete evaluation/scoring corpus.

## What is stored and released

- **Raw Amazon JSON:** keep as a local/server source cache. It is ignored by
  Git and should not be committed to the code repository.
- **Compact Stage-2 artifact:** generated once under
  `EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3/`. It contains only the
  fields and empirical summaries consumed by the generator; it does not
  contain Amazon stop coordinates. Its manifest stores portable upstream
  object IDs and the public registry/license identifiers rather than a local
  machine's absolute source path.
- **Generated instances:** store template provenance and derived operational
  fields. They do not claim to be observed Amazon routes.

For a public benchmark release, the reproducibility package should include the
download script and can include the compact artifact as a versioned dataset
asset. The raw JSON need not be mirrored because the authoritative public S3
source is available. If the compact artifact or generated instances are
redistributed, preserve Amazon attribution, the source citation, the upstream
license notice, and a statement describing the transformations.

## License and attribution boundary

Amazon publishes the challenge material under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). That license is
distinct from the repository's eventual code license. The project must not
present the code license as relicensing Amazon-derived data, and commercial
reuse of the Amazon-derived artifact/instances requires a separate license
review or permission from the rights holder.

Authoritative references:

- Registry and license: <https://registry.opendata.aws/amazon-last-mile-challenges/>
- Dataset paper: <https://doi.org/10.1287/trsc.2022.1173>
- Amazon Science page:
  <https://www.amazon.science/publications/2021-amazon-last-mile-routing-research-challenge-data-set>
