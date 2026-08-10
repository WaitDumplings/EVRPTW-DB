#!/usr/bin/env python3
"""Normalize an existing AFDC export to the frozen public EV-site contract."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evrptw_cle.preflight import AFDC_REQUIRED_COLUMNS
from evrptw_cle.util import sha256_file, write_json

FILTERS = {
    "Fuel Type Code": "ELEC",
    "Status Code": "E",
    "Access Code": "public",
    "Country": "US",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"AFDC source export is missing: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen AFDC snapshot: {output}")

    frame = pd.read_csv(source, low_memory=False)
    missing = AFDC_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"AFDC source export is missing columns: {sorted(missing)}")
    input_count = len(frame)
    for column, value in FILTERS.items():
        frame = frame.loc[frame[column].astype(str).eq(value)]
    if frame.empty:
        raise ValueError("AFDC source export produced no rows after declared filters")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="afdc-filter-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        frame.to_csv(staged, index=False)
        staged.replace(output)
    manifest = {
        "schema": "evrptw_afdc_local_snapshot_filter_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "row_count": input_count,
        },
        "filters": FILTERS,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "row_count": len(frame),
        },
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
