#!/usr/bin/env python3
"""Download and freeze the public/available U.S. AFDC electric-station table."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evrptw_cle.preflight import AFDC_REQUIRED_COLUMNS
from evrptw_cle.util import sha256_file, write_json

API_URL = "https://developer.nrel.gov/api/alt-fuel-stations/v1/all.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sources/afdc/afdc_us_public_available_electric.csv"),
    )
    parser.add_argument("--api-key", default=os.environ.get("NREL_API_KEY"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or set NREL_API_KEY")
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite frozen AFDC snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "api_key": args.api_key,
        "fuel_type": "ELEC",
        "access": "public",
        "status": "E",
        "country": "US",
        "limit": "all",
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "evrptw-cle/1.0 source-freezer"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
    with tempfile.TemporaryDirectory(prefix="afdc-download-", dir=output.parent) as temp:
        downloaded = Path(temp) / "afdc.csv"
        downloaded.write_bytes(payload)
        frame = pd.read_csv(downloaded, low_memory=False)
        missing = AFDC_REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"AFDC response is missing columns: {sorted(missing)}")
        required_values = {
            "Fuel Type Code": "ELEC",
            "Status Code": "E",
            "Access Code": "public",
            "Country": "US",
        }
        for column, value in required_values.items():
            frame = frame.loc[frame[column].astype(str).eq(value)]
        if frame.empty:
            raise ValueError("AFDC download produced no records after declared filters")
        staged = Path(temp) / output.name
        frame.to_csv(staged, index=False)
        staged.replace(output)
    manifest = {
        "schema": "evrptw_afdc_snapshot_v1",
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "source_url": API_URL,
        "query": {key: value for key, value in params.items() if key != "api_key"},
        "record_count": len(frame),
        "output": str(output.resolve()),
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
