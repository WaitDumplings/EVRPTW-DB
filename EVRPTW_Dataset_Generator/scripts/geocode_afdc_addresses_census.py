#!/usr/bin/env python3
"""Geocode AFDC street addresses with the public U.S. Census batch geocoder.

The output is address-level QA evidence.  It is not treated as an exact charger
parking-space coordinate by the downstream resolver.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from evrptw_cle.util import sha256_file, write_json

ENDPOINT = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
EXPECTED_AFDC_COLUMNS = {"ID", "Street Address", "City", "State", "ZIP"}


def _input_csv(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    for row in frame.itertuples(index=False):
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _parse_response(payload: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for values in csv.reader(io.StringIO(payload)):
        if not values:
            continue
        padded = values + [""] * (8 - len(values))
        afdc_id, input_address, match, match_type, matched_address, coordinates, tigerline, side = (
            padded[:8]
        )
        longitude = latitude = None
        if coordinates:
            pieces = coordinates.split(",", maxsplit=1)
            if len(pieces) == 2:
                try:
                    longitude, latitude = float(pieces[0]), float(pieces[1])
                except ValueError:
                    longitude = latitude = None
        rows.append(
            {
                "afdc_id": int(float(afdc_id)),
                "census_input_address": input_address,
                "census_match_status": match.strip().lower() or "unknown",
                "census_match_type": match_type,
                "census_matched_address": matched_address,
                "census_longitude": longitude,
                "census_latitude": latitude,
                "census_tigerline_id": tigerline,
                "census_side": side,
            }
        )
    return pd.DataFrame(rows)


def geocode_chunk(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    timeout_s: float,
    retries: int,
) -> pd.DataFrame:
    payload = _input_csv(frame[["ID", "Street Address", "City", "State", "ZIP"]])
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                ENDPOINT,
                data={"benchmark": benchmark},
                files={"addressFile": ("afdc_addresses.csv", payload, "text/csv")},
                timeout=timeout_s,
            )
            response.raise_for_status()
            parsed = _parse_response(response.text)
            if len(parsed) != len(frame):
                raise RuntimeError(
                    f"Census response returned {len(parsed)} rows for {len(frame)} inputs"
                )
            return parsed
        except (requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afdc", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sources/afdc/afdc_census_address_anchors.csv"),
    )
    parser.add_argument("--benchmark", default="Public_AR_Current")
    parser.add_argument("--chunk-size", type=int, default=9000)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.chunk_size < 1 or args.chunk_size > 10_000:
        parser.error("--chunk-size must be between 1 and 10000")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen Census results: {args.output}")
    afdc = pd.read_csv(args.afdc, low_memory=False)
    missing = EXPECTED_AFDC_COLUMNS - set(afdc.columns)
    if missing:
        raise ValueError(f"AFDC input is missing address columns: {sorted(missing)}")
    if afdc["ID"].duplicated().any():
        raise ValueError("AFDC ID values must be unique")

    outputs: list[pd.DataFrame] = []
    for start in range(0, len(afdc), args.chunk_size):
        chunk = afdc.iloc[start : start + args.chunk_size]
        print(f"CENSUS {start}:{start + len(chunk)}", flush=True)
        outputs.append(
            geocode_chunk(
                chunk,
                benchmark=args.benchmark,
                timeout_s=args.timeout_s,
                retries=args.retries,
            )
        )
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    status_counts = (
        result["census_match_status"].value_counts(dropna=False).to_dict()
        if not result.empty
        else {}
    )
    manifest = {
        "schema": "evrptw_afdc_census_address_anchors_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_endpoint": ENDPOINT,
        "benchmark": args.benchmark,
        "row_count": len(result),
        "match_status_counts": status_counts,
        "input": {"path": str(args.afdc.resolve()), "sha256": sha256_file(args.afdc)},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "semantic_limit": (
            "Census coordinates represent matched street-address anchors and are not "
            "claimed as exact charger parking-space coordinates."
        ),
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
