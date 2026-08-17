#!/usr/bin/env python3
"""Freeze national connector-compatible AFDC power medians for Stage-2 V2."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evrptw_cle.util import sha256_file


API_URL = "https://developer.nlr.gov/api/alt-fuel-stations/v1/ev-charging-units.csv"


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    ).dropna()
    frame = frame.loc[frame["value"].gt(0) & frame["weight"].gt(0)].sort_values(
        "value", kind="stable"
    )
    if frame.empty:
        raise ValueError("NATIONAL_MODE_MEDIAN_UNAVAILABLE: no positive observations")
    cutoff = float(frame["weight"].sum()) / 2.0
    return float(frame.loc[frame["weight"].cumsum().ge(cutoff), "value"].iloc[0])


def build_registry(snapshot: Path) -> dict[str, Any]:
    frame = pd.read_csv(snapshot, low_memory=False)
    specifications = {
        "ac_level2": ("EV J1772 Power Output (kW)", "EV J1772 Connector Count"),
        "dc_fast": ("EV CCS Power Output (kW)", "EV CCS Connector Count"),
    }
    missing = {
        column
        for columns in specifications.values()
        for column in columns
        if column not in frame
    }
    if missing:
        raise ValueError(f"AFDC charging-unit snapshot lacks: {sorted(missing)}")
    medians: dict[str, float] = {}
    evidence: dict[str, Any] = {}
    for mode, (power_column, count_column) in specifications.items():
        power = pd.to_numeric(frame[power_column], errors="coerce")
        count = pd.to_numeric(frame[count_column], errors="coerce")
        usable = power.gt(0) & count.gt(0)
        medians[mode] = _weighted_median(power.loc[usable], count.loc[usable])
        evidence[mode] = {
            "compatible_connector": "J1772" if mode == "ac_level2" else "CCS/J1772COMBO",
            "charging_unit_observation_count": int(usable.sum()),
            "connector_port_weight": int(count.loc[usable].sum()),
            "power_column": power_column,
            "weight_column": count_column,
        }
    return {
        "schema": "evrptw_national_charging_power_medians_v1",
        "source": "NLR_AFDC_electric_vehicle_charging_ports_API",
        "source_url": API_URL,
        "source_snapshot": snapshot.name,
        "source_snapshot_sha256": sha256_file(snapshot),
        "selection": {
            "fuel_type": "ELEC",
            "access": "public",
            "status": "E",
            "country": "US",
            "compatibility": "reference_vehicle_connector_only",
            "statistic": "connector_port_count_weighted_median",
        },
        "national_mode_medians_kw": medians,
        "evidence": evidence,
    }


def _download(output: Path, api_key: str) -> None:
    params = {
        "api_key": api_key,
        "fuel_type": "ELEC",
        "access": "public",
        "status": "E",
        "country": "US",
        "limit": "all",
    }
    request = urllib.request.Request(
        f"{API_URL}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "evrptw-cle/2.0 source-freezer"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
    with tempfile.TemporaryDirectory(prefix="afdc-units-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.write_bytes(payload)
        pd.read_csv(staged, nrows=1)
        staged.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NLR_API_KEY") or os.environ.get("NREL_API_KEY"),
    )
    parser.add_argument("--download-if-missing", action="store_true")
    args = parser.parse_args()
    if not args.snapshot.is_file():
        if not args.download_if_missing or not args.api_key:
            raise FileNotFoundError(args.snapshot)
        _download(args.snapshot, args.api_key)
    payload = build_registry(args.snapshot)
    payload["generated_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
