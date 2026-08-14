#!/usr/bin/env python3
"""Download an official FHWA 2018 HPMS city-window extract from ArcGIS REST."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd


def _request_json(url: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None
    headers = {"User-Agent": "evrptw-cle/0.4 US-city-adapter"}
    if parameters is not None:
        payload = urllib.parse.urlencode(parameters).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url, data=payload, headers=headers)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
            if "error" in result:
                raise RuntimeError(f"ArcGIS REST error: {result['error']}")
            return result
        except Exception as error:  # pragma: no cover - network retry path
            last_error = error
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("ArcGIS request failed") from last_error


def _layer_url(service_url: str) -> tuple[str, dict[str, Any]]:
    service = _request_json(f"{service_url.rstrip('/')}?f=pjson")
    layers = service.get("layers", [])
    feature_layers = [item for item in layers if item.get("type") == "Feature Layer"]
    if len(feature_layers) != 1:
        raise RuntimeError(
            f"Expected one HPMS Feature Layer at {service_url}; found {feature_layers}"
        )
    url = f"{service_url.rstrip('/')}/{feature_layers[0]['id']}"
    return url, _request_json(f"{url}?f=pjson")


def _city_envelope(boundary_path: Path, padding_deg: float) -> dict[str, float]:
    boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = (float(value) for value in boundary.total_bounds)
    return {
        "xmin": xmin - padding_deg,
        "ymin": ymin - padding_deg,
        "xmax": xmax + padding_deg,
        "ymax": ymax + padding_deg,
    }


def download_hpms_city(
    *,
    service_url: str,
    boundary_path: Path,
    output_path: Path,
    force: bool = False,
    padding_deg: float = 0.25,
) -> dict[str, Any]:
    """Download a bounded HPMS extract with reproducible source/version metadata."""

    output_path = output_path.resolve()
    if output_path.exists() and not force:
        print(f"REUSE {output_path}", flush=True)
        source_manifest = output_path.with_suffix(".source.json")
        if source_manifest.is_file():
            return json.loads(source_manifest.read_text(encoding="utf-8"))
        return {
            "schema": "evrptw_hpms_city_source_v1",
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "reused_without_manifest": True,
        }

    layer_url, layer = _layer_url(service_url)
    object_id = str(layer.get("objectIdField") or layer.get("objectIdFieldName") or "")
    if not object_id:
        raise RuntimeError(f"HPMS layer does not declare an object ID: {layer_url}")
    max_records = int(layer.get("maxRecordCount") or 2000)
    envelope = _city_envelope(boundary_path, padding_deg)
    common = {
        "where": "1=1",
        "geometry": json.dumps(envelope, separators=(",", ":")),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",
    }
    count_result = _request_json(
        f"{layer_url}/query",
        {**common, "returnCountOnly": "true", "f": "json"},
    )
    expected_count = int(count_result.get("count", 0))
    if expected_count <= 0:
        raise RuntimeError(
            f"Official HPMS service returned no road segment in {boundary_path}"
        )

    features: list[dict[str, Any]] = []
    offset = 0
    while offset < expected_count:
        page = _request_json(
            f"{layer_url}/query",
            {
                **common,
                "outFields": "*",
                "returnGeometry": "true",
                "orderByFields": object_id,
                "resultOffset": str(offset),
                "resultRecordCount": str(max_records),
                "f": "geojson",
            },
        )
        page_features = page.get("features", [])
        if not page_features:
            break
        features.extend(page_features)
        offset += len(page_features)
        print(f"HPMS {len(features):,}/{expected_count:,}", flush=True)

    if len(features) != expected_count:
        raise RuntimeError(
            f"HPMS pagination returned {len(features)} of {expected_count} expected features"
        )
    property_keys = {
        str(key).casefold()
        for feature in features[: min(50, len(features))]
        for key in (feature.get("properties") or {})
    }
    if "f_system" not in property_keys:
        raise RuntimeError("HPMS extract does not contain F_SYSTEM")
    payload = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_suffix(output_path.suffix + ".part")
    staged.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    staged.replace(output_path)
    manifest = {
        "schema": "evrptw_hpms_city_source_v1",
        "dataset": "FHWA 2018 HPMS Public Release",
        "service_url": service_url,
        "layer_url": layer_url,
        "layer_name": layer.get("name"),
        "boundary_path": str(boundary_path.resolve()),
        "query_envelope_wgs84": envelope,
        "query_padding_degrees": padding_deg,
        "feature_count": len(features),
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "integrity_mode": "metadata_only_research",
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "caveat": (
            "Official 2018 HPMS public spatial release; intended for functional-class "
            "and conditional posted-speed evidence, not as a routing network."
        ),
    }
    output_path.with_suffix(".source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding-deg", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = download_hpms_city(
        service_url=args.service_url,
        boundary_path=args.boundary,
        output_path=args.output,
        force=args.force,
        padding_deg=args.padding_deg,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
