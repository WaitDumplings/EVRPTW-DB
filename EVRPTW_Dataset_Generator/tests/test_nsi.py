from __future__ import annotations

import gzip
import io
import json
import urllib.error

from evrptw_cle.nsi import (
    _download_tile,
    classify_service_location,
    iter_rfc7464_features,
    structure_group_key,
)
from shapely.geometry import box, shape


def test_rfc7464_parser_handles_split_records() -> None:
    feature_one = {"type": "Feature", "properties": {"fd_id": 1}}
    feature_two = {"type": "Feature", "properties": {"fd_id": 2}}
    payload = (
        b"\x1e"
        + json.dumps(feature_one).encode()
        + b"\n\x1e"
        + json.dumps(feature_two).encode()
        + b"\n"
    )
    parsed = list(iter_rfc7464_features([payload[:17], payload[17:39], payload[39:]]))
    assert [item["properties"]["fd_id"] for item in parsed] == [1, 2]


def test_group_key_prefers_footprint_then_building_id() -> None:
    assert structure_group_key(
        {"ftprntid": "06037_1", "usastrucid": 8, "bid": "B", "fd_id": 2}
    ) == ("ftprntid:06037_1", "ftprntid")
    assert structure_group_key({"ftprntid": None, "usastrucid": None, "bid": "B", "fd_id": 2}) == (
        "bid:B",
        "bid",
    )


def test_location_type_uses_unit_bands_then_occupancy_fallback() -> None:
    assert classify_service_location(1, "RES1-1SNB") == (
        "house",
        "nsi_occtype_fallback",
    )
    assert classify_service_location(2, "RES1-1SNB")[0] == "small_apt"
    assert classify_service_location(7, "RES3C")[0] == "medium_apt"
    assert classify_service_location(60, "RES3F")[0] == "large_apt"
    assert classify_service_location(0, "RES2")[0] == "manufactured_home"


def test_nsi_persistent_5xx_uses_audited_quadrant_fallback(tmp_path, monkeypatch) -> None:
    class Response(io.BytesIO):
        status = 200

    calls = {"failed": 0, "successful": 0}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        body = json.loads(request.data)
        geometry = shape(body["features"][0]["geometry"])
        if geometry.area > 0.5:
            calls["failed"] += 1
            raise urllib.error.HTTPError(request.full_url, 500, "test", {}, None)
        calls["successful"] += 1
        feature = {
            "type": "Feature",
            "properties": {"fd_id": calls["successful"]},
            "geometry": None,
        }
        return Response(b"\x1e" + json.dumps(feature).encode() + b"\n")

    monkeypatch.setattr("evrptw_cle.nsi.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("evrptw_cle.nsi.time.sleep", lambda _: None)
    result = _download_tile(
        "tile",
        box(0.0, 0.0, 2.0, 2.0),
        tmp_path,
        "https://example.invalid/nsi",
        timeout_s=1,
        retries=2,
    )

    assert calls == {"failed": 10, "successful": 16}
    assert result["fallback_split"]["part_count"] == 16
    assert result["fallback_split"]["trigger_http_status"] == 500
    assert result["fallback_split"]["max_split_depth_used"] == 2
    assert (tmp_path / "tile.download.json").is_file()
    with gzip.open(tmp_path / "tile.geojsonseq.gz", "rb") as stream:
        parsed = list(iter_rfc7464_features([stream.read()]))
    assert [item["properties"]["fd_id"] for item in parsed] == list(range(1, 17))

    cached = _download_tile(
        "tile",
        box(0.0, 0.0, 2.0, 2.0),
        tmp_path,
        "https://example.invalid/nsi",
        timeout_s=1,
        retries=2,
    )
    assert cached["cached"] is True
    assert cached["fallback_split"] == result["fallback_split"]
