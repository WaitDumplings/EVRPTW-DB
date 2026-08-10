from __future__ import annotations

import json

from evrptw_cle.nsi import (
    classify_service_location,
    iter_rfc7464_features,
    structure_group_key,
)


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
