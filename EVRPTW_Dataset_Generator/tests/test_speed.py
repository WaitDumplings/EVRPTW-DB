from __future__ import annotations

import pandas as pd
import pytest

from evrptw_cle.speed import (
    _build_static_operational_scenarios,
    _select_directional_speed,
    operating_mode_from_hpms,
    operating_mode_from_osm,
    parse_osm_maxspeed,
)


def test_parse_osm_maxspeed_handles_mph_kph_and_multivalue() -> None:
    speed, status = parse_osm_maxspeed("25 mph")
    assert speed == pytest.approx(40.2336)
    assert status == "parsed_single"

    speed, status = parse_osm_maxspeed(["50 mph", "45 mph"])
    assert speed == pytest.approx(72.42048)
    assert status == "parsed_multivalue_conservative_min"

    speed, status = parse_osm_maxspeed("40 km/h")
    assert speed == pytest.approx(40.0)
    assert status == "parsed_single"


def test_parse_osm_maxspeed_quarantines_non_numeric_values() -> None:
    assert parse_osm_maxspeed(None) == (None, "missing")
    assert parse_osm_maxspeed("signals") == (None, "non_numeric_or_conditional")
    assert parse_osm_maxspeed("25 mph @ (school)") == (
        None,
        "non_numeric_or_conditional",
    )


def test_directional_speed_selection_follows_osm_way_direction() -> None:
    attrs = {
        "maxspeed": "35 mph",
        "maxspeed:forward": "30 mph",
        "maxspeed:backward": "25 mph",
        "maxspeed:hgv": "20 mph",
    }
    forward = _select_directional_speed({**attrs, "reversed": False}, "1:2:0")
    reverse = _select_directional_speed({**attrs, "reversed": True}, "2:1:0")
    assert forward["speed_limit_kph"] == pytest.approx(48.28032)
    assert forward["speed_limit_observed_source"] == "osm_maxspeed_forward"
    assert reverse["speed_limit_kph"] == pytest.approx(40.2336)
    assert reverse["speed_limit_observed_source"] == "osm_maxspeed_backward"
    assert reverse["hgv_speed_limit_kph_evidence"] == pytest.approx(32.18688)


def test_unparseable_directional_speed_falls_back_to_generic() -> None:
    selected = _select_directional_speed(
        {"reversed": False, "maxspeed:forward": "signals", "maxspeed": "30 mph"},
        "1:2:0",
    )
    assert selected["speed_limit_kph"] == pytest.approx(48.28032)
    assert selected["speed_limit_observed_source"] == "osm_maxspeed"
    assert selected["directional_maxspeed_present"] is True


def test_hmu_mode_mapping_prefers_explicit_functional_semantics() -> None:
    assert operating_mode_from_hpms(1) == "H"
    assert operating_mode_from_hpms("4") == "M"
    assert operating_mode_from_hpms(7.0) == "U"
    assert operating_mode_from_hpms(None) is None
    assert operating_mode_from_osm("motorway") == "H"
    assert operating_mode_from_osm("primary") == "M"
    assert operating_mode_from_osm("residential") == "U"
    assert operating_mode_from_osm("unexpected_new_tag") == "U"


def test_static_operational_scenarios_are_reproducible_capped_and_directional() -> None:
    frame = pd.DataFrame(
        {
            "edge_u": ["1", "2"],
            "edge_v": ["2", "1"],
            "edge_key": ["0", "0"],
            "edge_id": ["1:2:0", "2:1:0"],
            "physical_segment_id": ["segment", "segment"],
            "corridor_id": ["corridor", "corridor"],
            "direction_id": ["corridor|forward", "corridor|reverse"],
            "length_m": [100.0, 100.0],
            "highway": ["residential", "residential"],
            "road_group": ["local", "local"],
            "speed_limit_kph": [40.0, 40.0],
            "reference_speed_kph": [25.0, 25.0],
            "directional_variation_eligible": [True, True],
        }
    )
    kwargs = {
        "seed": 7,
        "scenarios_per_day_type": 1,
        "global_sigma": 0.02,
        "road_group_sigma": 0.04,
        "corridor_sigma": 0.05,
        "direction_sigma": 0.03,
        "factor_min": 0.75,
        "factor_max": 1.15,
    }
    first = _build_static_operational_scenarios(frame, **kwargs)
    second = _build_static_operational_scenarios(frame, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert set(first["day_type"]) == {"weekday", "weekend"}
    assert (first["speed_kph"] <= first["speed_limit_kph"]).all()
    paired = first.pivot(index="scenario_id", columns="edge_id", values="speed_kph")
    assert (paired["1:2:0"] != paired["2:1:0"]).all()
