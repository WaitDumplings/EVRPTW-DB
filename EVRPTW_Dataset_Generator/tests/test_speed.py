from __future__ import annotations

import networkx as nx
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import LineString

from evrptw_cle.speed import (
    _hpms_corridor_is_usable,
    _hpms_speed_is_usable,
    _select_directional_speed,
    build_legal_speed_layer,
    operating_mode_from_hpms,
    operating_mode_from_osm,
    parse_osm_maxspeed,
)


def test_hpms_speed_requires_direction_verified_evidence() -> None:
    corridor_only = {
        "match_confidence": "high",
        "corridor_match_usable": True,
        "direction_verified": False,
        "hpms_speed_usable": False,
    }
    assert _hpms_corridor_is_usable(corridor_only)
    assert not _hpms_speed_is_usable(corridor_only)
    assert _hpms_speed_is_usable(
        {
            **corridor_only,
            "direction_verified": True,
            "hpms_speed_usable": True,
        }
    )


def test_legal_speed_priority_osm_then_verified_hpms_then_imputation(
    tmp_path,
) -> None:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    for node, x in enumerate([-117.03, -117.02, -117.01, -117.00], start=1):
        graph.add_node(node, x=x, y=32.0)
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        attributes = {
            "osmid": 100 + u,
            "highway": "residential",
            "length": 100.0,
            "oneway": True,
            "reversed": False,
            "geometry": LineString(
                [
                    (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                    (graph.nodes[v]["x"], graph.nodes[v]["y"]),
                ]
            ),
        }
        if u == 1:
            attributes["maxspeed"] = "30 mph"
        graph.add_edge(u, v, key=0, **attributes)
    graph_path = tmp_path / "graph.graphml"
    ox.save_graphml(graph, graph_path)

    evidence_path = tmp_path / "hpms.parquet"
    pd.DataFrame(
        {
            "edge_id": ["1:2:0", "2:3:0", "3:4:0"],
            "F_SYSTEM": [7, 7, 7],
            "SPEED_LIMIT": [20, 25, 40],
            "match_confidence": ["high", "high", "high"],
            "corridor_match_usable": [True, True, True],
            "direction_verified": [True, True, False],
            "hpms_speed_usable": [True, True, False],
        }
    ).to_parquet(evidence_path, index=False)

    output_dir = tmp_path / "speed"
    build_legal_speed_layer(
        city_slug="test-city",
        graph_path=graph_path,
        output_dir=output_dir,
        hpms_edge_evidence_path=evidence_path,
    )
    speeds = pd.read_parquet(output_dir / "directed_legal_speeds.parquet").set_index(
        "edge_id"
    )
    assert speeds.loc["1:2:0", "speed_limit_source"] == "osm_maxspeed"
    assert speeds.loc["1:2:0", "speed_limit_kph"] == pytest.approx(48.28032)
    assert (
        speeds.loc["2:3:0", "speed_limit_source"]
        == "hpms_speed_limit_direction_verified"
    )
    assert speeds.loc["2:3:0", "speed_limit_kph"] == pytest.approx(40.2336)
    assert speeds.loc["3:4:0", "speed_limit_source"] == (
        "within_city_class_imputation"
    )
    assert speeds.loc["3:4:0", "speed_limit_kph"] == pytest.approx(40.2336)
    assert set(speeds["moves_road_type"]) == {"urban_unrestricted_access"}
    assert speeds.loc["1:2:0", "reference_speed_weekday_kph"] == pytest.approx(
        48.28032 * 0.545059133
    )
    assert speeds.loc["1:2:0", "reference_speed_weekend_kph"] == pytest.approx(
        48.28032 * 0.564078022
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
