from __future__ import annotations

from pathlib import Path

import pytest

from evrptw_cle.moves_speed import (
    load_moves_speed_profile,
    moves_road_type_from_hpms,
    moves_road_type_from_osm,
    speed_retention_factor,
)


def test_frozen_moves5_profile_is_internally_consistent() -> None:
    path = Path(__file__).parents[1] / "configs" / "us_moves5_speed_profile_v1.json"
    profile = load_moves_speed_profile(path)
    assert profile["source_database"] == "movesdb20241112"
    assert profile["source_type_id"] == 32
    assert speed_retention_factor(
        profile, road_type="urban_restricted_access", day_type="weekday"
    ) == pytest.approx(53.115367 / 70.0)
    assert speed_retention_factor(
        profile, road_type="urban_unrestricted_access", day_type="weekend"
    ) == pytest.approx(25.383511 / 45.0)


def test_moves_road_type_adapter_prefers_access_semantics() -> None:
    assert moves_road_type_from_hpms(1) == "urban_restricted_access"
    assert moves_road_type_from_hpms(4) == "urban_unrestricted_access"
    assert moves_road_type_from_osm("motorway") == "urban_restricted_access"
    assert moves_road_type_from_osm("motorway_link") == "urban_restricted_access"
    assert moves_road_type_from_osm("residential") == "urban_unrestricted_access"
