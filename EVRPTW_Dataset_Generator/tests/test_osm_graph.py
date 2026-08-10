import osmnx as ox

from evrptw_cle.osm_graph import _edge_is_drivable, _ensure_useful_way_tags


def test_drive_filter_accepts_public_residential_and_primary() -> None:
    assert _edge_is_drivable({"highway": "residential"})
    assert _edge_is_drivable({"highway": "primary", "oneway": "yes"})


def test_drive_filter_matches_osmnx_exclusions() -> None:
    assert not _edge_is_drivable({"highway": "footway"})
    assert not _edge_is_drivable({"highway": "service"})
    assert not _edge_is_drivable({"highway": "residential", "access": "private"})
    assert not _edge_is_drivable({"highway": "residential", "motor_vehicle": "no"})
    assert not _edge_is_drivable({"highway": "residential", "motorcar": "no"})


def test_drive_filter_handles_simplified_list_attributes() -> None:
    assert _edge_is_drivable({"highway": ["footway", "residential"]})
    assert not _edge_is_drivable({"highway": ["footway", "path"]})


def test_required_directional_speed_tags_are_retained() -> None:
    _ensure_useful_way_tags()
    assert "maxspeed:forward" in ox.settings.useful_tags_way
    assert "maxspeed:backward" in ox.settings.useful_tags_way
    assert "maxspeed:hgv:forward" in ox.settings.useful_tags_way
