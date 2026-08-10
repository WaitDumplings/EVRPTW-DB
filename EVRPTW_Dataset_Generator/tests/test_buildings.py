from __future__ import annotations

from evrptw_cle.buildings import (
    _bounds_overlap,
    _coordinate_bounds,
    _feature_from_line,
)


def test_feature_line_parser_accepts_trailing_comma() -> None:
    line = (
        b'{"type":"Feature","geometry":{"type":"Polygon","coordinates":'
        b"[[[-118.2,34.0],[-118.1,34.0],[-118.2,34.0]]]},"
        b'"properties":{"release":2}},\n'
    )
    feature = _feature_from_line(line)
    assert feature is not None
    assert feature["properties"]["release"] == 2


def test_non_feature_line_is_ignored() -> None:
    assert _feature_from_line(b'{"type":"FeatureCollection"}\n') is None


def test_coordinate_bounds_and_overlap() -> None:
    coordinates = [[[-118.4, 33.9], [-118.1, 34.2], [-118.4, 33.9]]]
    bounds = _coordinate_bounds(coordinates)
    assert bounds == (-118.4, 33.9, -118.1, 34.2)
    assert _bounds_overlap(bounds, (-118.3, 34.0, -118.2, 34.1))
    assert not _bounds_overlap(bounds, (-123.0, 37.0, -122.0, 38.0))
