from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

GENERATOR_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = GENERATOR_ROOT / "scripts"


def _load_preparer():
    sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location(
        "prepare_us11_sources", SCRIPTS_ROOT / "prepare_us11_sources.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_us11_source_plan_is_complete_and_deduplicated() -> None:
    module = _load_preparer()
    plan = module.source_plan()

    assert len(plan) == 36
    assert Counter(item["layer"] for item in plan) == {
        "osm_pbf": 7,
        "microsoft_buildings": 7,
        "hpms_city_window": 11,
        "afdc_raw": 1,
        "afdc_census_evidence": 1,
        "afdc_resolved": 1,
        "osm_charging_pois": 1,
        "census_block_groups": 7,
    }
    paths = [item["path"] for item in plan]
    assert len(paths) == len(set(paths))
    assert all(Path(path).is_absolute() for path in paths)


def test_hpms_registry_uses_bounded_city_sources() -> None:
    registry = json.loads(
        (GENERATOR_ROOT / "configs/us_11city_hpms_sources_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        city: item["source_stem"] for city, item in registry["cities"].items()
    } == {city: city for city in registry["cities"]}
