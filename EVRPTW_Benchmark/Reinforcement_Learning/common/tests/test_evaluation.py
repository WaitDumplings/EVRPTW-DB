from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import (
    select_min_verified_distance,
)


def test_selector_skips_shorter_environment_success_that_fails_verifier() -> None:
    info = {
        "success": np.asarray([True, True]),
        "objective_distance_km": np.asarray([4.0, 7.0]),
        "served_customers": np.asarray([2, 2]),
        "routes": [
            [[0, 1, 0]],
            [[0, 1, 2, 0]],
        ],
    }
    selected, routes, verification = select_min_verified_distance(
        _instance(), info
    )
    assert selected == 1
    assert routes == [[0, 1, 2, 0]]
    assert verification["passed"]
    assert verification["objective_distance_km"] == 7.0
