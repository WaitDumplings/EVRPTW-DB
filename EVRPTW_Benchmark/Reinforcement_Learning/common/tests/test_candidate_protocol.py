from __future__ import annotations

import numpy as np

from EVRPTW_Benchmark.Reinforcement_Learning.common.candidate_protocol import candidate_seed, independent_candidate_batch


def _solve(_instance, seed):
    rng = np.random.default_rng(seed)
    objective = rng.random(1)
    return {
        "success": np.asarray([True]),
        "objective_distance_km": objective,
        "served_customers": np.asarray([10]),
        "routes": [[[0, 1, 0]]],
    }, 0.0


def test_chunked_and_unchunked_candidate_union_are_identical() -> None:
    unchunked = independent_candidate_batch(
        [object()], candidate_count=50, candidate_chunk_size=50,
        base_seed=1234, instance_offset=9, solve_one=_solve,
    )
    chunked = independent_candidate_batch(
        [object()], candidate_count=50, candidate_chunk_size=5,
        base_seed=1234, instance_offset=9, solve_one=_solve,
    )
    np.testing.assert_array_equal(
        unchunked.infos[0]["objective_distance_km"],
        chunked.infos[0]["objective_distance_km"],
    )
    assert unchunked.infos[0]["routes"] == chunked.infos[0]["routes"]


def test_candidate_seeds_are_stable_and_unique() -> None:
    first = [candidate_seed(1234, 7, index) for index in range(50)]
    second = [candidate_seed(1234, 7, index) for index in range(50)]
    assert first == second
    assert len(set(first)) == 50
