from __future__ import annotations

import argparse
import copy

import numpy as np
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.benchmark_drl_hotpaths import (
    compare_infos,
    run_benchmark,
)


@pytest.mark.parametrize("method", ["am_evrptw", "evrptw_rl", "drl_ts", "terran"])
def test_hotpath_diagnostic_preserves_routes_and_verifier(method):
    args = argparse.Namespace(
        method=method, device="cpu", customers=3, stations=1, batch_size=1,
        candidates=2, steps=12, embedding_dim=32, encoder_layers=1,
        repeats=1, warmup=0, cpu_threads=1, seed=1234, no_jit=True,
        dataset_path=None, family_root=None,
    )
    old_threads = torch.get_num_threads()
    try:
        report = run_benchmark(args)
    finally:
        torch.set_num_threads(old_threads)
    assert report["passed"]
    assert not report["training_performed"]
    assert report["input_kind"] == "synthetic_performance_fixture_not_release_data"
    result = report["results"][0]
    assert result["comparisons"][0]["routes_equal"]
    assert result["verifier_checks"][0]["selection_and_verifier_equal"]
    assert result["reference_rollout_wall_s"][0] > 0
    assert result["optimized_rollout_wall_s"][0] > 0
    # Timing is diagnostic: do not turn noisy microbenchmark speed into a test.


@pytest.mark.parametrize("change", ["route", "flag", "cost", "nan"])
def test_hotpath_diagnostic_rejects_output_mismatch(change):
    left = dict(
        routes=[[[0, 1, 0]]], success=np.array([True]),
        served_customers=np.array([1]), vehicle_count=np.array([1]),
        objective_distance_km=np.array([2.0]),
    )
    right = copy.deepcopy(left)
    if change == "route":
        right["routes"] = [[[0, 2, 0]]]
    elif change == "flag":
        right["success"][0] = False
    elif change == "cost":
        right["objective_distance_km"][0] += 0.01
    else:
        right["objective_distance_km"][0] = np.nan
    assert not compare_infos([left], [right])["passed"]
