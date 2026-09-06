from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_diagnostics import summarize_values


@pytest.mark.parametrize("tensor", [False, True])
def test_masked_summary_counts_only_active_finite_observations(tensor):
    values = np.asarray([[1.0, 2.0, 99.0], [3.0, np.nan, np.inf]])
    mask = np.asarray([[True, True, False], [True, True, True]])
    data = torch.tensor(values, requires_grad=True) if tensor else values
    report = summarize_values(data, mask)
    assert report["count"] == 5
    assert report["finite_count"] == 3
    assert report["nonfinite_count"] == 2
    assert report["mean"] == 2.0
    assert report["std"] == pytest.approx(np.std([1, 2, 3]))
    assert report["p05"] == pytest.approx(1.1)
    assert report["p50"] == 2.0
    assert report["p95"] == pytest.approx(2.9)
    assert report["min"] == 1.0 and report["max"] == 3.0
    assert report["quantile_sample_count"] == 3
    json.dumps(report, allow_nan=False)
    if tensor:
        assert data.grad is None and data.requires_grad


@pytest.mark.parametrize("values,mask", [([], None), ([float("nan")], None), ([3.0], [False])])
def test_empty_populations_are_explicit_and_json_safe(values, mask):
    report = summarize_values(values, mask)
    assert report["finite_count"] == 0 and report["mean"] is None
    assert report["std"] is None and report["quantile_sample_count"] == 0
    json.dumps(report, allow_nan=False)


def test_bounded_quantiles_keep_full_moments_and_do_not_change_rng():
    values = torch.arange(30_000, dtype=torch.float32, requires_grad=True)
    torch_rng = torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    report = summarize_values(values, max_quantile_samples=101)
    reference = summarize_values(values.detach().numpy(), max_quantile_samples=101)
    assert report["count"] == 30_000
    assert report["quantile_sample_count"] == 101
    assert report["mean"] == 14999.5
    for key in ("mean", "std", "p05", "p50", "p95", "min", "max"):
        assert report[key] == pytest.approx(reference[key])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert values.grad is None
    values.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_constant_population_and_invalid_quantile_budget():
    assert summarize_values(torch.tensor([7.0]))["std"] == 0.0
    with pytest.raises(ValueError, match="positive"):
        summarize_values([1.0], max_quantile_samples=0)


def test_chunked_tensor_matches_numpy_with_padding_nonfinite_and_empty_chunks():
    values = np.arange(800_000, dtype=np.float64).reshape(8, -1)
    mask = np.ones(values.shape, dtype=bool)
    mask[:3] = False
    values[4, :17] = np.nan
    values[5, 21] = np.inf
    expected = summarize_values(values, mask, max_quantile_samples=73)
    actual = summarize_values(torch.tensor(values, requires_grad=True), mask, max_quantile_samples=73)
    for key in ("count", "finite_count", "nonfinite_count", "mean", "std", "min", "p05", "p50", "p95", "max", "quantile_sample_count"):
        assert actual[key] == pytest.approx(expected[key])
    empty = summarize_values(torch.tensor(values), np.zeros_like(mask))
    assert empty["count"] == 0 and empty["mean"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_summary_matches_cpu_and_preserves_cuda_rng():
    values = torch.arange(5000.0, device="cuda", requires_grad=True)
    state = torch.cuda.get_rng_state().clone()
    report = summarize_values(values, values.remainder(2) == 0)
    expected = summarize_values(np.arange(5000.0), np.arange(5000) % 2 == 0)
    for key in ("count", "mean", "std", "p05", "p50", "p95", "min", "max"):
        assert report[key] == pytest.approx(expected[key])
    assert torch.equal(torch.cuda.get_rng_state(), state)
    assert values.grad is None
