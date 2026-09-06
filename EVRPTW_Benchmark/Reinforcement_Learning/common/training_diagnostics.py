"""Detached, RNG-free summaries for training diagnostics, never loss inputs.

Moments and extrema cover every finite, unmasked observation. Quantiles use
at most 8192 evenly spaced observations, explicitly reported as a sample.
For GPU tensors only the scalar summary is copied to the host.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

_TENSOR_CHUNK_SIZE = 262_144


def _large_tensor_summary(values: torch.Tensor, mask: Any, budget: int) -> dict[str, Any]:
    """Two bounded passes avoid full-size masked and float64 GPU temporaries."""
    flat = values.detach().reshape(-1)
    active = None if mask is None else torch.broadcast_to(
        torch.as_tensor(mask, device=flat.device, dtype=torch.bool), values.shape,
    ).reshape(-1)
    dtype = torch.float32 if flat.device.type == "mps" else torch.float64
    pieces: list[tuple[int, int, int]] = []
    moments, counts = [], []
    observation_count = 0
    for start in range(0, flat.numel(), _TENSOR_CHUNK_SIZE):
        data = flat[start:start + _TENSOR_CHUNK_SIZE]
        if active is not None:
            data = data[active[start:start + _TENSOR_CHUNK_SIZE]]
        observation_count += data.numel()
        finite = data[torch.isfinite(data)].to(dtype=dtype)
        size = finite.numel()
        pieces.append((start, observation_count, size))
        if size:
            variance, mean = torch.var_mean(finite, unbiased=False)
            moments.append(torch.stack((mean, variance, finite.min(), finite.max())))
            counts.append(size)
        del finite, data
    result = summarize_values([])
    result["count"] = observation_count
    result["finite_count"] = sum(counts)
    result["nonfinite_count"] = observation_count - result["finite_count"]
    if not counts:
        return result
    merged = torch.stack(moments)
    weights = merged.new_tensor(counts) / sum(counts)
    mean = (weights * merged[:, 0]).sum()
    variance = (weights * (merged[:, 1] + (merged[:, 0] - mean).square())).sum()
    sample_count = min(sum(counts), budget)
    positions = np.linspace(0, sum(counts) - 1, sample_count).round().astype(np.int64)
    samples = []
    offset = 0
    for start, _count, size in pieces:
        lo, hi = np.searchsorted(positions, [offset, offset + size])
        if hi > lo:
            data = flat[start:start + _TENSOR_CHUNK_SIZE]
            if active is not None:
                data = data[active[start:start + _TENSOR_CHUNK_SIZE]]
            finite = data[torch.isfinite(data)]
            indices = torch.as_tensor(positions[lo:hi] - offset, device=flat.device)
            samples.append(finite[indices].to(dtype=dtype))
            del data, finite
        offset += size
    sample = torch.cat(samples)
    quantiles = torch.quantile(sample, sample.new_tensor([0.05, 0.50, 0.95]))
    numbers = torch.stack((mean, variance.sqrt(), merged[:, 2].min(),
                           *quantiles.unbind(), merged[:, 3].max())).cpu().tolist()
    for key, number in zip(("mean", "std", "min", "p05", "p50", "p95", "max"), numbers):
        result[key] = float(number) if np.isfinite(number) else None
    result["quantile_sample_count"] = sample_count
    return result


def summarize_values(
    values: Any, mask: Any = None, max_quantile_samples: int = 8192,
) -> dict[str, Any]:
    """Summarize without gradients, input mutation, or training RNG consumption.

``count`` includes nonfinite observations, but excludes masked padding.
Nonfinite values are counted rather than allowed to poison the statistics;
an empty finite population has JSON-safe null statistics. Standard deviation
is the population standard deviation, not an unbiased sample estimator.
"""
    if int(max_quantile_samples) < 1:
        raise ValueError("max_quantile_samples must be positive")
    result: dict[str, Any] = {
        "count": 0, "finite_count": 0, "nonfinite_count": 0,
        **{key: None for key in ("mean", "std", "min", "p05", "p50", "p95", "max")},
        "quantile_sample_count": 0,
        "quantile_method": "linear_on_evenly_spaced_finite_observations",
    }
    with torch.no_grad():
        if isinstance(values, torch.Tensor):
            if values.numel() > _TENSOR_CHUNK_SIZE:
                return _large_tensor_summary(values, mask, int(max_quantile_samples))
            data = values.detach()
            if mask is not None:
                active = torch.as_tensor(mask, device=data.device, dtype=torch.bool)
                data = data[torch.broadcast_to(active, data.shape)]
            data = data.reshape(-1)
            result["count"] = data.numel()
            # Double moments prevent float32 square overflow in diagnostics.
            # MPS does not support float64; CUDA/CPU do.
            dtype = torch.float32 if data.device.type == "mps" else torch.float64
            finite = data[torch.isfinite(data)].to(dtype=dtype)
            size = finite.numel()
            result["finite_count"] = size
            result["nonfinite_count"] = result["count"] - size
            if not size:
                return result
            count = min(size, int(max_quantile_samples))
            if count < size:
                positions = torch.linspace(0, size - 1, count, device=finite.device, dtype=dtype)
                sample = finite[positions.round().long()]
            else:
                sample = finite
            variance, mean = torch.var_mean(finite, unbiased=False)
            quantiles = torch.quantile(sample, sample.new_tensor([0.05, 0.50, 0.95]))
            summary = torch.stack((mean, variance.sqrt(), finite.min(), *quantiles.unbind(), finite.max()))
            numbers = summary.cpu().tolist()
        else:
            data = np.asarray(values)
            if mask is not None:
                active = np.asarray(mask, dtype=bool)
                data = data[np.broadcast_to(active, data.shape)]
            data = data.reshape(-1)
            result["count"] = int(data.size)
            finite = data[np.isfinite(data)].astype(np.float64, copy=False)
            size = int(finite.size)
            result["finite_count"] = size
            result["nonfinite_count"] = result["count"] - size
            if not size:
                return result
            count = min(size, int(max_quantile_samples))
            positions = np.linspace(0, size - 1, count).round().astype(np.int64)
            sample = finite[positions] if count < size else finite
            numbers = [finite.mean(), finite.std(), finite.min(),
                       *np.quantile(sample, [0.05, 0.50, 0.95]), finite.max()]
        result["quantile_sample_count"] = count
        for key, number in zip(("mean", "std", "min", "p05", "p50", "p95", "max"), numbers):
            result[key] = float(number) if np.isfinite(number) else None
    return result
